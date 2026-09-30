#!/usr/bin/env python
"""Start the robot bridge and the camera tracker together.

Camera-driven control needs two processes: robot.py owns the arm and serves
the websocket bridge, tracker.py owns the cameras and connects to it. They
stay separate on purpose -- if the tracker dies, robot.py sees the disconnect
and homes the arm within a second, which a single process could not do for
itself. This just saves opening two terminals.

    python main.py                    # auto-detect the serial port
    python main.py --port COM3 --fps 60
    python main.py --window           # old OpenCV popup instead of the page
    python main.py --no-robot         # cameras only, no arm attached
    python main.py --num-cameras 1    # one camera (no triangulation)
    python main.py --record --record-note "grip test"   # save the session
    python main.py --record --record-video --mask-face block
    python main.py --replay session-20260930-142211.jsonl   # review a capture

The tracker serves its UI at http://localhost:8080/ -- open that once both
processes are up. Pass --web-host 0.0.0.0 to reach it from another device on
the network, which is worth doing: the page is what you read while standing
in front of the cameras, and a phone can be propped where you can see it.

Ctrl+C stops both. The children deliberately share this console rather than
having their output piped through here: a console Ctrl+C reaches every
process attached to it, so both run their own cleanup -- the tracker closes
its cameras, robot.py disconnects the arm. Piping the output to prefix it
would mean putting the children in their own process group, and then the
only way to stop them on Windows is TerminateProcess, which skips cleanup
entirely and leaves the arm holding torque on a port that was never closed.
Legible output is not worth that trade.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER = os.path.join(HERE, "arm_motion_tracker", "tracker.py")
# Must match WS_PORT in robot.py.
WS_PORT = 8765
# How long to wait before suggesting that the bridge may be sat at a prompt.
HINT_AFTER_S = 20.0


def ask_to_stop(port):
    """Ask a child to shut itself down, and say whether it accepted.

    Windows has no SIGTERM. Popen.terminate() is TerminateProcess, which
    stops a process where it stands and runs none of its cleanup -- for
    robot.py that means follower.disconnect() never happens, so the servos
    keep their torque and the arm stays stiff with the serial port still
    open. Both children serve HTTP already, so asking politely costs one
    request and is the difference between an arm that goes limp and one
    that does not.

    This matters most in the case nobody thinks about: quitting from the
    tracker page. Then there is no Ctrl+C anywhere, and without this the
    robot bridge would sit there until it was killed for not stopping.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/shutdown", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def bridge_is_up(port):
    with socket.socket() as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_bridge(proc, port):
    """Block until robot.py is accepting connections, or until it exits.

    Starting the tracker first would only make it fail its initial connect:
    robot.py has to reach the arm before it ever opens the socket. Polling the
    port rather than parsing stdout keeps this working whatever robot.py
    prints.

    Deliberately no deadline. On an arm with no calibration file, connecting
    runs lerobot's interactive calibration first -- you move every joint
    through its range, which takes as long as it takes. Any timeout short
    enough to catch a genuinely stuck bridge would fire in the middle of
    that. The child is sharing this console, so it can say what it is waiting
    for, and Ctrl+C is the way out.
    """
    started = time.monotonic()
    hinted = False
    while proc.poll() is None:
        if bridge_is_up(port):
            return True
        if not hinted and time.monotonic() - started > HINT_AFTER_S:
            hinted = True
            print("\n  (still waiting on the robot bridge -- if this arm has never been\n"
                  "   calibrated, it is prompting for that above. Ctrl+C to give up.)\n")
        time.sleep(0.25)
    return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="robot serial port (default: auto-detect)")
    parser.add_argument("--id", help="calibration id for the arm")
    parser.add_argument("--fps", type=int, help="camera frame rate to request")
    parser.add_argument("--side", choices=["LEFT", "RIGHT"], help="arm to track")
    parser.add_argument("--camera", type=int, help="index of the front camera")
    parser.add_argument("--camera2", type=int, help="index of the side camera")
    parser.add_argument("--num-cameras", type=int, choices=[0, 1, 2],
                        metavar="N",
                        help="how many cameras to open: 2 (default), 1, or 0")
    # Tracking without the arm: checking calibration, recording a session to
    # replay later, or working on the camera side while the robot is packed
    # away. The tracker is the half that runs standalone -- robot.py exists to
    # hold the arm, so there is nothing to start without one.
    parser.add_argument("--no-robot", action="store_true",
                        help="run only the tracker, without starting the robot bridge")
    parser.add_argument("--no-views", action="store_true", help="hide the 3D projections")
    parser.add_argument("--trust-inferred", action="store_true",
                        help="triangulate every joint, labelling rather than dropping the "
                             "ones the cameras did not clearly see")
    parser.add_argument("--hands", choices=["metrics", "all", "off"],
                        help="which cameras run the hand model (default: only the one driving grip)")
    # Capturing a teleop run is the case worth recording without having to
    # remember to press anything: what the cameras saw, and what the robot
    # was told, for the session you are about to do rather than the next one.
    parser.add_argument("--record", action="store_true",
                        help="record the session from the moment tracking starts")
    parser.add_argument("--record-note", help="note stored in the recording's header")
    parser.add_argument("--record-name", metavar="NAME",
                        help="filename for --record (default: session-<timestamp>.jsonl)")
    parser.add_argument("--record-compress", action="store_true",
                        help="gzip the recording (~20x smaller)")
    parser.add_argument("--record-video", action="store_true",
                        help="also keep each camera's raw frames as video (~15-40 MB/min per camera)")
    parser.add_argument("--mask-face", choices=["off", "blur", "block"],
                        help="obscure the face in recorded video and drop face landmarks; "
                             "'block' for anonymity (see tracker.py --help)")
    parser.add_argument("--video-codec", help="fourcc for --record-video (default avc1)")
    parser.add_argument("--recordings", metavar="DIR", help="directory recordings are written to")
    # Reviewing a capture is a tracker-only job, so --replay implies
    # --no-robot rather than being something to combine with it: there is no
    # live arm pose to follow, and driving the arm from a recording would
    # move it to poses nobody is standing in front of the cameras to stop.
    # The tracker opens no cameras for a replay either, which is what makes
    # this work on a machine nowhere near the rig.
    parser.add_argument("--replay", metavar="FILE",
                        help="review a recording instead of the cameras, no arm and no "
                             "cameras (implies --no-robot); a bare filename is looked up "
                             "in --recordings")
    parser.add_argument("--derive", choices=["stored", "points", "cameras"],
                        help="how a replayed frame is read: 'stored' as it was recorded, "
                             "'points' recomputed from the recorded 3D, 'cameras' "
                             "re-triangulated from the recorded 2D with the calibration "
                             "loaded now (default: points; also switchable on the page)")
    # The tracker UI is a browser page by default now. The OpenCV window it
    # replaced could not be resized without rescaling the video with it, and
    # put the alignment readout -- the thing you stand and read while holding
    # a pose in front of the cameras -- at 0.4-scale cv2.putText.
    parser.add_argument("--window", action="store_true",
                        help="use the old OpenCV popup window instead of the browser UI")
    parser.add_argument("--web-port", type=int, help="port for the tracker UI (default 8080)")
    parser.add_argument("--web-host",
                        help="interface for the tracker UI; 0.0.0.0 to open it on a phone or tablet")
    args = parser.parse_args()

    # See the note on --replay. Forced here, before anything reads no_robot,
    # so neither the bridge nor the tracker's robot link is set up at all.
    if args.replay:
        args.no_robot = True
        # Recording a replay would write a file derived from another one. The
        # tracker refuses it from the page, but --record is applied before the
        # recording loads, where it would open a capture that then receives no
        # frames -- an empty file, and no sign of why.
        if args.record:
            parser.error("--record and --replay contradict each other: a replay has "
                         "nothing live to capture. Drop one.")
    elif args.derive:
        parser.error("--derive says how to read a replayed frame, so it only means "
                     "something with --replay.")

    robot_cmd = [sys.executable, "-u", os.path.join(HERE, "robot.py"), "--input", "webcam"]
    for flag in ("port", "id"):
        if getattr(args, flag):
            robot_cmd += [f"--{flag}", str(getattr(args, flag))]

    tracker_cmd = [sys.executable, "-u", TRACKER]
    if not args.no_robot:
        tracker_cmd += ["--robot", "--robot-host", f"localhost:{WS_PORT}"]
    for flag in ("fps", "side", "camera", "camera2", "num_cameras", "hands"):
        if getattr(args, flag) is not None:
            tracker_cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    if args.no_views:
        tracker_cmd.append("--no-views")
    if args.trust_inferred:
        tracker_cmd.append("--trust-inferred")
    for flag in ("record", "record_compress", "record_video"):
        if getattr(args, flag):
            tracker_cmd.append(f"--{flag.replace('_', '-')}")
    for flag in ("record_note", "record_name", "mask_face", "video_codec", "recordings",
                 "replay", "derive"):
        if getattr(args, flag):
            tracker_cmd += [f"--{flag.replace('_', '-')}", getattr(args, flag)]
    if not args.window:
        tracker_cmd.append("--web")
        for flag in ("web_port", "web_host"):
            if getattr(args, flag) is not None:
                tracker_cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]

    web_port = None if args.window else (args.web_port or 8080)

    if not args.no_robot and bridge_is_up(WS_PORT):
        raise SystemExit(
            f"Something is already listening on port {WS_PORT} -- most likely a robot.py\n"
            f"from an earlier run. Stop it before starting another."
        )

    robot = tracker = None
    try:
        if args.replay:
            print(f"Replaying {args.replay}: no cameras are opened and the arm is "
                  f"not being driven.\n")
        elif args.no_robot:
            print("Starting the tracker only (--no-robot); the arm is not being "
                  "driven.\n")
        else:
            print("Starting the robot bridge...")
            robot = subprocess.Popen(robot_cmd, cwd=HERE)
            if not wait_for_bridge(robot, WS_PORT):
                raise SystemExit(f"\nThe robot bridge exited ({robot.returncode}) before it "
                                 f"started listening. See its output above.")
            print(f"\nBridge is up on {WS_PORT}. Starting the tracker...\n")
        tracker = subprocess.Popen(tracker_cmd, cwd=HERE)

        # Whichever stops first, stop the other -- a tracker with no robot, or
        # an engaged robot with no tracker, is never what you wanted. With
        # --no-robot there is only the tracker to wait on.
        while True:
            if robot is not None and robot.poll() is not None:
                print(f"\nRobot bridge exited ({robot.returncode}); stopping the tracker.")
                break
            if tracker.poll() is not None:
                if robot is None:
                    print(f"\nTracker exited ({tracker.returncode}).")
                else:
                    print(f"\nTracker exited ({tracker.returncode}); stopping the robot bridge.")
                break
            time.sleep(0.3)
    except KeyboardInterrupt:
        # The console delivered the same Ctrl+C to both children already, so
        # there is nothing to forward -- just stop printing and let them run
        # their own shutdown below.
        print("\nStopping...")
    finally:
        # Ask first. Ctrl+C reaches both children by itself, but every other
        # way out of the loop above -- quitting from the tracker page, either
        # child exiting on its own -- reaches neither, and the only thing
        # left would be to kill them.
        if tracker is not None and tracker.poll() is None and web_port:
            ask_to_stop(web_port)
        if robot is not None and robot.poll() is None:
            if not ask_to_stop(WS_PORT):
                print("  the robot bridge did not answer; it may be left holding torque")
        for proc in (tracker, robot):
            if proc is None or proc.poll() is not None:
                continue
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Only reached when a child ignored the interrupt, or when it
                # was the *other* child that exited and this one never saw a
                # Ctrl+C at all. Abrupt, but it has had its chance.
                print(f"  forcing {os.path.basename(proc.args[2])} to stop")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Stopped." if robot is None else "Both stopped.")


if __name__ == "__main__":
    main()
