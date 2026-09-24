#!/usr/bin/env python
"""Start the robot bridge and the camera tracker together.

Camera-driven control needs two processes: main.py owns the arm and serves
the websocket bridge, tracker.py owns the cameras and connects to it. They
stay separate on purpose -- if the tracker dies, main.py sees the disconnect
and homes the arm within a second, which a single process could not do for
itself. This just saves opening two terminals.

    python run_teleop.py                    # auto-detect the serial port
    python run_teleop.py --port COM3 --fps 60
    python run_teleop.py --window           # old OpenCV popup instead of the page

The tracker serves its UI at http://localhost:8080/ -- open that once both
processes are up. Pass --web-host 0.0.0.0 to reach it from another device on
the network, which is worth doing: the page is what you read while standing
in front of the cameras, and a phone can be propped where you can see it.

Ctrl+C stops both. The children deliberately share this console rather than
having their output piped through here: a console Ctrl+C reaches every
process attached to it, so both run their own cleanup -- the tracker closes
its cameras, main.py disconnects the arm. Piping the output to prefix it
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
# Must match WS_PORT in main.py.
WS_PORT = 8765
# How long to wait before suggesting that the bridge may be sat at a prompt.
HINT_AFTER_S = 20.0


def ask_to_stop(port):
    """Ask a child to shut itself down, and say whether it accepted.

    Windows has no SIGTERM. Popen.terminate() is TerminateProcess, which
    stops a process where it stands and runs none of its cleanup -- for
    main.py that means follower.disconnect() never happens, so the servos
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
    """Block until main.py is accepting connections, or until it exits.

    Starting the tracker first would only make it fail its initial connect:
    main.py has to reach the arm before it ever opens the socket. Polling the
    port rather than parsing stdout keeps this working whatever main.py
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

    robot_cmd = [sys.executable, "-u", os.path.join(HERE, "main.py"), "--input", "webcam"]
    for flag in ("port", "id"):
        if getattr(args, flag):
            robot_cmd += [f"--{flag}", str(getattr(args, flag))]

    tracker_cmd = [sys.executable, "-u", TRACKER,
                   "--robot", "--robot-host", f"localhost:{WS_PORT}"]
    for flag in ("fps", "side", "camera", "camera2", "hands"):
        if getattr(args, flag) is not None:
            tracker_cmd += [f"--{flag}", str(getattr(args, flag))]
    if args.no_views:
        tracker_cmd.append("--no-views")
    if args.trust_inferred:
        tracker_cmd.append("--trust-inferred")
    if args.record:
        tracker_cmd.append("--record")
    if args.record_note:
        tracker_cmd += ["--record-note", args.record_note]
    if not args.window:
        tracker_cmd.append("--web")
        for flag in ("web_port", "web_host"):
            if getattr(args, flag) is not None:
                tracker_cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]

    web_port = None if args.window else (args.web_port or 8080)

    if bridge_is_up(WS_PORT):
        raise SystemExit(
            f"Something is already listening on port {WS_PORT} -- most likely a main.py\n"
            f"from an earlier run. Stop it before starting another."
        )

    robot = tracker = None
    try:
        print("Starting the robot bridge...")
        robot = subprocess.Popen(robot_cmd, cwd=HERE)
        if not wait_for_bridge(robot, WS_PORT):
            raise SystemExit(f"\nThe robot bridge exited ({robot.returncode}) before it "
                             f"started listening. See its output above.")

        print(f"\nBridge is up on {WS_PORT}. Starting the tracker...\n")
        tracker = subprocess.Popen(tracker_cmd, cwd=HERE)

        # Whichever stops first, stop the other -- a tracker with no robot, or
        # an engaged robot with no tracker, is never what you wanted.
        while True:
            if robot.poll() is not None:
                print(f"\nRobot bridge exited ({robot.returncode}); stopping the tracker.")
                break
            if tracker.poll() is not None:
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
        print("Both stopped.")


if __name__ == "__main__":
    main()
