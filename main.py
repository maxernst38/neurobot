import argparse
import asyncio
import json
import os
import select
import sys
import termios
import tty

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_model", "so101_new_calib.urdf")
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

STEP_M = 0.01        # end-effector move per keypress, in meters
GRIPPER_STEP = 2.0   # gripper move per keypress, in percent (0-100 scale)
GRIPPER_LIMITS = (0, 100)

# If a move takes the target outside the arm's reachable workspace, the IK
# solver stops converging and starts returning large, unstable joint jumps
# instead of failing cleanly. Reject any move whose solution doesn't actually
# reach the target within this tolerance, so the target just stops at the
# workspace boundary instead of drifting further and destabilizing the solve.
IK_ERROR_TOLERANCE_M = 0.02

# Safety cap: max degrees any joint may move in a single send_action call,
# regardless of how far the IK solution jumps.
MAX_RELATIVE_TARGET = 5.0

# key -> (xyz axis index, direction); frame is the URDF's gripper_frame_link axes.
# Sign/axis conventions may need flipping once tested against the real arm.
TRANSLATION_KEYS = {
    "w": (0, 1),   # +x
    "s": (0, -1),  # -x
    "d": (1, 1),   # +y
    "a": (1, -1),  # -y
    "r": (2, 1),   # +z (up)
    "f": (2, -1),  # -z (down)
}
GRIPPER_KEYS = {"e": 1, "q": -1}  # open / close

# --- webcam control mode (arm_motion_tracker/) ---
# Direct joint-space position control driven by the browser's predicted joint
# state, instead of IK-based cartesian jogging. The tracker classifies each
# human joint into one of three bands - "high"/"mid"/"low" (e.g. extended /
# neutral / flexed) - and each control tick, every robot joint eases toward
# the preset target for its corresponding metric's current band, moving at
# most JOINT_STEP_DEG per tick so it can't jump.
WS_PORT = 8765
CONTROL_HZ = 20
JOINT_STEP_DEG = 1.5  # max degrees moved per tick, i.e. ~JOINT_STEP_DEG * CONTROL_HZ deg/sec top speed
GRIPPER_TICK_STEP = 2.0  # gripper's equivalent of JOINT_STEP_DEG, in percent

# Maps arm_motion_tracker's six tracked human-joint metrics to this robot's joints.
METRIC_TO_JOINT = {
    "shoulder_rotation": "shoulder_pan",
    "shoulder_flexion": "shoulder_lift",
    "elbow_flexion": "elbow_flex",
    "wrist_flexion": "wrist_flex",
    "wrist_rotation": "wrist_roll",
}

# Sign of each joint's "high" target. Flip an entry to -1 if that motor's
# high/low targets come out swapped (moves toward flexed when the tracker
# says extended, etc.) once tested on the real arm.
JOINT_SIGN = {
    "shoulder_pan": 1,
    "shoulder_lift": 1,
    "elbow_flex": 1,
    "wrist_flex": 1,
    "wrist_roll": 1,
}

# Degree limits per joint, converted from robot_model/so101_new_calib.urdf's
# radian <limit> tags.
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}

# The "high"/"low" preset targets use this fraction of each joint's full
# range rather than the hard limit itself, leaving mechanical margin.
TARGET_FRACTION = 0.6


def _joint_state_targets():
    targets = {}
    for name, (low, high) in JOINT_LIMITS_DEG.items():
        sign = JOINT_SIGN[name]
        high_extreme, low_extreme = (high, low) if sign == 1 else (low, high)
        targets[name] = {
            "high": high_extreme * TARGET_FRACTION,
            "mid": 0.0,
            "low": low_extreme * TARGET_FRACTION,
        }
    return targets


JOINT_STATE_TARGETS_DEG = _joint_state_targets()

# gripper_pos convention (see GRIPPER_KEYS above): 0 = closed, 100 = open.
# "high" grip state = gripping = closed; "low" = open hand.
GRIPPER_STATE_TARGETS = {"high": 0.0, "mid": 50.0, "low": 100.0}


def read_key(timeout=0.02):
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def run_keyboard(follower):
    kinematics = RobotKinematics(URDF_PATH, target_frame_name="gripper_frame_link", joint_names=ARM_JOINTS)

    print("Controls: w/s=forward/back  a/d=left/right  r/f=up/down  e/q=gripper open/close")
    print("Ctrl+C to quit.")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        obs = follower.get_observation()
        joint_deg = np.array([obs[f"{name}.pos"] for name in ARM_JOINTS])
        gripper_pos = obs["gripper.pos"]
        target_pose = kinematics.forward_kinematics(joint_deg)

        while True:
            key = read_key()
            if key in TRANSLATION_KEYS:
                axis, direction = TRANSLATION_KEYS[key]
                candidate_pose = target_pose.copy()
                candidate_pose[axis, 3] += direction * STEP_M

                # Warm-started from the current solution, this converges within a few
                # loop iterations for reachable moves rather than in a single call.
                candidate_joint_deg = kinematics.inverse_kinematics(joint_deg, candidate_pose)
                reached = kinematics.forward_kinematics(candidate_joint_deg)
                if np.linalg.norm(reached[:3, 3] - candidate_pose[:3, 3]) <= IK_ERROR_TOLERANCE_M:
                    target_pose = candidate_pose
                    joint_deg = candidate_joint_deg
                # else: unreachable, ignore this move and keep the last good target/joints
            elif key in GRIPPER_KEYS:
                low, high = GRIPPER_LIMITS
                gripper_pos = max(low, min(high, gripper_pos + GRIPPER_KEYS[key] * GRIPPER_STEP))

            action = {f"{name}.pos": val for name, val in zip(ARM_JOINTS, joint_deg)}
            action["gripper.pos"] = gripper_pos
            follower.send_action(action)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def run_webcam(follower):
    from aiohttp import web

    obs = follower.get_observation()
    joint_deg = {name: obs[f"{name}.pos"] for name in ARM_JOINTS}
    gripper_pos = obs["gripper.pos"]

    # Targets start pinned to the arm's current pose (not the "mid" preset),
    # so nothing moves until the browser actually reports a joint's state.
    joint_target = dict(joint_deg)
    gripper_target = gripper_pos

    latest_states = {}
    lock = asyncio.Lock()

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("Webcam tracker connected")
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                async with lock:
                    latest_states.clear()
                    latest_states.update(data)
        finally:
            async with lock:
                latest_states.clear()  # hold the last commanded target if the browser disconnects
            print("Webcam tracker disconnected")
        return ws

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WS_PORT)
    await site.start()
    print(f"Webcam control bridge listening on ws://0.0.0.0:{WS_PORT}/ws")
    print("Point arm_motion_tracker's robot-connect field at this host. Ctrl+C to quit.")

    try:
        while True:
            async with lock:
                states = dict(latest_states)

            for metric_key, joint_name in METRIC_TO_JOINT.items():
                state = states.get(metric_key)
                if state in ("high", "mid", "low"):
                    joint_target[joint_name] = JOINT_STATE_TARGETS_DEG[joint_name][state]
                # else ("none" or missing): no fresh reading, keep easing toward the last target

            grip_state = states.get("grip")
            if grip_state in ("high", "mid", "low"):
                gripper_target = GRIPPER_STATE_TARGETS[grip_state]

            for name in ARM_JOINTS:
                low, high = JOINT_LIMITS_DEG[name]
                step = max(-JOINT_STEP_DEG, min(JOINT_STEP_DEG, joint_target[name] - joint_deg[name]))
                joint_deg[name] = max(low, min(high, joint_deg[name] + step))

            low, high = GRIPPER_LIMITS
            step = max(-GRIPPER_TICK_STEP, min(GRIPPER_TICK_STEP, gripper_target - gripper_pos))
            gripper_pos = max(low, min(high, gripper_pos + step))

            action = {f"{name}.pos": joint_deg[name] for name in ARM_JOINTS}
            action["gripper.pos"] = gripper_pos
            follower.send_action(action)

            await asyncio.sleep(1 / CONTROL_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", choices=["keyboard", "webcam"], default="keyboard",
        help="keyboard: IK cartesian jog via keypresses. "
             "webcam: joints ease toward extended/neutral/flexed presets driven by the "
             "arm_motion_tracker webapp over a websocket.",
    )
    args = parser.parse_args()

    follower = SO101Follower(SO101FollowerConfig(
        port="/dev/ttyACM0", id="my_follower",
        use_degrees=True,  # kinematics/joint limits work in real joint degrees, not normalized percent
        max_relative_target=MAX_RELATIVE_TARGET,
    ))
    follower.connect()

    try:
        if args.input == "keyboard":
            run_keyboard(follower)
        else:
            asyncio.run(run_webcam(follower))
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
