import argparse
import asyncio
import json
import os
import sys
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import msvcrt
else:
    import select
    import termios
    import tty

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

GRIPPER_STEP = 2.0   # gripper move per keypress, in percent (0-100 scale)
GRIPPER_LIMITS = (0, 100)

# Safety backstop: max degrees a goal may sit away from the measured
# position in a single send_action call, enforced by lerobot.
#
# This is deliberately NOT the slew limit -- JOINT_STEP_DEG below is, and it
# is nearly an order of magnitude tighter. At 5.0 this backstop was instead
# the binding constraint on how hard a joint could be asked to hold, which
# is not what a backstop is for: shoulder_lift needs more than 5 degrees of
# lead to hold its own weight, so the cap silently prevented it from ever
# reaching a target it was perfectly capable of holding.
MAX_RELATIVE_TARGET = 15.0

# Degrees per keypress. Small by default, because a held-down key repeats --
# the step is as much a speed as a distance. Adjustable live with [ and ].
JOG_STEP_DEG = 2.0
JOG_STEP_RANGE = (0.25, 15.0)

# Two adjacent keyboard rows: the top row drives a joint one way and the row
# directly below it drives the same joint the other way, in ARM_JOINTS order
# with the gripper last. Physical adjacency is the whole mnemonic -- there is
# nothing to memorise beyond "upper row one way, lower row the other".
JOG_KEYS_UP = "asdfgh"
JOG_KEYS_DOWN = "zxcvbn"

# The SO-101 controller board presents a WCH USB-serial bridge (CH340 on
# older boards, CH343 on newer). Matching the vendor rather than one chip's
# product id keeps port detection working across board revisions.
WCH_VENDOR_ID = 0x1A86

# --- webcam control mode (arm_motion_tracker/tracker.py) ---
# Joint-space position control driven by tracker.py's triangulated joint
# angles. The tracker streams continuous angles; JOINT_MAP scales each onto
# one robot joint, and every tick the arm eases toward that target by at most
# JOINT_STEP_DEG from its *measured* position, so a command can neither jump
# nor drift away from where the hardware actually is.
WS_PORT = 8765
CONTROL_HZ = 20
JOINT_STEP_DEG = 1.5  # max degrees moved per tick, i.e. ~JOINT_STEP_DEG * CONTROL_HZ deg/sec top speed
GRIPPER_TICK_STEP = 2.0  # gripper's equivalent of JOINT_STEP_DEG, in percent

# Degree limits per joint, converted from robot_model/so101_new_calib.urdf's
# radian <limit> tags.
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}

# Fraction of each joint's mechanical range that the mapping is allowed to
# command. At 1.0 every pose the arm can physically hold is also a pose you
# can line your body up with, which is what the alignment handshake needs:
# the target posture is derived from wherever the arm actually is, so any
# joint parked outside the mapped span produces a target angle outside human
# range and can never be matched. Lower it to trade workspace for a gentler
# ratio of robot motion to your own -- the action loop clamps to
# JOINT_LIMITS_DEG regardless, so this is about feel, not safety.
TARGET_FRACTION = 1.0

# lerobot's own SO101Follower.configure() halves every motor's P_Coefficient
# from the servo's default of 32 down to 16, to keep the lightly-loaded
# joints from oscillating. elbow_flex is not lightly loaded -- it alone
# carries the forearm, wrist and gripper against gravity -- and at half gain
# it lacks the torque authority to close the last stretch toward a target,
# so it stalls short instead of reaching it. Restoring just this joint to
# the servo's stock value after connect fixes that without touching the
# other joints' (correctly) gentler tuning.
ELBOW_P_COEFFICIENT = 32

# Continuous map from each tracked human metric onto one robot joint: the
# human range maps linearly onto the robot range, and both endpoints are
# explicit so a joint can be reversed simply by writing its robot range
# backwards. That replaces the old separate JOINT_SIGN table -- direction is
# now part of the same declaration as magnitude, and cannot disagree with it.
#
# The human ranges are the usable span of each metric as tracker.py defines
# it (see compute_joint_metrics), not full anatomical range: mapping the last
# few degrees of a joint most people cannot reach comfortably just wastes
# robot travel.
#
# CAVEAT: shoulder_rotation and wrist_rotation are still expressed in camera
# 1's axes rather than a body frame, so their zero depends on where that
# camera sits. Moving a camera changes what they mean, and the neutral pose
# has to be re-captured. The other three are frame-independent.
JOINT_MAP = {
    "shoulder_rotation": {"joint": "shoulder_pan", "human": (-60.0, 60.0)},
    "shoulder_flexion": {"joint": "shoulder_lift", "human": (0.0, 150.0)},
    "elbow_flexion": {"joint": "elbow_flex", "human": (30.0, 180.0)},
    "wrist_flexion": {"joint": "wrist_flex", "human": (-70.0, 70.0)},
    "wrist_rotation": {"joint": "wrist_roll", "human": (-90.0, 90.0)},
}


# Curl span for grip, in the same spirit as JOINT_MAP's human ranges.
GRIP_RANGE = (0.0, 1.0)

ARM_RANGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "arm_ranges.json")

# A recorded range narrower than this is rejected. The mapping divides by the
# span, so a span near zero is a near-infinite gain: a degree of arm movement
# would throw the joint across its whole travel. In practice it means the
# recording never actually moved that joint, and keeping the default is the
# right answer.
MIN_HUMAN_SPAN_DEG = 20.0
MIN_GRIP_SPAN = 0.15


def apply_arm_ranges(data, source="file"):
    """Replace the declared human ranges with measured ones.

    The counterpart of --recalibrate for the human side. JOINT_MAP's default
    ranges are assumptions about a generic arm; every one that is wrong shows
    up as the wrong gain, so a joint saturates before you reach the end of
    your reach or never gets near the robot's limits. Measuring them removes
    the assumption.

    Each entry is validated on its own and a bad one falls back to the
    default rather than rejecting the file: a recording where you forgot to
    rotate your wrist should still fix the four joints you did move.

    Returns the list of human-readable notes about what was applied.
    """
    global GRIP_RANGE
    notes = []
    for metric, spec in JOINT_MAP.items():
        pair = data.get(metric)
        if pair is None:
            continue
        try:
            lo, hi = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError, IndexError):
            notes.append(f"{metric}: not a pair of numbers, keeping default")
            continue
        if hi - lo < MIN_HUMAN_SPAN_DEG:
            notes.append(f"{metric}: recorded span {hi - lo:.0f} deg is under "
                         f"{MIN_HUMAN_SPAN_DEG:.0f}, keeping default "
                         f"{spec['human'][0]:+.0f}..{spec['human'][1]:+.0f}")
            continue
        spec["human"] = (lo, hi)
        notes.append(f"{metric}: {lo:+.0f}..{hi:+.0f} deg")

    pair = data.get("grip")
    if pair is not None:
        try:
            lo, hi = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError, IndexError):
            notes.append("grip: not a pair of numbers, keeping default")
        else:
            if hi - lo < MIN_GRIP_SPAN:
                notes.append(f"grip: recorded span {hi - lo:.2f} is under "
                             f"{MIN_GRIP_SPAN}, keeping default")
            else:
                GRIP_RANGE = (lo, hi)
                notes.append(f"grip: {lo:.2f}..{hi:.2f}")
    if notes:
        print(f"Arm ranges from {source}:")
        for note in notes:
            print(f"    {note}")
    return notes


def load_arm_ranges():
    """Apply config/arm_ranges.json if it exists. Never fatal."""
    if not os.path.exists(ARM_RANGES_FILE):
        print(f"No recorded arm ranges ({ARM_RANGES_FILE}); using the generic "
              f"defaults. Record yours from the tracker page.")
        return
    try:
        with open(ARM_RANGES_FILE) as f:
            data = json.load(f)
    except (ValueError, OSError) as exc:
        print(f"Ignoring {ARM_RANGES_FILE}: {exc}")
        return
    apply_arm_ranges(data, source=os.path.basename(ARM_RANGES_FILE))


def _robot_span(joint):
    lo, hi = JOINT_LIMITS_DEG[joint]
    return lo * TARGET_FRACTION, hi * TARGET_FRACTION


def map_metric_to_joint(metric, value):
    """One human metric onto its robot joint, clamped to the usable range."""
    spec = JOINT_MAP[metric]
    h_lo, h_hi = spec["human"]
    r_lo, r_hi = spec.get("robot") or _robot_span(spec["joint"])
    frac = (value - h_lo) / (h_hi - h_lo)
    return min(max(r_lo + frac * (r_hi - r_lo), min(r_lo, r_hi)), max(r_lo, r_hi))


def map_grip(value):
    """Grip curl onto gripper percent (0 closed .. 100 open).

    GRIP_RANGE is the curl span a hand actually covers, the same declaration
    JOINT_MAP makes for the angles. 0..1 is the theoretical span of the
    ratio, not one any hand reaches: fingers do not straighten to a perfect
    line or curl to a perfect point, so an uncalibrated grip uses only the
    middle of the gripper's travel.
    """
    g_lo, g_hi = GRIP_RANGE
    lo, hi = GRIPPER_LIMITS
    frac = (value - g_lo) / (g_hi - g_lo)
    return min(hi, max(lo, hi - frac * (hi - lo)))


NEUTRAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "neutral_pose.json")


def load_neutral():
    """The startup pose, as (joints, gripper), or (None, None) if unset.

    Clamped to JOINT_LIMITS_DEG and warned about rather than rejected. An
    earlier version refused to start on a pose outside a derated span, which
    meant a pose the arm could physically hold -- and had been jogged to by
    hand -- could still stop the whole program before it connected. A startup
    pose is not worth refusing to run over; the worst a clamp costs is a few
    degrees, and it says so.
    """
    if not os.path.exists(NEUTRAL_FILE):
        return None, None
    try:
        with open(NEUTRAL_FILE) as f:
            data = json.load(f)
    except (ValueError, OSError) as exc:
        print(f"Ignoring {NEUTRAL_FILE}: {exc}")
        return None, None
    if not all(j in data for j in ARM_JOINTS) or "gripper" not in data:
        print(f"Ignoring {NEUTRAL_FILE}: missing joints; expected "
              f"{', '.join(ARM_JOINTS)} and gripper.")
        return None, None

    joints = {}
    for name in ARM_JOINTS:
        lo, hi = JOINT_LIMITS_DEG[name]
        value = float(data[name])
        joints[name] = min(hi, max(lo, value))
        if joints[name] != value:
            print(f"Neutral pose: {name} {value:+.1f} clamped to "
                  f"{joints[name]:+.1f} (limit {lo:+.1f} to {hi:+.1f}).")
    lo, hi = GRIPPER_LIMITS
    return joints, min(hi, max(lo, float(data["gripper"])))


def save_neutral(joint_deg, gripper_pos):
    data = {j: round(float(joint_deg[j]), 2) for j in ARM_JOINTS}
    data["gripper"] = round(float(gripper_pos), 2)
    os.makedirs(os.path.dirname(NEUTRAL_FILE), exist_ok=True)
    with open(NEUTRAL_FILE, "w") as f:
        json.dump(data, f, indent=2)


def invert_metric(metric, robot_value):
    """The human angle that JOINT_MAP would send to this joint value.

    The inverse of map_metric_to_joint. This is what makes "the human matches
    the robot" work without a step at engage: the alignment target is the
    posture that maps *onto* the pose the arm is already holding, so at the
    moment you engage, the new command equals the current position.
    """
    spec = JOINT_MAP[metric]
    h_lo, h_hi = spec["human"]
    r_lo, r_hi = spec.get("robot") or _robot_span(spec["joint"])
    return h_lo + (robot_value - r_lo) / (r_hi - r_lo) * (h_hi - h_lo)


def invert_grip(gripper_percent):
    g_lo, g_hi = GRIP_RANGE
    lo, hi = GRIPPER_LIMITS
    return g_lo + (hi - gripper_percent) / (hi - lo) * (g_hi - g_lo)


def human_pose_for(present, present_grip):
    """The posture that lines your body up with the arm's current pose.

    Derived from where the arm *is*, every tick, rather than from a stored
    park pose. That is what lets the arm stay put until told otherwise: there
    is no target to drive to, so the handshake's job is only to tell you
    which posture matches the pose it is already holding.

    Exactly invertible against map_metric_to_joint, so matching this posture
    and engaging commands the position the arm is already at -- no step.
    """
    angles = {metric: invert_metric(metric, present[spec["joint"]])
              for metric, spec in JOINT_MAP.items()}
    angles["grip"] = invert_grip(present_grip)
    return angles


def unmatchable_metrics(target_human):
    """Metrics whose target posture falls outside what an arm can hold.

    Only possible when a joint sits outside its mapped span -- past the URDF
    limit the calibration allows, or with TARGET_FRACTION below 1.0. The
    inverse mapping then asks for an angle no human reaches, so alignment
    could never complete; naming the joint beats silently never engaging.
    """
    out = []
    for metric, spec in JOINT_MAP.items():
        lo, hi = spec["human"]
        value = target_human.get(metric)
        if value is None or not min(lo, hi) <= value <= max(lo, hi):
            out.append(metric)
    return out


# How far the commanded position may get ahead of the measured one.
#
# A gravity-loaded joint under proportional control always settles short of
# what it was told. The servo's P term produces effort proportional to the
# error, so with weight to hold the error cannot reach zero -- the joint sits
# a few degrees below its goal and stays there. shoulder_lift and elbow_flex
# carry the most load on this arm, and they were the two that reported "not
# moving" while several degrees off a target well inside their range. They
# were not stuck. They were holding as close as a P-only loop can.
#
# 10 rather than something tighter because the reported gaps were 4.0 and
# 6.7 degrees and those are lower bounds -- the arm was sagging and the
# stall flag froze it, so the true figure is whatever it would have reached.
# A lead this size only ever appears on a joint that is refusing to follow,
# and STALL_HOLD_LEAD_DEG takes it away again two seconds later.
#
# So the command has to be free to lead the measurement by more than one
# tick's slew. Anchoring every goal to `present + one step` capped that lead
# at JOINT_STEP_DEG, and any joint needing more than that to hold station
# could never hold it: the loop pushed, the joint sagged, the loop pushed
# again. Doubling as anti-windup, this is also what stops the command
# running away from an arm that is genuinely stuck.
MAX_LEAD_DEG = 10.0

# --- holding against gravity: the tuning knobs -------------------------
#
# Symptom -> knob:
#   joint still reports "not moving" a few degrees off target  -> MAX_LEAD_DEG up
#   joint overshoots and comes back                            -> HOLD_GAIN down
#   joint creeps to the target slowly after arriving           -> HOLD_GAIN up
#   joint buzzes or hunts while holding still                  -> HOLD_DEADBAND_DEG up
#   correction never engages on a joint that drifts            -> HOLD_STILL_DEG up
#   a servo runs hot                                           -> MAX_LEAD_DEG down

# Errors smaller than this are left alone. Without a deadband the loop chases
# the last fraction of a degree forever, and since it is closing on a
# measurement it would be turning encoder noise into command noise -- a joint
# that is supposed to be holding still would buzz.
HOLD_DEADBAND_DEG = 0.3

# The hold offset only grows while the joint is stationary -- moving slower
# than this many degrees per tick.
#
# This is what separates "holding short of the target" from "still on its way
# there", and getting it wrong in the lenient direction is what overshoot is.
# An offset that accumulates during travel is integral windup: the joint is
# behind simply because it has not arrived yet, and every degree banked while
# it was moving is a degree it sails past when it does arrive. Steady-state
# error is only steady-state once the motion has stopped.
HOLD_STILL_DEG = 0.15

# How much of the remaining error is taken into the offset each tick, and the
# most it may move in one. Deliberately slow: this correction competes with
# nothing on the way to the target, so it can afford to take a second, and a
# gentle integral is a stable one.
HOLD_GAIN = 0.25
HOLD_STEP_DEG = 0.25

# The lead kept on a joint that has been declared stalled. Enough to hold it
# where it is, not enough to fight whatever is in the way: at full lead a
# jammed servo sits at its current limit and cooks, and at zero lead it drops
# the weight it was holding and sags -- then the periodic retry lifts it
# again, which is the slow bounce that reads as jitter.
STALL_HOLD_LEAD_DEG = 1.0

# The lead allowed while re-testing a joint that has stalled. The retry only
# has to find out whether whatever was in the way has gone, and a nudge
# answers that as well as a shove does -- so a joint that is still jammed is
# not pushed back up to MAX_LEAD_DEG every retry cycle. Restored to the full
# lead the moment the joint actually moves.
STALL_RETRY_LEAD_DEG = 3.0

# A joint that is being asked to move but has not measurably moved for this
# many consecutive ticks is stalled -- obstructed, at a calibration end stop,
# or fighting more load than it can hold. Commanding it harder achieves
# nothing and draws stall current, so the loop stops pushing and says so.
STALL_TICKS = 40
STALL_EPS_DEG = 0.5

# Once flagged, a joint is re-tried for one fresh STALL_TICKS window every
# this many ticks, rather than held forever. Holding a stalled joint at its
# measured position is itself indistinguishable from "not moving" -- with no
# retry, that one flag would latch for the rest of the run even if the cause
# was transient (servo torque ramp-up, momentary stiction at rest) rather
# than a real obstruction. Bounds the duty cycle to STALL_TICKS-on out of
# STALL_RETRY_TICKS, so a genuinely stuck joint still spends most of its time
# holding rather than leaning on whatever is blocking it.
STALL_RETRY_TICKS = 100


# How close every joint must be before engaging is offered.
ALIGN_TOLERANCE_DEG = 15.0
ALIGN_GRIP_TOLERANCE = 0.25

# How long alignment has to hold continuously before it engages on its own.
# E still engages immediately regardless of this -- the dwell only guards the
# automatic path, so a pose you are merely passing through on your way
# somewhere else doesn't trigger control by accident.
AUTO_ENGAGE_DWELL_S = 1.5
AUTO_ENGAGE_TICKS = max(1, round(AUTO_ENGAGE_DWELL_S * CONTROL_HZ))

# Per-press step for the live jog keys in run_webcam, same scheme as run_jog
# (JOG_KEYS_UP/DOWN over ARM_JOINTS + gripper). Jogging moves the pose the
# arm is holding, so it reuses the same rate-limited, stall-aware motion the
# engaged path uses rather than being a second, separate motion path.
JOG_STEP_WEBCAM = JOG_STEP_DEG
# What a jog request from the tracker UI is allowed to name.
JOG_TARGETS = ARM_JOINTS + ["gripper"]

# How close to the startup pose counts as arrived, and how long homing may
# take before it gives up and hands over anyway. The deadline exists because
# homing must never be able to trap the session: a joint that cannot reach
# the pose previously left the run stuck with no way forward, and nothing
# downstream actually needs homing to have finished -- the alignment target
# is read from wherever the arm ended up.
HOME_TOLERANCE_DEG = 2.0
HOME_TIMEOUT_S = 25.0


class KeyReader:
    """Non-blocking single-keypress reader, as a context manager.

    POSIX has to put the terminal into cbreak mode so a key arrives without
    waiting for Enter, and must restore the old settings afterwards or it
    leaves the shell in a strange state. Windows needs neither: msvcrt reads
    the console directly. Hence a context manager -- the two platforms differ
    in setup and teardown, not in the read itself.
    """

    def __enter__(self):
        self._fd = None
        if not IS_WINDOWS:
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self, timeout=0.02):
        """One keypress lowercased, or None if nothing arrived in time.

        Lowercased so Caps Lock does not silently disable every jog key.
        """
        key = self._read_raw(timeout)
        return key.lower() if key else None

    def _read_raw(self, timeout):
        if IS_WINDOWS:
            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    # Arrows and function keys arrive as two reads led by a
                    # null or 0xE0. Swallow the pair rather than letting the
                    # second half through as if it were a plain letter.
                    if ch in ("\x00", "\xe0"):
                        msvcrt.getwch()
                        return None
                    return ch
                time.sleep(0.001)
            return None
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None


def describe_jog_keys(step_deg):
    lines = ["", f"Joint jog -- {step_deg:.2f} deg per press", ""]
    for up, down, name in zip(JOG_KEYS_UP, JOG_KEYS_DOWN, ARM_JOINTS + ["gripper"]):
        unit = "%" if name == "gripper" else "deg"
        lines.append(f"    {up} / {down}    {name:<15} +/- ({unit})")
    lines += ["", "    [ / ]    smaller / larger step",
              "    p        save this pose as the startup pose",
              "    Esc or Ctrl+C to quit", ""]
    return "\n".join(lines)


def run_jog(follower, step_deg=JOG_STEP_DEG):
    """Joint-space jogging: each key nudges one joint by a fixed step.

    No IK and no URDF -- the keys address the motors directly. That makes
    this usable as a plain "does the arm move" check, and it drops the placo
    dependency the Cartesian version carried, which has no wheels on current
    Python.

    An action is sent every iteration rather than only on a keypress, so the
    arm actively holds its commanded pose instead of sagging between presses.
    """
    obs = follower.get_observation()
    joint_deg = {name: obs[f"{name}.pos"] for name in ARM_JOINTS}
    gripper_pos = obs["gripper.pos"]
    jog = list(zip(JOG_KEYS_UP, JOG_KEYS_DOWN, ARM_JOINTS + ["gripper"]))

    print(describe_jog_keys(step_deg))
    ticks = 0
    try:
        with KeyReader() as keys:
            while True:
                key = keys.read()
                # Ctrl+C reaches msvcrt as a plain \x03 rather than raising,
                # so it has to be handled as a key as well as an exception.
                if key in ("\x1b", "\x03"):
                    break
                if key in ("[", "]"):
                    lo, hi = JOG_STEP_RANGE
                    step_deg = min(hi, max(lo, step_deg * (0.5 if key == "[" else 2.0)))
                elif key == "p":
                    save_neutral(joint_deg, gripper_pos)
                    print(f"\n  Saved to {NEUTRAL_FILE}; the arm will move here "
                          f"on startup.")
                else:
                    for up, down, name in jog:
                        if key not in (up, down):
                            continue
                        direction = 1 if key == up else -1
                        if name == "gripper":
                            lo, hi = GRIPPER_LIMITS
                            gripper_pos = min(hi, max(lo, gripper_pos + direction * GRIPPER_STEP))
                        else:
                            lo, hi = JOINT_LIMITS_DEG[name]
                            joint_deg[name] = min(hi, max(lo, joint_deg[name] + direction * step_deg))
                        break

                action = {f"{name}.pos": joint_deg[name] for name in ARM_JOINTS}
                action["gripper.pos"] = gripper_pos
                follower.send_action(action)

                # Live readout, throttled to ~10 Hz: at the loop's own rate
                # the line flickers and the writes cost more than the jog.
                ticks += 1
                if ticks % 5 == 0:
                    # Each value is tagged with the key that raises it, rather
                    # than an abbreviated joint name -- shoulder_pan and
                    # shoulder_lift truncate to the same thing, and the key is
                    # the more useful label while your hands are on it anyway.
                    state = "  ".join(
                        f"{up}{(gripper_pos if name == 'gripper' else joint_deg[name]):+7.1f}"
                        for up, _, name in jog)
                    print(f"\r  {state}   step {step_deg:5.2f}  ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()


def alignment_error(angles, neutral):
    """Per-metric distance from the neutral pose, or None where unmeasured.

    Returned per metric rather than as one scalar because the point of it is
    to tell you *which* joint is holding up the engage, and in which
    direction -- a single number would only say "not yet".
    """
    out = {}
    for key, target in (neutral or {}).items():
        value = angles.get(key)
        if value is None or target is None:
            out[key] = None
            continue
        delta = value - target
        if key in JOINT_MAP:
            # Rotations wrap, so a reading of +179 against a target of -179 is
            # 2 degrees out, not 358.
            delta = (delta + 180.0) % 360.0 - 180.0
        out[key] = delta
    return out


def is_aligned(errors):
    """True when every metric is present and inside its tolerance."""
    if not errors:
        return False
    for key, delta in errors.items():
        if delta is None:
            return False
        tol = ALIGN_GRIP_TOLERANCE if key == "grip" else ALIGN_TOLERANCE_DEG
        if abs(delta) > tol:
            return False
    return True


async def run_webcam(follower):
    """Camera-driven control, gated behind an alignment handshake.

    Three states. HOMING drives the arm to the pose in config/neutral_pose.json
    and only runs at startup; READY holds position, reads the tracker but
    commands nothing, and waits for your pose to match whatever the arm is
    currently holding; ENGAGED maps your angles onto the joints continuously.

    HOMING cannot trap the session. It hands over to READY on arrival, on a
    stall, on HOME_TIMEOUT_S, or the moment you touch a jog key -- and it is
    skipped entirely when no neutral pose is saved. Nothing downstream needs
    it to have finished, because the alignment target is read from wherever
    the arm actually ended up. An earlier version blocked on reaching the
    pose exactly, which left a run stuck with no way forward whenever a joint
    would not get there.

    The gate exists because the arm has no idea where you are until you step
    in front of the cameras. Without it, the first frame of tracking commands
    whatever pose you happen to be standing in, and the arm sweeps to meet it
    -- rate-limited, so not violent, but abrupt and unasked for.

    Because the alignment target is derived from where the arm *is* (see
    human_pose_for), repositioning it is just jogging it: the jog keys work
    in READY, on this console, same layout as run_jog, and the posture you
    need to adopt follows the arm as you move it. 'p' saves the current pose
    as the startup pose. Requires this console window to have keyboard focus;
    the tracker window is a separate one.

    READY engages on its own once the pose has matched continuously for
    AUTO_ENGAGE_DWELL_S -- E in the tracker also engages immediately,
    without waiting out the dwell.

    Disengaging, or losing tracking, drops back to READY holding wherever the
    arm ended up, rather than homing again: past startup, it only does what
    you ask. It cannot resume on its own either -- re-engaging means matching
    the new pose and waiting out the dwell again, so a recovered tracker
    never picks the arm back up unannounced.
    """
    from aiohttp import web

    def quiet_connection_reset(loop, context):
        """Drop the traceback a vanished client leaves in the console.

        On Windows the proactor event loop tries to shut a socket down after
        the peer has already gone, which raises ConnectionResetError inside
        an asyncio callback -- nothing awaits it, so the default handler
        prints a full traceback. It is noise: every client here is the
        tracker, losing one is an ordinary event, and the code that cares
        already discards it from `clients` and holds the arm. Printing a
        traceback for it in the middle of a teleop session buries the
        messages that do matter.
        """
        if isinstance(context.get("exception"), ConnectionResetError):
            return
        loop.default_exception_handler(context)

    asyncio.get_running_loop().set_exception_handler(quiet_connection_reset)

    obs = follower.get_observation()
    gripper_pos = obs["gripper.pos"]
    stall_ref = {name: obs[f"{name}.pos"] for name in ARM_JOINTS}
    # What was last sent to each joint, kept as state rather than recomputed
    # from the measurement each tick. This is the loop's integral term: it is
    # what lets the command settle a few degrees past a target and hold it
    # there, and it is why measurement noise no longer feeds straight back
    # out as command noise.
    commanded = {name: obs[f"{name}.pos"] for name in ARM_JOINTS}
    prev_present = {name: obs[f"{name}.pos"] for name in ARM_JOINTS}
    # The standing offset each joint needs to hold its own weight, learned
    # while it is stationary. Kept apart from the command on purpose: the
    # command has a target to travel to, and this has only a load to oppose.
    hold_offset = {name: 0.0 for name in ARM_JOINTS}
    stalled_ticks = {name: 0 for name in ARM_JOINTS}
    stalled = set()
    # Stalled at least once and being re-tested. Separate from `stalled` so
    # the retry can be gentler than a first attempt without being as timid
    # as simply holding station.
    retrying = set()
    aligned_ticks = 0
    driven = set()
    jog_pairs = list(zip(JOG_KEYS_UP, JOG_KEYS_DOWN, ARM_JOINTS + ["gripper"]))

    # What the arm holds when not homing or engaged. Seeded from where it
    # already is, so if there is no startup pose to home to, connecting
    # commands no motion at all.
    joint_target = {name: obs[f"{name}.pos"] for name in ARM_JOINTS}
    gripper_target = obs["gripper.pos"]
    target_human = human_pose_for(joint_target, gripper_target)

    home_joints, home_grip = load_neutral()
    state = "HOMING" if home_joints else "READY"
    home_deadline = time.monotonic() + HOME_TIMEOUT_S

    # Anything worth telling the operator goes through here, so it reaches the
    # tracker window as well as this terminal. Printing only to stdout put
    # every refusal in a console the operator is not looking at -- they are
    # standing in front of the cameras, holding a pose.
    message = {"text": "", "until": 0.0}

    def say(text, hold=4.0):
        message["text"] = text
        message["until"] = time.monotonic() + hold
        print(text)
    incoming = {"angles": {}, "tracking": False, "request": None, "jog": [],
                "ranges": None}
    lock = asyncio.Lock()
    clients = set()
    lost_frames = 0

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
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
                    incoming["angles"] = data.get("angles") or {}
                    incoming["tracking"] = bool(data.get("tracking"))
                    # Requests are one-shot, so a new one never silently
                    # overwrites an unhandled one.
                    if data.get("request"):
                        incoming["request"] = data["request"]
                    # Jog nudges from the tracker UI. Queued rather than
                    # latched: every press has to land, and two presses are
                    # not the same as one. Bounded because a client that
                    # spams them must not be able to grow this without limit
                    # -- 32 is well past what a held key produces between
                    # ticks, and dropping the excess is better than a queue
                    # the arm is still working through a minute later.
                    for item in (data.get("jog") or [])[:8]:
                        if len(incoming["jog"]) < 32:
                            incoming["jog"].append(item)
                    if data.get("ranges"):
                        incoming["ranges"] = data["ranges"]
        finally:
            clients.discard(ws)
            async with lock:
                incoming["angles"] = {}
                incoming["tracking"] = False
            print("Webcam tracker disconnected")
        return ws

    stopping = asyncio.Event()

    async def shutdown_handler(request):
        """Stop cleanly on request, so the arm is released rather than killed.

        run_teleop.py needs a way to end this process that is not a signal.
        On Windows there is no SIGTERM: Popen.terminate() is TerminateProcess,
        which skips every finally block -- including follower.disconnect(),
        which is what drops torque. An arm left holding position with the
        serial port never closed is the one outcome worth building a whole
        endpoint to avoid, and the server is already here.
        """
        print("\nShutdown requested; releasing the arm.")
        stopping.set()
        return web.Response(text="stopping")

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/shutdown", shutdown_handler)
    # Not the 60 s default: cleanup waits on in-flight handlers, and a
    # connected tracker always has one parked in `async for msg in ws`. The
    # wait would outlast run_teleop.py's patience and get this process killed
    # at exactly the moment it was trying to shut down tidily.
    runner = web.AppRunner(app, shutdown_timeout=1.0)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WS_PORT)
    await site.start()
    def print_posture(nh):
        print("Posture to adopt:")
        for metric in list(JOINT_MAP) + ["grip"]:
            unit = "" if metric == "grip" else " deg"
            print(f"    {metric:<19} {nh[metric]:+8.1f}{unit}")

    print()
    load_arm_ranges()
    print(f"\nWebcam control bridge listening on ws://0.0.0.0:{WS_PORT}/ws")
    print("Run arm_motion_tracker/tracker.py --robot pointed at this host.")
    if home_joints:
        print(f"\nMoving to the startup pose from {NEUTRAL_FILE}:")
        print("    " + "  ".join(f"{n}={home_joints[n]:+.1f}" for n in ARM_JOINTS)
              + f"  gripper={home_grip:.0f}%")
        print("Any jog key takes over and stops it. After that it holds position")
        print("and only does what you ask.")
        print_posture(human_pose_for(home_joints, home_grip))
    else:
        print("\nNo startup pose saved, so the arm holds where it is. Jog it where")
        print("you want it and press p to save that as the startup pose.")
        print_posture(target_human)
    print("\nMatch the posture above and it engages on its own after holding")
    print("still; E in the tracker skips the wait.")
    print("\nJog keys (this console needs keyboard focus; under run_teleop.py it\n"
          "usually does not have it -- use the jog buttons in the tracker page):")
    print(describe_jog_keys(JOG_STEP_WEBCAM))
    print("Tracker keys: E engage, D disengage. Ctrl+C to quit.\n")

    def apply_jog(name, direction):
        """One jog nudge, from the console keys or from the tracker UI.

        Both go through here because they mean the same thing, and because
        the console is the wrong place to ask from: under run_teleop.py you
        are looking at the tracker page in a browser, and these keys only
        reach a console window that is behind it and does not have focus.
        That is not a broken keyboard, it is keystrokes going somewhere else.
        """
        nonlocal state, joint_target, gripper_target
        if state == "ENGAGED":
            return
        if state == "HOMING":
            # Touching a jog control is taking over. Homing would otherwise
            # overwrite the nudge on the very next tick.
            state = "READY"
            joint_target, gripper_target = dict(present), present_grip
            say("Homing cancelled; holding position.")
        if name == "gripper":
            lo, hi = GRIPPER_LIMITS
            gripper_target = min(hi, max(lo, gripper_target + direction * GRIPPER_STEP))
            print(f"\r  gripper {gripper_target:+7.1f}%  ", end="", flush=True)
            return
        lo, hi = JOINT_LIMITS_DEG[name]
        joint_target[name] = min(hi, max(lo, joint_target[name] + direction * JOG_STEP_WEBCAM))
        # A joint the operator is actively nudging deserves a fresh chance
        # rather than waiting out the rest of its retry cycle.
        stalled.discard(name)
        retrying.discard(name)
        stalled_ticks[name] = 0
        # One line, rewritten in place: a held key repeats at the OS rate,
        # and the full posture block would scroll the console away faster
        # than it could be read.
        metric = next((m for m, sp in JOINT_MAP.items() if sp["joint"] == name), None)
        matching = (f"  ->  match {metric} {invert_metric(metric, joint_target[name]):+6.1f}deg"
                    if metric else "")
        print(f"\r  {name} {joint_target[name]:+7.1f}deg{matching}  ", end="", flush=True)

    def save_here():
        save_neutral(joint_target, gripper_target)
        say(f"Saved this pose to {NEUTRAL_FILE}; it is where the arm "
            f"will move to on startup.")

    keys = KeyReader()
    keys.__enter__()
    try:
        while not stopping.is_set():
            async with lock:
                angles = dict(incoming["angles"])
                tracking = incoming["tracking"]
                request = incoming["request"]
                incoming["request"] = None
                # Applied below, once this tick's measured position is in
                # hand: taking over from homing needs somewhere to hold.
                jog_requests, incoming["jog"] = incoming["jog"], []
                new_ranges, incoming["ranges"] = incoming["ranges"], None

            # Closed loop: what the arm actually reports, not what this loop
            # believes it commanded. Integrating an internal model open-loop
            # meant that once the arm fell behind -- stalled, obstructed, or
            # asked for somewhere it cannot reach -- nothing could ever notice,
            # and the loop re-sent the same unreachable target forever. Read
            # before handling input, so a key that takes over from homing has
            # this tick's position to hold, not the previous tick's.
            obs = follower.get_observation()
            present = {n: obs[f"{n}.pos"] for n in ARM_JOINTS}
            present_grip = obs.get("gripper.pos", gripper_pos)

            local_key = keys.read(timeout=0)
            if local_key == "p" and state != "ENGAGED":
                save_here()
            elif local_key:
                for up, down, name in jog_pairs:
                    if local_key in (up, down):
                        apply_jog(name, 1 if local_key == up else -1)
                        break

            if new_ranges:
                # Not while engaged: the map is what turns your arm into joint
                # targets, so changing it mid-flight would step the arm the
                # moment the new gain took effect.
                if state == "ENGAGED":
                    say("Cannot change arm ranges while engaged. Release first.")
                else:
                    apply_arm_ranges(new_ranges, source="the tracker")
                    target_human = human_pose_for(present, present_grip)
                    say("Arm ranges updated; the posture to match has moved with them.")

            for item in jog_requests:
                if not isinstance(item, dict):
                    continue
                if item.get("save"):
                    if state != "ENGAGED":
                        save_here()
                elif item.get("joint") in JOG_TARGETS:
                    apply_jog(item["joint"], -1 if item.get("direction", 1) < 0 else 1)

            for name in ARM_JOINTS:
                wants_to_move = abs(joint_target[name] - present[name]) > STALL_EPS_DEG
                # Progress is measured against a reference position that is
                # re-anchored whenever the joint gains ground -- NOT against
                # the previous tick.
                #
                # The per-tick test this replaces was wrong in a way that
                # only showed up under load. Commands are rate limited to
                # JOINT_STEP_DEG ahead of the measured position, so a loaded
                # joint travels a fraction of a degree per 50 ms tick even
                # when it is moving perfectly well; comparing that against a
                # 0.5 deg threshold made every slow joint look stalled from
                # the first tick. elbow_flex carries the most gravity load of
                # any joint here, so it was the one that got abandoned --
                # reported "not moving" while visibly travelling from +97 to
                # +60, and dropped from the homing set 60 deg short.
                #
                # Cumulatively, the same threshold means what it says: a
                # joint that has not gained 0.5 deg in STALL_TICKS is stuck.
                if not wants_to_move or abs(present[name] - stall_ref[name]) > STALL_EPS_DEG:
                    stall_ref[name] = present[name]
                    stalled_ticks[name] = 0
                    stalled.discard(name)
                    # It moved, so whatever was wrong is over: full authority
                    # again, not the reduced retry lead.
                    retrying.discard(name)
                    continue

                stalled_ticks[name] += 1
                if name in stalled:
                    # Periodic retry, not a one-way latch: a stalled joint is
                    # commanded to hold its measured position (below), which
                    # is indistinguishable from being stuck, so without this
                    # the flag could never clear even once the cause passed.
                    if stalled_ticks[name] >= STALL_RETRY_TICKS:
                        stalled.discard(name)
                        retrying.add(name)
                        stalled_ticks[name] = 0
                        stall_ref[name] = present[name]
                elif stalled_ticks[name] >= STALL_TICKS:
                    stalled.add(name)
                    say(f"{name} is not moving: commanded {joint_target[name]:+.1f}, "
                        f"holding at {present[name]:+.1f}. Retrying periodically.",
                        hold=15.0)

            # The posture to match tracks the arm, not a stored pose, so
            # jogging the arm moves the target your body is measured against.
            target_human = human_pose_for(present, present_grip)
            errors = alignment_error(angles, target_human)
            aligned = tracking and is_aligned(errors)
            unmatchable = unmatchable_metrics(target_human)

            # Every engage request gets an answer, in every state. Silently
            # dropping the ones that arrive outside READY is indistinguishable
            # from the key not working at all.
            if request == "engage" and state == "ENGAGED":
                say("Already engaged.")
                request = None

            if state == "HOMING":
                joint_target, gripper_target = dict(home_joints), home_grip
                # A stalled joint is already being held safely and is not
                # going to arrive, so it is excluded rather than allowed to
                # cut the move short for the joints that still can finish.
                pending = [n for n in ARM_JOINTS if n not in stalled
                           and abs(present[n] - home_joints[n]) > HOME_TOLERANCE_DEG]
                timed_out = time.monotonic() > home_deadline
                if not pending or timed_out:
                    # Hands over holding the pose it actually reached, so a
                    # joint that would not get there costs the rest of its own
                    # homing move and nothing else.
                    state = "READY"
                    aligned_ticks = 0
                    joint_target, gripper_target = dict(present), present_grip
                    short = sorted(stalled) + (pending if timed_out else [])
                    say(f"At the startup pose, except {', '.join(short)}. "
                        f"Holding position." if short else "At the startup pose.")

            elif state == "READY":
                aligned_ticks = aligned_ticks + 1 if aligned else 0
                if request == "engage" or aligned_ticks >= AUTO_ENGAGE_TICKS:
                    if aligned:
                        state = "ENGAGED"
                        lost_frames = 0
                        aligned_ticks = 0
                        say("Engaged." if request == "engage" else "Engaged (auto).")
                    elif unmatchable:
                        # Not your posture's fault -- the arm is somewhere the
                        # mapping cannot express, so no posture would match.
                        say(f"Cannot engage: {', '.join(unmatchable)} needs a pose "
                            f"outside human range. Jog that joint back.")
                    elif not tracking:
                        say("Cannot engage: no triangulated pose.")
                    else:
                        # Name the joint that blocked it, rather than just refusing.
                        worst = max((k for k, v in errors.items() if v is not None),
                                    key=lambda k: abs(errors[k]), default=None)
                        say(f"Cannot engage: {worst} is {errors[worst]:+.0f} out."
                            if worst else "Cannot engage: no readings.")
            elif state == "ENGAGED":
                lost_frames = 0 if tracking else lost_frames + 1
                if request == "disengage" or lost_frames > CONTROL_HZ:
                    state = "READY"
                    aligned_ticks = 0
                    # Hold wherever it ended up rather than returning anywhere.
                    joint_target, gripper_target = dict(present), present_grip
                    say("Disengaged; holding position." if request == "disengage"
                        else "Lost tracking; holding position.")
                else:
                    # A metric that came back None is not zero, it is absent:
                    # that joint keeps its last target. Worth reporting, since
                    # a joint frozen for want of a reading looks exactly like
                    # a dead motor from the outside -- which is how a wrist
                    # the side camera cannot see reads as "the elbow does not
                    # move when engaged".
                    driven.clear()
                    for metric, spec in JOINT_MAP.items():
                        value = angles.get(metric)
                        if value is not None:
                            joint_target[spec["joint"]] = map_metric_to_joint(metric, value)
                            driven.add(spec["joint"])
                    if angles.get("grip") is not None:
                        gripper_target = map_grip(angles["grip"])

            # Two separate jobs, deliberately not mixed.
            #
            # Travel: the command slews toward the target at JOINT_STEP_DEG a
            # tick and is never asked to go past it. Nothing accumulates here,
            # so nothing is banked up to be paid out as overshoot.
            #
            # Hold: a gravity-loaded joint under proportional control settles
            # short of whatever it was told, so once it has stopped moving and
            # is still off target, the shortfall is taken into a standing
            # offset that the command is then aimed at instead. That offset IS
            # the holding effort -- rebuilding the goal from the measurement
            # every tick, as this used to, capped it at one slew step and left
            # the loaded joints sagging away from every target they were given.
            #
            # Bounded throughout: JOINT_STEP_DEG is the slew limit, MAX_LEAD_DEG
            # bounds the offset, and the command may never sit further than
            # `lead` from where the arm actually is, whatever the arithmetic
            # above produced.
            action = {}
            for name in ARM_JOINTS:
                low, high = JOINT_LIMITS_DEG[name]
                if name in stalled:
                    lead = STALL_HOLD_LEAD_DEG
                elif name in retrying:
                    lead = STALL_RETRY_LEAD_DEG
                else:
                    lead = MAX_LEAD_DEG

                error = joint_target[name] - present[name]
                if (name not in stalled
                        and abs(present[name] - prev_present[name]) < HOLD_STILL_DEG
                        and abs(error) > HOLD_DEADBAND_DEG):
                    hold_offset[name] += max(-HOLD_STEP_DEG,
                                             min(HOLD_STEP_DEG, HOLD_GAIN * error))
                    hold_offset[name] = max(-MAX_LEAD_DEG,
                                            min(MAX_LEAD_DEG, hold_offset[name]))

                if name not in stalled:
                    desired = joint_target[name] + hold_offset[name]
                    commanded[name] += max(-JOINT_STEP_DEG,
                                           min(JOINT_STEP_DEG, desired - commanded[name]))
                commanded[name] = max(present[name] - lead,
                                      min(present[name] + lead, commanded[name]))
                commanded[name] = max(low, min(high, commanded[name]))
                action[f"{name}.pos"] = commanded[name]
            prev_present = present

            low, high = GRIPPER_LIMITS
            step = max(-GRIPPER_TICK_STEP, min(GRIPPER_TICK_STEP, gripper_target - present_grip))
            gripper_pos = max(low, min(high, present_grip + step))
            action["gripper.pos"] = gripper_pos
            follower.send_action(action)

            # Tell the tracker what to draw. It renders the alignment view, so
            # it needs the state and the per-joint errors; computing the errors
            # here keeps the mapping in one place.
            if clients:
                payload = json.dumps({
                    "state": state,
                    "aligned": aligned,
                    "errors": errors,
                    "stalled": sorted(stalled),
                    "driving": sorted(driven) if state == "ENGAGED" else [],
                    "joints": {n: round(present[n], 1) for n in ARM_JOINTS},
                    "gripper": round(present_grip, 1),
                    "step": JOG_STEP_WEBCAM,
                    "message": message["text"] if time.monotonic() < message["until"] else "",
                    "tolerance": {"angle": ALIGN_TOLERANCE_DEG, "grip": ALIGN_GRIP_TOLERANCE},
                })
                for ws in list(clients):
                    try:
                        await ws.send_str(payload)
                    except Exception:
                        clients.discard(ws)

            await asyncio.sleep(1 / CONTROL_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        keys.__exit__(None, None, None)
        await runner.cleanup()


def find_arm_port():
    """The arm's serial port, or an error naming what was actually found.

    "Wrong port" and "not plugged in" need opposite fixes and produce the
    same opaque timeout from the motor bus, so it is worth deciding which
    applies before connecting rather than after.
    """
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit(
            "No serial ports found at all -- the arm is not plugged in, is not\n"
            "powered, or its USB-serial driver is not installed."
        )
    wch = [p for p in ports if p.vid == WCH_VENDOR_ID]
    if len(wch) == 1:
        return wch[0].device
    if len(ports) == 1:
        return ports[0].device

    listing = "\n".join(f"    {p.device:<8} {p.description}" for p in ports)
    which = "Several WCH bridges" if len(wch) > 1 else "No WCH bridge among the"
    raise SystemExit(f"{which} ports present; pass --port explicitly. Found:\n{listing}")


def run_check(follower, port):
    """Report the arm's state and exit, without commanding any motion.

    The first thing to establish on a new setup is that the port is right and
    that every motor answers -- and that question is worth being able to ask
    without the arm moving.
    """
    calibrated = follower.is_calibrated
    print(f"\nConnected on {port}.")

    if calibrated:
        print("Calibration: loaded.\n")
        obs = follower.get_observation()
        readings = {n: obs.get(f"{n}.pos") for n in ARM_JOINTS + ["gripper"]}
        unit = "deg"
    else:
        # Normalised reads need calibration to map encoder counts onto degrees
        # -- lerobot raises rather than guessing. Raw ticks still answer the
        # question a connection test is asking, which is whether every motor
        # is on the bus and talking.
        print("Calibration: NONE (raw encoder counts shown, not degrees).\n")
        readings = follower.bus.sync_read("Present_Position", normalize=False)
        unit = "ticks"

    missing = []
    for name in ARM_JOINTS + ["gripper"]:
        value = readings.get(name)
        if value is None:
            missing.append(name)
            print(f"    {name:<15}   no reading")
            continue
        limits = ""
        if calibrated and name in JOINT_LIMITS_DEG:
            lo, hi = JOINT_LIMITS_DEG[name]
            flag = "" if lo <= value <= hi else "   <-- outside the URDF limit"
            limits = f"    (limit {lo:+7.1f} to {hi:+7.1f}){flag}"
        print(f"    {name:<15} {value:+9.2f} {unit}{limits}")

    if missing:
        print(f"\nNo reading from: {', '.join(missing)}. Check power and the daisy chain.")
    else:
        print(f"\nAll {len(ARM_JOINTS) + 1} motors responded.")
    if not calibrated:
        print("\nCalibrate before jogging. Until then the arm has no mapping from\n"
              "encoder counts to degrees, so JOINT_LIMITS_DEG means nothing to it\n"
              "and a jog command would not land where you expect.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", choices=["keyboard", "webcam"], default="keyboard",
        help="keyboard: joint-space jog via keypresses. "
             "webcam: joints ease toward extended/neutral/flexed presets driven by "
             "arm_motion_tracker/tracker.py over a websocket.",
    )
    parser.add_argument("--port", help="serial port (default: auto-detect, e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--id", default="my_follower", help="calibration id for this arm")
    parser.add_argument("--step", type=float, default=JOG_STEP_DEG,
                        help="degrees per keypress when jogging")
    parser.add_argument("--check", action="store_true",
                        help="report the arm's state and exit, without moving it")
    parser.add_argument("--recalibrate", action="store_true",
                        help="re-run the arm's range-of-motion calibration, then exit")
    args = parser.parse_args()

    port = args.port or find_arm_port()
    follower = SO101Follower(SO101FollowerConfig(
        port=port, id=args.id,
        use_degrees=True,  # joint limits work in real joint degrees, not normalized percent
        max_relative_target=MAX_RELATIVE_TARGET,
    ))
    print(f"Connecting to {port}...")
    # --check is a read-only probe, so it must not calibrate: lerobot runs
    # calibration on connect whenever no calibration file exists, and that is
    # an interactive physical procedure (torque off, move every joint through
    # its range) -- not something a "is this plugged in properly" test should
    # start. The check reports calibration status instead.
    follower.connect(calibrate=not (args.check or args.recalibrate))
    if not (args.check or args.recalibrate):
        follower.bus.write("P_Coefficient", "elbow_flex", ELBOW_P_COEFFICIENT)

    try:
        if args.recalibrate:
            # calibrate() prompts before overwriting an existing calibration,
            # and connect() only calls it when there is none -- so with a valid
            # calibration on disk it never runs. Calling it directly is the way
            # to redo the ranges without deleting the file by hand.
            follower.calibrate()
        elif args.check:
            run_check(follower, port)
        elif args.input == "keyboard":
            run_jog(follower, args.step)
        else:
            asyncio.run(run_webcam(follower))
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
