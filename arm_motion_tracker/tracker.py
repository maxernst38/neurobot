"""Webcam pose/hand tracking -> predicted shoulder/elbow/wrist/grip motion.

OpenCV window replacement for the old browser app (index.html/main.js/motion.js/
robotLink.js). Same MediaPipe models, same joint-angle geometry, same six
tracked metrics, and the same websocket wire format to main.py's
`--input webcam` bridge, so main.py needs no changes.

Layout: a top row of one annotated feed per camera plus the metrics panel,
over a row of front/side/top orthographic projections. Each camera runs its
own independent detection pipeline; the feeds show what each camera detected
in 2D, which is how you diagnose a triangulation that is dropping joints.

The 3D views draw ONE skeleton: the triangulated one. With a stereo
calibration present (see calibrate_cameras.py) the two views are fused into
it, and every metric but grip is measured from it. That is the point of
calibrating -- a monocular estimate has to guess depth, and MediaPipe's
guess is poor enough to invert joint angles, whereas triangulated depth is
measured from known geometry. Drawing the two monocular skeletons alongside
it only made the guesses compete visually with the measurement.

Grip is the one exception, and it stays monocular by necessity: it needs the
finger PIP/DIP joints, which the pose model does not have. It is also the
metric that suffers least, being a ratio of distances along each finger.
Because it is read from one camera either way, the hand model runs only on
that camera by default -- see --hands, which is the cheapest 40% of a
second camera's per-frame cost to give back.

Controls: SPACE = toggle tracked side, T = toggle triangulation,
V = toggle the 3D views, C = toggle robot connection, Q/Esc = quit.

`--web` serves the same thing as a browser page instead of opening the
window: same controls, same keys, plus buttons and a readable alignment
readout, at http://localhost:8080/. The OpenCV window cannot be resized
without rescaling the video with it and puts the text you have to read while
standing in front of the cameras at 0.4-scale cv2.putText; a page can also be
opened on a phone propped where you can see it (see --web-host). The camera
feeds and 3D views are still drawn here and streamed as MJPEG -- only the
text panel is rebuilt as DOM. See webui.py.

`--record` captures the session to a file and `--replay FILE` plays one
back, both also drivable from the page. A capture keeps what each camera
saw, not only what was made of it, so it can be re-processed later against a
new calibration or changed geometry code -- which is what makes it usable as
research data rather than as a screen recording. See recording.py, and
--derive for the three ways a replayed frame can be interpreted.

Replay needs no cameras: with `--no-cameras` the tracker opens none and
serves the page alone, which is how a capture is reviewed on a machine
nowhere near the rig.

With the robot connected (C), the arm holds wherever it already is -- it
never moves on its own. Match the pose it is holding (the panel shows how far
each joint is out, and the 3D views draw it as a ghost arm) and it engages
after a moment; E engages immediately and D releases. Nothing is commanded
until you engage, so the arm never sweeps off to meet whatever pose you
happened to be standing in.
"""

import argparse
import json
import math
import os
import time
import urllib.request
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import recording
from mediapipe import Image, ImageFormat

from camera_io import (DEFAULT_CAMERA_INDICES, Camera, fit_scale, is_torn,
                       open_cameras, read_all)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


# --- geometry + classification (port of motion.js) --------------------------

POSE = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    # Pose carries three coarse hand points per side as well as the wrist.
    # They are nothing like the hand model's 21 joints, but they belong to the
    # SAME model as the arm and so share its frame of reference -- and they
    # are triangulated along with everything else. That is what lets wrist
    # flexion and rotation be *measured* here rather than inferred from one
    # camera's guess at depth, which is the whole point of calibrating.
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
}

# Both signed metrics are geometrically well defined, but which direction
# counts as positive depends on anatomy this code cannot check for itself.
# Flip an entry to -1 if a metric reads inverted against a real arm (the same
# convention main.py uses for JOINT_SIGN).
METRIC_SIGN = {"wrist_flexion": 1}

HAND = {
    "WRIST": 0,
    "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_MCP": 5, "INDEX_PIP": 6, "INDEX_DIP": 7, "INDEX_TIP": 8,
    "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_DIP": 11, "MIDDLE_TIP": 12,
    "RING_MCP": 13, "RING_PIP": 14, "RING_DIP": 15, "RING_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_DIP": 19, "PINKY_TIP": 20,
}

# Matched (where possible) to the SO-101 joint names used in main.py, so this
# can feed the same webcam teleop mapping.
METRICS = [
    # Heading of the upper arm about the torso, not a rotation of it: zero
    # is the arm straight out to the side, positive is swung forward. The
    # range is what a shoulder can reach rather than the +-180 an atan2 can
    # return, so the display bands mean something and the wrap sits well
    # outside it. The key keeps its name because it still drives
    # shoulder_pan; see GEOMETRY_VERSION for why the meaning changed.
    {"key": "shoulder_rotation", "label": "Shoulder Heading", "unit": "deg", "range": (-45, 135), "highLabel": "Swung Forward", "midLabel": "Half Forward", "lowLabel": "Out to the Side"},
    {"key": "shoulder_flexion", "label": "Shoulder Ext/Flex", "unit": "deg", "range": (0, 180), "highLabel": "Flexed (raised)", "midLabel": "Neutral", "lowLabel": "Extended (lowered)"},
    {"key": "elbow_flexion", "label": "Elbow Ext/Flex", "unit": "deg", "range": (0, 180), "highLabel": "Extended", "midLabel": "Neutral", "lowLabel": "Flexed"},
    {"key": "wrist_flexion", "label": "Wrist Ext/Flex", "unit": "deg", "range": (-90, 90), "highLabel": "Extended", "midLabel": "Neutral", "lowLabel": "Flexed"},
    {"key": "grip", "label": "Hand Grip", "unit": "", "range": (0, 1), "highLabel": "Gripping", "midLabel": "Half", "lowLabel": "Open"},
]

POSE_CONNECTIONS = [
    (POSE["LEFT_SHOULDER"], POSE["RIGHT_SHOULDER"]),
    (POSE["LEFT_SHOULDER"], POSE["LEFT_ELBOW"]), (POSE["LEFT_ELBOW"], POSE["LEFT_WRIST"]),
    (POSE["RIGHT_SHOULDER"], POSE["RIGHT_ELBOW"]), (POSE["RIGHT_ELBOW"], POSE["RIGHT_WRIST"]),
    (POSE["LEFT_SHOULDER"], POSE["LEFT_HIP"]), (POSE["RIGHT_SHOULDER"], POSE["RIGHT_HIP"]),
    (POSE["LEFT_HIP"], POSE["RIGHT_HIP"]),
    # The coarse hand triangle. Drawn because it is what wrist rotation is
    # measured from: if this fan looks wrong, the rotation reading is wrong.
    (POSE["LEFT_WRIST"], POSE["LEFT_INDEX"]), (POSE["LEFT_WRIST"], POSE["LEFT_PINKY"]),
    (POSE["LEFT_INDEX"], POSE["LEFT_PINKY"]),
    (POSE["RIGHT_WRIST"], POSE["RIGHT_INDEX"]), (POSE["RIGHT_WRIST"], POSE["RIGHT_PINKY"]),
    (POSE["RIGHT_INDEX"], POSE["RIGHT_PINKY"]),
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# Bumped whenever a metric's geometry changes meaning. A recorded human
# range is a measurement of a *definition*, so one taken under an older
# definition is not merely stale, it is a wrong gain -- and a wrong gain is
# an arm that moves further and faster than the person driving it expects.
# Recordings carry this; main.py refuses a range that does not match.
#   1: original. shoulder_rotation was the forearm's bearing in camera 1's
#      axes -- entangled with elbow flexion (measured: 120 deg of elbow
#      swung it 99.5 deg) and dependent on where the camera sat.
#   2: shoulder_rotation is the upper arm's heading in a torso frame built
#      from the hips and shoulders. Zero is the arm straight out to the
#      side, positive forward.
GEOMETRY_VERSION = 2

# How far the upper arm must be off the torso's own axis before its heading
# means anything: 0.20 is about 11.5 degrees from straight down. Below that
# the horizontal component is small enough that landmark noise dominates the
# direction entirely.
HEADING_MIN_HORIZONTAL = 0.20


def torso_frame(pose_world):
    """Right/up/forward unit axes of the torso, or None.

    An anatomical frame, so angles built on it mean the same thing wherever
    the cameras are put. Needs both hips and both shoulders; returns None
    rather than guessing when the body is half out of frame, because a
    frame built from three points and an assumption is worse than no
    reading at all -- it looks like a measurement.
    """
    if not pose_world:
        return None

    def at(name):
        p = pose_world[POSE[name]]
        return None if p is None else np.array([p.x, p.y, (p.z or 0.0)])

    sl, sr = at("LEFT_SHOULDER"), at("RIGHT_SHOULDER")
    hl, hr = at("LEFT_HIP"), at("RIGHT_HIP")
    if any(v is None for v in (sl, sr, hl, hr)):
        return None
    # y points DOWN in MediaPipe's world convention, so hips-minus-shoulders
    # is the direction from the chest toward the head.
    up = (hl + hr) / 2.0 - (sl + sr) / 2.0
    right = sr - sl
    if np.linalg.norm(up) < 1e-9 or np.linalg.norm(right) < 1e-9:
        return None
    up = up / np.linalg.norm(up)
    right = right - np.dot(right, up) * up      # square it against the spine
    if np.linalg.norm(right) < 1e-9:            # shoulders stacked over hips
        return None
    right = right / np.linalg.norm(right)
    forward = np.cross(right, up)
    n = np.linalg.norm(forward)
    return (right, up, forward / n) if n > 1e-9 else None


def shoulder_heading(pose_world, side):
    """Where the UPPER arm points, about the torso's vertical axis.

    This is what shoulder_pan can actually follow: swinging the whole arm
    left and right in front of the body. Zero is the arm straight out to
    that side, positive is forward across the body.

    Measured from the upper arm and not the forearm. Reading it off the
    forearm, as this did originally, meant bending the elbow swung the
    answer while the shoulder never moved -- 99.5 degrees of it over a
    normal elbow range, which is most of the metric's useful travel.

    Centred on "out to the side" so the +-180 wrap falls behind the back
    where nobody reaches. Where an angle wraps matters as much as what it
    measures: a reading that flips sign mid-range is unusable however
    correct it is.
    """
    frame = torso_frame(pose_world)
    shoulder = pose_world[POSE[f"{side}_SHOULDER"]] if pose_world else None
    elbow = pose_world[POSE[f"{side}_ELBOW"]] if pose_world else None
    if frame is None or shoulder is None or elbow is None:
        return None
    right, _up, forward = frame
    upper = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y,
                      (elbow.z or 0.0) - (shoulder.z or 0.0)])
    if np.linalg.norm(upper) < 1e-9:
        return None
    outward = right if side == "RIGHT" else -right
    along, across = float(np.dot(upper, forward)), float(np.dot(upper, outward))
    # An arm hanging straight down has no heading: it lies along the torso's
    # own axis, so its horizontal part is nearly zero and atan2 of two tiny
    # numbers is noise that swings through the whole range on landmark
    # jitter. Report nothing instead. Downstream already treats a missing
    # metric as "hold the last target", which is exactly right here -- the
    # alternative is a shoulder_pan command thrashing while the person is
    # simply standing at rest.
    if math.hypot(along, across) < HEADING_MIN_HORIZONTAL * np.linalg.norm(upper):
        return None
    return math.degrees(math.atan2(along, across))


def _vec(a, b):
    return np.array([b.x - a.x, b.y - a.y, (b.z or 0) - (a.z or 0)])


def _mag(v):
    return float(np.linalg.norm(v)) or 1e-6


def _angle_at(a, b, c):
    """Angle at vertex b, formed by rays b->a and b->c, in degrees.

    Expects *world* landmarks (metres, all three axes on one isotropic
    scale). Do not pass normalized image landmarks: their x/y are fractions
    of width/height and their z is a loosely-scaled relative depth estimate,
    so a dot product over those mixes incompatible units and yields angles
    that are compressed and can even invert (an extended arm reading as more
    bent than a flexed one).
    """
    v1, v2 = _vec(b, a), _vec(b, c)
    cos = min(1.0, max(-1.0, float(np.dot(v1, v2)) / (_mag(v1) * _mag(v2))))
    return math.degrees(math.acos(cos))


def _dist(a, b):
    return _mag(_vec(a, b))


def _signed_angle(a, b, axis):
    """Angle from a to b measured about axis, in degrees.

    The unsigned _angle_at cannot distinguish a joint bent one way from the
    same joint bent the other -- both shrink the angle between the two
    segments identically. Projecting the turn onto an explicit axis recovers
    the direction, positive by the right-hand rule about that axis.
    """
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return None
    return math.degrees(math.atan2(float(np.dot(np.cross(a, b), axis / n)),
                                   float(np.dot(a, b))))


def compute_joint_metrics(pose_world, hand_world, side):
    """Six joint metrics for one side of the body.

    Every metric except grip is measured from `pose_world` alone -- one
    model, one frame of reference, no cross-model bridging. When those points
    come from triangulate_pose the whole set is measured geometry rather than
    a monocular depth guess, which is the entire reason for calibrating.

    Grip is the exception and stays on the hand model, because it needs the
    finger PIP/DIP joints and Pose has no fingers at all. It is also the
    metric that needs triangulation least: it is a ratio of distances along
    each finger, so it is invariant to scale and largely immune to the depth
    error that makes the arm angles unreliable from a single view.

    Entries of pose_world may individually be None -- triangulate_pose
    returns None for any joint it could not resolve -- so each metric is
    gated on exactly the points it needs rather than on the set as a whole.
    A hip out of frame must not cost you an elbow angle.
    """
    def pt(name):
        return pose_world[POSE[f"{side}_{name}"]] if pose_world else None

    shoulder, elbow, wrist = pt("SHOULDER"), pt("ELBOW"), pt("WRIST")
    hip, index, pinky = pt("HIP"), pt("INDEX"), pt("PINKY")

    out = {
        "shoulder_rotation": None, "shoulder_flexion": None, "elbow_flexion": None,
        "wrist_flexion": None, "grip": None,
    }

    if shoulder and elbow and wrist:
        # Rotation-invariant by construction: the angle between two vectors
        # at the elbow does not care how the arm as a whole is turned. That
        # is why rotating the shoulder cannot move this reading, and why any
        # coupling seen between them on a real rig is measurement error
        # rather than geometry -- see the epipolar note in triangulate_pose.
        out["elbow_flexion"] = _angle_at(shoulder, elbow, wrist)

    # Needs the hips as well as the arm, so it is gated separately: a
    # torso frame cannot be built from the arm alone.
    out["shoulder_rotation"] = shoulder_heading(pose_world, side)

    if shoulder and elbow and hip:
        out["shoulder_flexion"] = _angle_at(hip, shoulder, elbow)

    # The palm plane, shared by both wrist metrics. Index and pinky sit on
    # opposite sides of a left vs. right hand, so this cross product follows
    # the palm on a right hand but the back of the hand on a left one --
    # mirror it so that "normal" means the palm on both.
    normal = None
    if wrist and index and pinky:
        normal = np.cross(_vec(wrist, index), _vec(wrist, pinky))
        if np.linalg.norm(normal) < 1e-9:
            normal = None
        elif side == "LEFT":
            normal = -normal

    if normal is not None:
        # The palm normal is still needed -- wrist flexion is measured about
        # it. What is gone is the bearing that used to be read off it as
        # "wrist rotation": see the note on wrist_roll in main.py's JOINT_MAP.
        if elbow and wrist:
            # Flexion/extension turns about the side-to-side axis of the
            # wrist. Taking that axis as perpendicular to both the hand's
            # pointing direction and the palm normal, rather than as the raw
            # index->pinky line, keeps it orthogonal to the bend being
            # measured -- so radial/ulnar deviation does not leak into this
            # reading as a spurious flexion.
            forearm = _vec(elbow, wrist)
            hand_dir = (_vec(wrist, index) + _vec(wrist, pinky)) / 2.0
            signed = _signed_angle(forearm, hand_dir, np.cross(hand_dir, normal))
            if signed is not None:
                out["wrist_flexion"] = METRIC_SIGN["wrist_flexion"] * signed

    if hand_world:
        hand_wrist = hand_world[HAND["WRIST"]]

        fingers = [
            ("INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP"),
            ("MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP"),
            ("RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP"),
            ("PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"),
        ]
        if hand_wrist:
            curl_sum, n = 0.0, 0
            for mcp_k, pip_k, dip_k, tip_k in fingers:
                m, p, d, t = hand_world[HAND[mcp_k]], hand_world[HAND[pip_k]], hand_world[HAND[dip_k]], hand_world[HAND[tip_k]]
                straight = _dist(hand_wrist, t)
                path = _dist(hand_wrist, m) + _dist(m, p) + _dist(p, d) + _dist(d, t)
                extension = straight / (path or 1e-6)
                curl_sum += 1 - min(1.0, max(0.0, extension))
                n += 1
            if n > 0:
                out["grip"] = curl_sum / n

    return out


_GhostPt = namedtuple("_GhostPt", "x y z")


def _rodrigues(v, axis, angle_deg):
    """Rotate vector v by angle_deg about axis, exactly (Rodrigues' formula).

    Used to build the ghost target arm below: rotating a vector about the
    same axis its angle-at-a-vertex is measured around changes that angle by
    exactly angle_deg, so "rotate the real, currently-observed segment by
    (target - current)" lands the ghost on the target angle exactly, rather
    than approximating it with a synthetic pose that has to guess anatomy the
    six tracked metrics don't actually specify.
    """
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return v
    k = axis / n
    t = math.radians(angle_deg)
    return (v * math.cos(t) + np.cross(k, v) * math.sin(t)
            + k * np.dot(k, v) * (1 - math.cos(t)))


def ghost_arm_points(pose_world, side, results, errors):
    """Shoulder/elbow/wrist of the pose to match, anchored to the live body.

    Not a synthetic idealised skeleton -- it takes the person's own, actually
    observed upper arm and forearm and rotates them by exactly the amount
    each tracked angle is currently out, about the same axis that angle is
    measured about. That keeps the ghost arm attached to wherever their body
    actually is (torso position, height, camera framing all fall out of the
    computation for free) and shows only the rotation still needed.

    shoulder_flexion and elbow_flexion are each a single exact rotation
    (Rodrigues about the vertex's own rotation axis), and both land on target
    exactly, including together -- the second rotation only bends the elbow
    relative to wherever the first left the (rigidly carried) upper arm, so
    it cannot undo the first. shoulder_rotation is not drawn: it is a swing
    about yet another axis, and forcing the forearm onto it as a third step
    would fight the elbow bend just fixed (same vector, two competing angle
    constraints, no rotation that satisfies both in general). It stays a
    numbers-only reading in the alignment panel instead of a misleading ghost
    limb that can't actually reach where the numbers say it can.

    Returns (shoulder, ghost_elbow, ghost_wrist) as points with .x/.y/.z, in
    the same triangulated frame as pose_world -- or None if there isn't
    enough of either the pose or the target to place it.
    """
    def pt(name):
        return pose_world[POSE[f"{side}_{name}"]] if pose_world else None

    shoulder, elbow, wrist, hip = pt("SHOULDER"), pt("ELBOW"), pt("WRIST"), pt("HIP")
    if not (shoulder and elbow and wrist and hip):
        return None

    def target(key):
        value = (results or {}).get(key, {}).get("value")
        err = (errors or {}).get(key)
        if value is None or err is None:
            return None
        return value - err

    s = np.array([shoulder.x, shoulder.y, shoulder.z or 0.0])
    e = np.array([elbow.x, elbow.y, elbow.z or 0.0])
    w = np.array([wrist.x, wrist.y, wrist.z or 0.0])
    h = np.array([hip.x, hip.y, hip.z or 0.0])
    upper, fore = e - s, w - e

    # Axes matched to _angle_at(a, b, c), which measures from vertex b: its
    # first ray is a-b, not b-a. shoulder_flexion is _angle_at(hip, shoulder,
    # elbow) -> first ray hip-shoulder; elbow_flexion is _angle_at(shoulder,
    # elbow, wrist) -> first ray shoulder-elbow, i.e. -upper. Getting either
    # backwards rotates the correct *magnitude* but the wrong *direction*.
    t_flex = target("shoulder_flexion")
    if t_flex is not None:
        axis = np.cross(h - s, upper)
        delta = t_flex - results["shoulder_flexion"]["value"]
        upper = _rodrigues(upper, axis, delta)
        fore = _rodrigues(fore, axis, delta)   # carried rigidly with the upper arm

    t_elbow = target("elbow_flexion")
    if t_elbow is not None:
        axis = np.cross(-upper, fore)
        fore = _rodrigues(fore, axis, t_elbow - results["elbow_flexion"]["value"])

    ghost_elbow = s + upper
    ghost_wrist = ghost_elbow + fore
    return (_GhostPt(*s), _GhostPt(*ghost_elbow), _GhostPt(*ghost_wrist))


def _shortest_delta_deg(curr, prev):
    """Shortest signed difference between two degree values, so a rotation
    crossing the +-180 wraparound doesn't register as a ~360deg spike."""
    return (((curr - prev + 180) % 360) + 360) % 360 - 180


class JointStateClassifier:
    """Smooths each metric, then classifies it into "high"/"mid"/"low" bands
    (top/bottom bandFraction of range vs. everything between), with
    hysteresis so a value sitting near a threshold doesn't flicker."""

    def __init__(self, smoothing=0.35, band_fraction=0.3, hysteresis=0.05):
        self.smoothing = smoothing
        self.band_fraction = band_fraction
        self.hysteresis = hysteresis
        self.state = {}

    def reset(self):
        self.state = {}

    def update(self, raw_metrics):
        results = {}
        for m in METRICS:
            key = m["key"]
            raw = raw_metrics.get(key)
            prev = self.state.get(key)

            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                results[key] = {"value": None, "state": "none"}
                continue

            smoothed = raw
            if prev and prev["smoothed"] is not None:
                is_deg = m["unit"] == "deg"
                delta = _shortest_delta_deg(raw, prev["smoothed"]) if is_deg else raw - prev["smoothed"]
                smoothed = prev["smoothed"] + self.smoothing * delta

            lo, hi = m["range"]
            span = hi - lo
            high_threshold = hi - self.band_fraction * span
            low_threshold = lo + self.band_fraction * span
            margin = self.hysteresis * span

            prev_band = prev["band"] if prev else "mid"
            if prev_band == "high":
                band = "high" if smoothed >= high_threshold - margin else ("low" if smoothed <= low_threshold else "mid")
            elif prev_band == "low":
                band = "low" if smoothed <= low_threshold + margin else ("high" if smoothed >= high_threshold else "mid")
            else:
                band = "high" if smoothed >= high_threshold else ("low" if smoothed <= low_threshold else "mid")

            self.state[key] = {"smoothed": smoothed, "band": band}
            results[key] = {"value": smoothed, "state": band}

        return results


def state_label(metric_def, state):
    if state == "high":
        return metric_def["highLabel"]
    if state == "low":
        return metric_def["lowLabel"]
    if state == "mid":
        return metric_def["midLabel"]
    return "No data"


# --- range of motion recording ----------------------------------------------

ARM_RANGES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "arm_ranges.json")

# Discarded from each end before taking the min and max. The recording runs
# for tens of seconds at 30 Hz, and a single frame where triangulation put a
# joint somewhere impossible would otherwise set an endpoint by itself --
# which is exactly the sort of outlier --trust-inferred produces more of.
RANGE_TRIM = 0.02
# Below this a recording is not a measurement of anything.
RANGE_MIN_SAMPLES = 60
# Keeps a long recording from growing without bound; 20000 is over ten
# minutes at 30 Hz, well past any useful session.
RANGE_MAX_SAMPLES = 20000


class RangeRecorder:
    """Measures the span each metric actually covers, for one person.

    JOINT_MAP's human ranges in main.py are declarations about a generic
    arm -- every one that is wrong is a wrong gain, so a joint saturates
    before you reach the end of your own reach, or never gets near the
    robot's limit. This is the human-side counterpart of the robot's
    --recalibrate: move everything through its full range while it watches.
    """

    def __init__(self):
        self.samples = {m["key"]: [] for m in METRICS}
        self.active = False
        self.started = 0.0
        self.elapsed = 0.0

    def start(self):
        self.samples = {m["key"]: [] for m in METRICS}
        self.active = True
        self.started = time.monotonic()
        self.elapsed = 0.0

    def stop(self):
        self.active = False

    def clear(self):
        self.stop()
        self.samples = {m["key"]: [] for m in METRICS}
        self.elapsed = 0.0

    def update(self, results):
        """One frame of metrics. Only what was actually measured is kept."""
        if not self.active:
            return
        self.elapsed = time.monotonic() - self.started
        for m in METRICS:
            value = results[m["key"]]["value"]
            if value is None:
                continue
            bucket = self.samples[m["key"]]
            if len(bucket) < RANGE_MAX_SAMPLES:
                bucket.append(value)

    def spans(self):
        """Per metric: the trimmed range, and whether it is usable."""
        out = {}
        for m in METRICS:
            key = m["key"]
            values = sorted(self.samples[key])
            n = len(values)
            if n < RANGE_MIN_SAMPLES:
                out[key] = {"n": n, "lo": None, "hi": None, "ok": False,
                            "reason": f"only {n} readings (need {RANGE_MIN_SAMPLES})"}
                continue
            cut = int(n * RANGE_TRIM)
            lo, hi = values[cut], values[n - 1 - cut]
            floor = MIN_GRIP_SPAN if key == "grip" else MIN_HUMAN_SPAN_DEG
            ok = (hi - lo) >= floor
            out[key] = {
                "n": n, "lo": lo, "hi": hi, "ok": ok,
                "raw_lo": values[0], "raw_hi": values[-1],
                "reason": "" if ok else
                          (f"only moved {hi - lo:.2f} of range" if key == "grip"
                           else f"only moved {hi - lo:.0f} deg"),
            }
        return out

    def recorded(self):
        """The spans that passed, in the shape main.py reads."""
        return {key: [round(v["lo"], 3), round(v["hi"], 3)]
                for key, v in self.spans().items() if v["ok"]}


# Must match main.py; a span under these is a gain high enough to throw a
# joint across its travel on a degree of arm movement.
MIN_HUMAN_SPAN_DEG = 20.0
MIN_GRIP_SPAN = 0.15


def save_arm_ranges(recorder):
    """Write the recording out. Returns (path, payload) or (None, reason)."""
    payload = recorder.recorded()
    if not payload:
        return None, "nothing moved far enough to record"
    payload["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["samples"] = max((v["n"] for v in recorder.spans().values()), default=0)
    # Which definitions these numbers measure. Without it, a range recorded
    # under older geometry loads silently as a gain for an angle that no
    # longer means the same thing.
    payload["geometry"] = GEOMETRY_VERSION
    os.makedirs(os.path.dirname(ARM_RANGES_FILE), exist_ok=True)
    with open(ARM_RANGES_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    return ARM_RANGES_FILE, payload


# --- robot link (port of robotLink.js) ---------------------------------------

class RobotLink:
    """Websocket client for main.py's `--input webcam` control bridge.

    Sends continuous joint angles at ~15 Hz rather than the three-band
    high/mid/low classification it used to: the bands threw away almost
    everything the triangulation measured, needing something like 50 degrees
    of elbow movement before the robot noticed.

    Bidirectional, because the robot owns the engagement state machine but
    this process owns the screen you are looking at while you line up with
    it. Receives are non-blocking -- the render loop cannot afford to wait on
    a socket -- and drain to the newest message, since an older state frame
    is of no interest once a newer one has arrived.
    """

    def __init__(self):
        self.ws = None
        self.status = "disconnected"
        self.last_send_ms = 0
        self.send_interval_ms = 66
        self.pending_request = None
        self.pending_jog = []
        self.pending_ranges = None
        self.robot_state = None

    def connect(self, url):
        import websockets.sync.client as ws_sync

        self.disconnect()
        self.status = "connecting"
        try:
            self.ws = ws_sync.connect(url, open_timeout=3)
            self.status = "connected"
        except Exception:
            self.ws = None
            self.status = "error"

    def disconnect(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        if self.status != "error":
            self.status = "disconnected"

    @property
    def is_connected(self):
        return self.ws is not None

    def request(self, name):
        """Queue a one-shot request (engage / disengage)."""
        self.pending_request = name

    def jog(self, joint, direction):
        """Queue one jog nudge for the arm.

        A list, not a latch: each press has to land, and two presses are not
        the same as one. Bounded so that jogging with the link down piles up
        a nudge or two to deliver rather than a minute of stored movement.
        """
        if len(self.pending_jog) < 32:
            self.pending_jog.append({"joint": joint, "direction": direction})

    def save_pose(self):
        if len(self.pending_jog) < 32:
            self.pending_jog.append({"save": True})

    def send_ranges(self, ranges):
        """Hand a fresh range recording to the bridge.

        Sent as well as written to disk so it takes effect now rather than at
        the next restart -- the whole point of recording it is to try it.
        """
        self.pending_ranges = ranges

    def send(self, angles, tracking, now_ms):
        """Forward the current angles, plus any queued request or jog.

        A queued request is only cleared once it has actually gone out, so
        pressing E while the link is throttled or briefly down does not
        silently drop the keypress. Jogs override the send throttle for the
        same reason: a nudge that waits 66 ms for the next scheduled frame
        feels like a button that did not work.
        """
        if self.ws is None:
            return
        if (now_ms - self.last_send_ms < self.send_interval_ms
                and not self.pending_request and not self.pending_jog
                and not self.pending_ranges):
            return
        self.last_send_ms = now_ms
        payload = {"angles": angles, "tracking": bool(tracking),
                   "request": self.pending_request, "jog": self.pending_jog,
                   "ranges": self.pending_ranges}
        try:
            self.ws.send(json.dumps(payload))
            self.pending_request = None
            self.pending_jog = []
            self.pending_ranges = None
        except Exception:
            self.disconnect()
            self.status = "error"

    def poll(self):
        """Drain any waiting state frames, keeping the newest. Never blocks."""
        if self.ws is None:
            return
        for _ in range(8):   # bounded, so a fast producer cannot stall the loop
            try:
                message = self.ws.recv(timeout=0)
            except TimeoutError:
                return
            except Exception:
                self.disconnect()
                self.status = "error"
                return
            try:
                self.robot_state = json.loads(message)
            except ValueError:
                pass


# --- model download ------------------------------------------------------

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "calibration", "stereo_calibration.json")


def adapt_intrinsics(K, from_size, to_size):
    """Adapt a camera matrix measured at one capture mode to another.

    Same aspect ratio means the sensor is simply resampled, so a uniform
    scale is exact.

    Different aspect ratios are NOT a matter of scaling x and y separately.
    A UVC webcam switching aspect ratio keeps one axis' field of view and
    centre-crops the other, so the principal point shifts by the crop as
    well. Measured on these cameras: 640x480 is a centred horizontal crop of
    the 1280x720 view with identical vertical FOV -- feature-matching the two
    modes recovered scale 1.504 (=720/480) and offset 159.1 px, against the
    160 px this model predicts.

    Still prefer calibrating at the resolution you track at: this models the
    common behaviour, not a guarantee about a particular camera.
    """
    (fw, fh), (tw, th) = from_size, to_size

    # Vertical FOV is the invariant on these cameras, so height alone fixes
    # the scale. Width then follows: crop_x is positive when the target is
    # narrower (sensor cropped in) and negative when it is wider (FOV gained),
    # which makes the same expression correct in both directions. For a pure
    # resolution change at the same aspect ratio it falls out as zero, leaving
    # a plain uniform scale.
    s = th / fh
    crop_x = (fw - tw / s) / 2

    K = K.copy()
    K[0, 2] -= crop_x
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], float) @ K


def load_calibration(path, frame_sizes=None):
    """Load calibrate_cameras.py's output, or None if it isn't there yet.

    Returns per-camera K/dist plus the extrinsics, and the two projection
    matrices triangulation needs, expressed in camera 1's frame:
    P1 = K1 [I|0] and P2 = K2 [R|T].
    """
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)

    cams = []
    for i, cam in enumerate(data["cameras"]):
        K = np.array(cam["K"], float)
        calib_size = tuple(cam["image_size"])
        if frame_sizes and tuple(frame_sizes[i]) != calib_size:
            K = adapt_intrinsics(K, calib_size, frame_sizes[i])
        cams.append({"K": K, "dist": np.array(cam["dist"], float),
                     "calib_size": calib_size, "name": cam.get("name", f"cam{i + 1}")})

    R = np.array(data["stereo"]["R"], float)
    T = np.array(data["stereo"]["T"], float).reshape(3, 1)
    P1 = cams[0]["K"] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = cams[1]["K"] @ np.hstack([R, T])
    return {"cameras": cams, "R": R, "T": T, "P1": P1, "P2": P2,
            "board": data.get("board", {}), "report": data.get("report", {})}


class _Point:
    """Landmark-shaped holder so triangulated points can flow through the
    same geometry code as MediaPipe's landmarks."""
    __slots__ = ("x", "y", "z")

    def __init__(self, v):
        self.x, self.y, self.z = float(v[0]), float(v[1]), float(v[2])


def _confidence(landmark):
    """How much this landmark can be trusted, as min(visibility, presence).

    Both fields are optional and either may be legitimately 0.0, so absent
    values are filtered explicitly rather than with `or` -- a falsy 0.0 would
    otherwise be replaced by 1.0 and turn a fully occluded joint into a
    fully trusted one.
    """
    values = [v for v in (getattr(landmark, "visibility", None),
                          getattr(landmark, "presence", None)) if v is not None]
    return min(values) if values else 1.0


def triangulate_pose(calibration, poses_norm, sizes, min_visibility=0.3,
                     max_reproj_px=40.0, trust_inferred=False):
    """Recover true 3D pose landmarks from the two calibrated views.

    This is the payoff of calibrating: instead of trusting each model's
    guessed monocular depth, every joint is intersected from two rays whose
    geometry is known, so depth comes from triangulation rather than from a
    network's prior. Correspondence is free -- both views report the same 33
    landmark indices -- so no feature matching is needed.

    Returns (points, reproj_px), where points is a list of 33 _Point in
    metres re-centred on the hip midpoint (matching MediaPipe world-landmark
    convention, so the same views and angle code apply), with None wherever a
    joint was too poorly seen to trust. reproj_px is the mean reprojection
    error, a direct read on how well the calibration and the two detections
    agree.

    With trust_inferred, the visibility and reprojection gates stop deleting
    joints and only label them: every landmark comes back, and info["inferred"]
    marks the ones that would otherwise have been dropped. Two cameras 60
    degrees apart cannot both see an arm held across the body, so on some rigs
    the strict gates leave the metrics with nothing at all -- a guessed
    landmark is then better than no reading, as long as nothing downstream
    mistakes it for a measured one. What it is NOT is free: MediaPipe infers
    an occluded joint from body proportions, so what gets triangulated is the
    intersection of one real ray and one plausible guess, and it will look
    perfectly steady while being wrong. The gates still run, and the labels
    are the whole point of keeping them.

    Two things are still dropped, because they are not points rather than
    being uncertain ones: a degenerate intersection, and anything that
    resolves behind camera 1.
    """
    info = {"reason": None, "reproj": None, "n_valid": 0,
            "lost_visibility": 0, "lost_reproj": 0}

    missing = [i + 1 for i, p in enumerate(poses_norm) if p is None]
    if missing:
        info["reason"] = f"no pose in cam {','.join(map(str, missing))}"
        return None, info

    cams = calibration["cameras"]
    pix, vis = [], []
    for pose, (w, h), cam in zip(poses_norm, sizes, cams):
        pix.append(np.array([[lm.x * w, lm.y * h] for lm in pose], np.float64))
        vis.append(np.array([_confidence(lm) for lm in pose]))

    # Undistort into the same ideal-pixel frame the projection matrices are
    # expressed in, so lens distortion doesn't bend the rays being intersected.
    und = [cv2.undistortPoints(p.reshape(-1, 1, 2), c["K"], c["dist"], P=c["K"]).reshape(-1, 2)
           for p, c in zip(pix, cams)]

    X = cv2.triangulatePoints(calibration["P1"], calibration["P2"], und[0].T, und[1].T)
    w_ = X[3]
    valid = np.abs(w_) > 1e-9
    pts_mm = np.zeros((X.shape[1], 3))
    pts_mm[valid] = (X[:3, valid] / w_[valid]).T

    # A joint only counts if both cameras actually saw it; one occluded view
    # gives a confidently wrong intersection rather than a missing one.
    seen = (vis[0] > min_visibility) & (vis[1] > min_visibility)
    info["lost_visibility"] = int((valid & ~seen).sum())
    suspect = valid & ~seen
    if not trust_inferred:
        valid &= seen
    # Not a confidence judgement: a point behind the camera that saw it is
    # not a worse estimate of the joint, it is not a position at all, and the
    # angles built from it would be meaningless rather than merely noisy.
    valid &= pts_mm[:, 2] > 0

    # Reproject every point back into both views. A joint that lands far from
    # where either camera actually saw it was not really the same joint --
    # usually because the two cameras grabbed at different instants and the
    # arm moved between them, or because one view mislocalised it. Those
    # points are dropped rather than averaged into the result.
    reproj = None
    per_point = np.full(len(pts_mm), np.inf)
    if valid.any():
        errs = []
        for (p, c, rt) in ((pix[0], cams[0], (np.zeros(3), np.zeros(3))),
                           (pix[1], cams[1], (cv2.Rodrigues(calibration["R"])[0], calibration["T"]))):
            proj, _ = cv2.projectPoints(pts_mm[valid], rt[0], rt[1], c["K"], c["dist"])
            errs.append(np.linalg.norm(proj.reshape(-1, 2) - p[valid], axis=1))
        worst = np.maximum(errs[0], errs[1])
        per_point[valid] = worst
        reproj = float(np.mean(np.concatenate(errs)))
        info["reproj"] = reproj
        keep = per_point <= max_reproj_px
        info["lost_reproj"] = int((valid & ~keep).sum())
        suspect |= valid & ~keep
        if not trust_inferred:
            valid &= keep

    # Keep the evidence, not just the tally. "24/33 joints" cannot answer the
    # only question worth asking -- whether the six landmarks the metrics
    # actually need came through -- and a joint can fail either gate for
    # opposite reasons: seen by neither camera, or seen by both and placed
    # somewhere the two views cannot reconcile.
    # Which surviving joints only survived because the gates were relaxed.
    # Everything downstream that presents a number to a human reads this.
    info["inferred"] = (suspect & valid).tolist()
    info["trust_inferred"] = bool(trust_inferred)
    info["confidence"] = [vis[0].tolist(), vis[1].tolist()]
    info["per_joint_reproj"] = [None if not np.isfinite(v) else float(v) for v in per_point]
    info["valid"] = valid.tolist()

    if not valid.any():
        info["reason"] = ("all joints below visibility threshold"
                          if info["lost_visibility"] else "all joints failed reprojection")
        return None, info

    # Re-centring only sets where the skeleton is drawn -- angles are
    # translation-invariant -- so a missing hip must not discard an otherwise
    # good triangulation. Prefer the hip midpoint to match MediaPipe's own
    # convention, then the shoulders, then whatever was resolved.
    origin = None
    for pair in ([POSE["LEFT_HIP"], POSE["RIGHT_HIP"]],
                 [POSE["LEFT_SHOULDER"], POSE["RIGHT_SHOULDER"]]):
        if all(valid[i] for i in pair):
            origin = pts_mm[pair].mean(axis=0)
            break
    if origin is None:
        origin = pts_mm[valid].mean(axis=0)

    info["n_valid"] = int(valid.sum())
    pts_m = (pts_mm - origin) / 1000.0  # mm -> metres, hip-centred
    return [(_Point(p) if ok else None) for p, ok in zip(pts_m, valid)], info


# The landmarks compute_joint_metrics needs from the tracked side. Anything
# else triangulating is nice but changes no reading, which is why the raw
# "n/33 joints" count is a poor guide to whether tracking is working.
TRACKED_LANDMARKS = ("SHOULDER", "ELBOW", "WRIST", "HIP", "INDEX", "PINKY")


def joint_diagnosis(info, side, min_visibility, max_reproj_px):
    """Why each landmark the metrics need is, or is not, triangulated.

    Both cameras drawing a skeleton is not evidence that a joint was seen.
    MediaPipe always returns all 33 landmarks, including the ones it inferred
    from the rest of the body rather than observed, and the overlay draws
    those identically -- a joint hidden behind your torso from one camera
    still gets a confident-looking dot. The visibility numbers here are the
    model's own estimate of which is which, and they are what the gate reads.

    The other way to fail is stranger: seen clearly by both cameras, but the
    two rays do not meet where either view says the joint is. With the
    cameras 62 degrees apart that is usually genuine disagreement about a
    foreshortened limb, or the ~33 ms the two free-running cameras are out of
    step, during which a moving wrist travels far enough to matter.
    """
    out = []
    for name in TRACKED_LANDMARKS:
        idx = POSE[f"{side}_{name}"]
        entry = {"name": name.capitalize(), "ok": False, "trusted": False,
                 "reason": "no pose detected", "cam1": None, "cam2": None,
                 "reproj": None}
        if info and info.get("valid"):
            c1, c2 = info["confidence"][0][idx], info["confidence"][1][idx]
            err = info["per_joint_reproj"][idx]
            entry.update(cam1=round(c1, 2), cam2=round(c2, 2),
                         reproj=None if err is None else round(err, 1))
            if info["valid"][idx] and not info["inferred"][idx]:
                entry.update(ok=True, trusted=True, reason="triangulated")
            elif info["valid"][idx]:
                # Kept under --trust-inferred. Say which gate it failed: an
                # occluded joint and a joint the two views cannot reconcile
                # are wrong in different ways and at different magnitudes.
                why = (f"camera {'1' if c1 <= c2 else '2'} only {min(c1, c2):.2f} sure"
                       if min(c1, c2) <= min_visibility else
                       f"views {err:.0f} px apart")
                entry.update(ok=True, reason=f"inferred — {why}")
            elif min(c1, c2) <= min_visibility:
                which = "camera 1" if c1 <= c2 else "camera 2"
                entry["reason"] = (f"{which} only {min(c1, c2):.2f} sure it saw this "
                                   f"(needs {min_visibility:.2f})")
            elif err is not None and err > max_reproj_px:
                entry["reason"] = (f"the two views place it {err:.0f} px apart "
                                   f"(limit {max_reproj_px:.0f})")
            else:
                entry["reason"] = "resolved behind camera 1"
        out.append(entry)
    return out


def ensure_model(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {os.path.basename(dest_path)}...")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


# --- drawing --------------------------------------------------------------

PANEL_W = 430
PANEL_BG = (33, 27, 23)
PANEL_BORDER = (53, 44, 38)
TEXT = (239, 233, 230)
MUTED = (161, 146, 138)
ACCENT = (255, 140, 79)
STATE_HIGH = (129, 196, 51)
STATE_LOW = (74, 158, 255)
TRI_COLOR = (120, 255, 160)   # the fused, triangulated skeleton
GHOST_COLOR = (0, 210, 255)   # the pose to match, before engagement

VIEW_BG = (26, 21, 18)
# Per-camera colours. Still used for each feed's title bar and its 2D
# landmark overlay -- those show what each camera actually detected, which is
# how you diagnose a triangulation that is dropping joints. The 3D views no
# longer draw per-camera skeletons at all.
CAM_COLORS = [(255, 140, 79), (120, 130, 255)]
# Metres spanned by a view's full height. Sized for the upper body these
# views actually draw (hips to raised hands, roughly a metre), not a whole
# standing person, so the skeleton fills the panel.
VIEW_METRES = 1.3

VIEW_AXES = {
    # mode: (title, footer describing where the viewer is standing)
    "front": ("FRONT VIEW", "looking from camera 1"),
    "side": ("SIDE VIEW", "camera 1 at left"),
    "top": ("TOP VIEW", "camera 1 at bottom"),
}


def to_px(landmark, w, h):
    """True (unmirrored) pixel position, matching the raw camera frame that
    detection ran on."""
    return int(landmark.x * w), int(landmark.y * h)


def draw_label(view, anchor_px, lines, color=TEXT):
    """Small translucent caption pinned beside a point.

    Clamped to stay inside the view: the joints these annotate are often near
    an edge, and an unclamped box would slide out of frame exactly when the
    arm is somewhere worth reading.
    """
    lines = [l for l in lines if l]
    if not lines or anchor_px is None:
        return
    h, w = view.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, line_h, pad = 0.44, 1, 15, 5
    text_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    box_w, box_h = text_w + pad * 2, len(lines) * line_h + pad
    x = min(max(anchor_px[0] + 9, 0), max(0, w - box_w))
    y = min(max(anchor_px[1] - box_h // 2, 0), max(0, h - box_h))

    patch = view[y:y + box_h, x:x + box_w]
    if patch.size:
        patch[:] = (patch * 0.3).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(view, line, (x + pad, y + pad + line_h * (i + 1) - 4),
                    font, scale, color, thick, cv2.LINE_AA)


def draw_landmarks(frame, pose, hands):
    """Draws skeleton lines/dots in true (unmirrored) coordinates, onto the
    raw frame *before* it's flipped for display. Lines/dots are symmetric,
    so flipping the whole frame afterward keeps them correctly aligned with
    the body without any coordinate math here."""
    h, w = frame.shape[:2]

    if pose:
        for a, b in POSE_CONNECTIONS:
            pa, pb = pose[a], pose[b]
            cv2.line(frame, to_px(pa, w, h), to_px(pb, w, h), (255, 140, 79), 3, cv2.LINE_AA)
        for idx in (POSE["LEFT_SHOULDER"], POSE["RIGHT_SHOULDER"], POSE["LEFT_ELBOW"],
                    POSE["RIGHT_ELBOW"], POSE["LEFT_WRIST"], POSE["RIGHT_WRIST"]):
            cv2.circle(frame, to_px(pose[idx], w, h), 5, (239, 233, 230), -1, cv2.LINE_AA)

    for landmarks in hands:
        if not landmarks:
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, to_px(landmarks[a], w, h), to_px(landmarks[b], w, h), (129, 196, 51), 3, cv2.LINE_AA)


# --- replay -------------------------------------------------------------------

# How a replayed frame is turned back into readings. Three depths into the
# same capture, and the reason the raw detections are kept at all:
#   stored   -- the angles exactly as they were recorded. What the robot was
#               actually told that day; the only mode whose numbers are not
#               produced by today's code.
#   points   -- angles recomputed from the recorded xyz. Tests changed
#               geometry against an unchanged triangulation.
#   cameras  -- xyz re-triangulated from the recorded 2D with whatever
#               calibration is loaded now, then angles from that. Tests a new
#               calibration, or changed visibility/reprojection gates,
#               against a capture taken under the old ones.
DERIVE_MODES = ("stored", "points", "cameras")
DERIVE_LABELS = {
    "stored": "as recorded",
    "points": "angles recomputed from the recorded 3D points",
    "cameras": "re-triangulated from the recorded camera views",
}

# Size the replayed camera panes are drawn at. These are line drawings of
# recorded landmarks, not video, so there is no detail to preserve -- only
# enough room to see which camera lost the arm.
REPLAY_PANE = (640, 360)


def replay_camera_view(pose_norm, name, index, color, size=REPLAY_PANE, note="",
                       footage=None):
    """One camera's recorded detection, drawn as its feed would have been.

    The video itself is not recorded -- it would dwarf everything else in the
    file and answers no question the landmarks do not. What is worth seeing
    on replay is exactly what was kept: where each camera put the body, and
    which one stopped seeing it. Drawn through the same draw_landmarks and
    flipped the same way, so a replayed pane and a live one can be compared
    without allowing for a difference in how they were made.
    """
    w, h = size
    if footage is not None:
        # Landmarks drawn over the frame they were read from, at the
        # footage's own size, so the two cannot disagree about where the
        # body was. Scaled down only afterwards, for the same reason the
        # live path draws before it resizes.
        view = footage if footage.shape[1] <= w else cv2.resize(
            footage, (w, round(footage.shape[0] * w / footage.shape[1])),
            interpolation=cv2.INTER_AREA)
        h = view.shape[0]
    else:
        view = np.zeros((h, w, 3), np.uint8)
        view[:] = (18, 16, 14)
    if pose_norm:
        draw_landmarks(view, pose_norm, [None, None])
    view = cv2.flip(view, 1)
    cv2.rectangle(view, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(view, f"{name}  (index {index})", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    label = note or (("REPLAY" if footage is not None else "REPLAY - landmarks only")
                     if pose_norm else "REPLAY - no pose recorded")
    cv2.putText(view, label, (w - 8 - 7 * len(label), 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                MUTED if pose_norm else STATE_LOW, 1, cv2.LINE_AA)
    return view


def derive_replay_frame(rec, index, mode, calibration, min_visibility,
                        max_reproj_px, trust_inferred, side):
    """Turn one recorded frame back into (points, info, raw metrics).

    Returns raw, unsmoothed metrics: smoothing is stateful and belongs to the
    caller's classifier, so that replaying at 4x or stepping frame by frame
    filters the same sequence the live loop would have.
    """
    frame = rec.frames[index]
    if mode == "cameras" and calibration:
        poses = rec.poses_norm(index)
        if len(poses) == 2 and all(p is not None for p in poses):
            points, info = triangulate_pose(
                calibration, poses, rec.sizes(), min_visibility=min_visibility,
                max_reproj_px=max_reproj_px, trust_inferred=trust_inferred)
            hand = rec.hand(index)
            return points, info, compute_joint_metrics(points, hand, side)
        # Fall through: a frame where a camera saw nothing cannot be
        # re-triangulated, and silently showing the stored answer instead
        # would misreport which mode produced the number.
        return None, rec.info(index), {}
    points, info = rec.points(index), rec.info(index)
    if mode == "stored":
        return points, info, {k: v for k, v in (frame.get("m") or {}).items()}
    return points, info, compute_joint_metrics(points, rec.hand(index), side)


def replay_results(classifier, raw, smooth):
    """Band-classify replayed metrics, smoothing only when they are raw.

    The stored values were already smoothed once, when they were recorded;
    running them through the filter again would show a reading the live
    session never produced, and would make "as recorded" the one mode that
    does not report what was recorded. Banding still runs either way, since
    that is presentation rather than filtering -- so the pass-through is a
    smoothing factor of 1.0 rather than a separate code path that could drift
    from the real one.
    """
    if smooth:
        return classifier.update(raw)
    saved = classifier.smoothing
    classifier.smoothing = 1.0
    try:
        return classifier.update(raw)
    finally:
        classifier.smoothing = saved


# The reading shown beside each joint in the 3D views: (landmark, metric,
# prefix). One number per joint so the views stay readable -- the panel
# carries the full set of six. The prefix matters because in some poses two
# joints project to nearly the same point (an arm hanging straight down
# collapses shoulder and elbow in the top view), and bare numbers there are
# indistinguishable.
JOINT_READOUTS = [("SHOULDER", "shoulder_flexion", "SH"),
                  ("ELBOW", "elbow_flexion", "EL"),
                  ("WRIST", "wrist_flexion", "WR")]

# Height of the front/side/top row. The row is as wide as the feeds above it,
# so this is the only lever on how large the skeleton is drawn.
VIEW_ROW_H = 420


def _dashed_line(view, p1, p2, color, thickness=2, dash=8, gap=6):
    """Dashed segment, so the ghost target reads as a target and not as a
    second real skeleton competing with the solid, measured one."""
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1e-6:
        return
    direction = (p2 - p1) / length
    dist, draw = 0.0, True
    while dist < length:
        seg = min(dash if draw else gap, length - dist)
        if draw:
            a, b = p1 + direction * dist, p1 + direction * (dist + seg)
            cv2.line(view, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)
        dist += seg
        draw = not draw


def draw_pose_view(width, height, pose_world, mode, side, metrics=None, status=None, ghost=None):
    """Orthographic projection of the triangulated skeleton.

    "front" looks down camera 1's axis, "side" looks along the body's
    left/right axis, "top" is a bird's-eye view. Every mode is expressed in
    camera 1's frame.

    Only the fused skeleton is drawn. Overlaying the two monocular estimates
    on top of it made three skeletons in one pane, and the two that were
    guessing at depth were the ones drawing the eye -- while the metrics and
    the robot were being driven by the third.

    World landmarks are metres about the hip midpoint, so this uses a fixed
    metres-per-pixel scale anchored on that origin -- the skeleton keeps a
    stable size and position rather than rescaling itself every frame.
    """
    view = np.full((height, width, 3), VIEW_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = height / VIEW_METRES
    ox = width // 2
    oy = height // 2 if mode == "top" else int(height * 0.62)

    # Faint crosshair marking the hip-midpoint origin the projection is
    # anchored to, so a still skeleton still reads as positioned.
    cv2.line(view, (0, oy), (width, oy), PANEL_BORDER, 1)
    cv2.line(view, (ox, 0), (ox, height), PANEL_BORDER, 1)

    title, footer = VIEW_AXES[mode]
    cv2.putText(view, title, (10, 18), font, 0.42, MUTED, 1, cv2.LINE_AA)
    cv2.putText(view, footer, (10, height - 9), font, 0.35, MUTED, 1, cv2.LINE_AA)

    def project(lm):
        # x is negated wherever it is a screen axis, so these views mirror the
        # same way the displayed video does: raising a hand moves it to the
        # same side in both. In "top", depth is negated too, which puts
        # near-the-camera at the bottom.
        if mode == "top":
            return int(ox - lm.x * scale), int(oy - lm.z * scale)
        if mode == "front":
            return int(ox - lm.x * scale), int(oy + lm.y * scale)
        return int(ox + lm.z * scale), int(oy + lm.y * scale)

    if not pose_world:
        # With only one skeleton on screen there is no other trace to fall
        # back on, so say *why* it is empty rather than leaving a blank pane.
        cv2.putText(view, status or "no triangulated pose", (12, oy - 6),
                    font, 0.42, MUTED, 1, cv2.LINE_AA)
        return view

    tracked = {
        (POSE[f"{side}_SHOULDER"], POSE[f"{side}_ELBOW"]),
        (POSE[f"{side}_ELBOW"], POSE[f"{side}_WRIST"]),
    }

    for a, b in POSE_CONNECTIONS:
        if pose_world[a] is None or pose_world[b] is None:
            continue
        highlight = (a, b) in tracked or (b, a) in tracked
        cv2.line(view, project(pose_world[a]), project(pose_world[b]),
                 TRI_COLOR, 3 if highlight else 1, cv2.LINE_AA)
    for idx in (POSE["LEFT_SHOULDER"], POSE["RIGHT_SHOULDER"], POSE["LEFT_ELBOW"],
                POSE["RIGHT_ELBOW"], POSE["LEFT_WRIST"], POSE["RIGHT_WRIST"]):
        if pose_world[idx] is not None:
            cv2.circle(view, project(pose_world[idx]), 3, TEXT, -1, cv2.LINE_AA)

    # The angle at each joint, written next to that joint. A number beside
    # the geometry it describes is far quicker to sanity-check than the same
    # number in a list somewhere else on screen.
    if metrics:
        for joint, key, prefix in JOINT_READOUTS:
            lm = pose_world[POSE[f"{side}_{joint}"]]
            value = metrics.get(key, {}).get("value")
            if lm is None or value is None:
                continue
            draw_label(view, project(lm), [f"{prefix} {value:.0f}deg"], TRI_COLOR)

    if ghost:
        g_shoulder, g_elbow, g_wrist = ghost
        _dashed_line(view, project(g_shoulder), project(g_elbow), GHOST_COLOR, 2)
        _dashed_line(view, project(g_elbow), project(g_wrist), GHOST_COLOR, 2)
        for p in (g_elbow, g_wrist):
            cv2.circle(view, project(p), 4, GHOST_COLOR, 2, cv2.LINE_AA)
        draw_label(view, project(g_wrist), ["TARGET"], GHOST_COLOR)

    return view


def build_pose_views(width, height, pose_world, side, metrics=None, status=None, ghost=None):
    """The front/side/top projections laid out as one row."""
    w = width // 3
    widths = [w, w, width - 2 * w]   # last pane absorbs the rounding remainder
    views = [draw_pose_view(pw, height, pose_world, mode, side, metrics, status, ghost)
             for pw, mode in zip(widths, ("front", "side", "top"))]
    row = np.hstack(views)
    for i in (1, 2):
        cv2.line(row, (i * w, 0), (i * w, height), PANEL_BORDER, 1)
    return row


def wrap_text(text, font, scale, max_w, thick=1):
    """Split text into lines that fit max_w pixels."""
    lines, current = [], ""
    for word in (text or "").split(" "):
        trial = f"{current} {word}".strip()
        if current and cv2.getTextSize(trial, font, scale, thick)[0][0] > max_w:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def draw_alignment(panel, y, robot_state):
    """The engage handshake: how far each joint is from the arm's pose.

    Drawn per joint rather than as a single "aligned / not aligned" light,
    because the useful question while you are standing there holding a pose
    is *which* joint is still out and which way to move it. A bare refusal
    to engage leaves you guessing.

    Returns the y the caller should continue from.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    state = robot_state.get("state", "?")
    colors = {"ENGAGED": STATE_HIGH, "READY": ACCENT, "HOMING": STATE_LOW}

    cv2.line(panel, (12, y), (PANEL_W - 12, y), PANEL_BORDER, 1)
    y += 20
    cv2.putText(panel, "ROBOT", (12, y), font, 0.4, MUTED, 1, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(state, font, 0.5, 1)
    cv2.putText(panel, state, (PANEL_W - 12 - sw, y), font, 0.5,
                colors.get(state, MUTED), 1, cv2.LINE_AA)
    y += 20

    if state == "ENGAGED":
        cv2.putText(panel, "tracking your arm   -   D to release", (12, y),
                    font, 0.4, MUTED, 1, cv2.LINE_AA)
        return y + 8

    # A stalled joint is worth saying out loud here: the arm cannot reach its
    # park pose, so alignment will never complete and the reason is mechanical
    # rather than anything you can fix by moving your arm.
    for joint in robot_state.get("stalled") or []:
        cv2.putText(panel, f"{joint} is not moving", (12, y),
                    font, 0.4, STATE_LOW, 1, cv2.LINE_AA)
        y += 15

    errors = robot_state.get("errors") or {}
    tol = robot_state.get("tolerance") or {}
    angle_tol = tol.get("angle", 15.0)
    grip_tol = tol.get("grip", 0.25)
    for m in METRICS:
        key = m["key"]
        if key not in errors:
            continue
        delta = errors[key]
        limit = grip_tol if key == "grip" else angle_tol
        label = m["label"].replace(" Ext/Flex", "")
        if delta is None:
            text, color = "no reading", STATE_LOW
        else:
            ok = abs(delta) <= limit
            color = STATE_HIGH if ok else ACCENT
            # An arrow is faster to act on than a signed number when you are
            # holding a pose and cannot study the panel.
            arrow = "ok" if ok else ("+" if delta < 0 else "-")
            text = f"{arrow} {abs(delta):.0f}" if not ok else "ok"
        cv2.putText(panel, label, (12, y), font, 0.4, MUTED, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, font, 0.4, 1)
        cv2.putText(panel, text, (PANEL_W - 12 - tw, y), font, 0.4, color, 1, cv2.LINE_AA)
        y += 15

    y += 4
    if robot_state.get("aligned"):
        cv2.putText(panel, "ALIGNED  -  press E to engage", (12, y),
                    font, 0.45, STATE_HIGH, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "match the arm's pose", (12, y),
                    font, 0.42, MUTED, 1, cv2.LINE_AA)
    y += 16
    # Whatever the robot last said -- including why it refused to engage.
    # This used to go only to main.py's console, which is not the screen you
    # are looking at while holding a pose in front of the cameras.
    for line in wrap_text(robot_state.get("message") or "", font, 0.4, PANEL_W - 24)[:2]:
        cv2.putText(panel, line, (12, y), font, 0.4, ACCENT, 1, cv2.LINE_AA)
        y += 14
    return y + 8


def build_panel(height, results, active, robot_status, title="METRICS", fps=None,
                calib_status="none", band_fraction=0.3, robot_state=None):
    """The six metric readouts.

    The number is what you are actually reading, so it is set large and
    coloured by the band it falls in; the label beside it only says which
    joint it belongs to. Each bar carries ticks at the two band thresholds,
    which is what turns "142 deg" into "...and that is why the robot is
    holding Extended" without having to remember where the boundaries sit.
    """
    panel = np.full((height, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, title, (12, 16), font, 0.4, MUTED, 1, cv2.LINE_AA)
    if fps is not None:
        fps_text = f"{fps:4.1f} fps"
        (fw, _), _ = cv2.getTextSize(fps_text, font, 0.4, 1)
        cv2.putText(panel, fps_text, (PANEL_W - 12 - fw, 16), font, 0.4, MUTED, 1, cv2.LINE_AA)
    y = 44

    banner = "  ·  ".join(active) if active else "All joints neutral"
    banner_color = STATE_HIGH if active else MUTED
    # Wrapped rather than truncated: which joints are off-neutral is the whole
    # point of the line, and cutting it mid-phrase silently drops one.
    lines, current = [], ""
    for word in banner.split(" "):
        trial = f"{current} {word}".strip()
        if current and cv2.getTextSize(trial, font, 0.46, 1)[0][0] > PANEL_W - 24:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    for line in lines[:2]:
        cv2.putText(panel, line, (12, y), font, 0.46, banner_color, 1, cv2.LINE_AA)
        y += 19
    y += 7
    cv2.line(panel, (12, y - 10), (PANEL_W - 12, y - 10), PANEL_BORDER, 1)

    bar_x, bar_w, bar_h = 12, PANEL_W - 24, 8
    # Spread the six rows over whatever is left above the footer, so the panel
    # fills the frame height instead of bunching at the top under a gap. This
    # also absorbs a banner that wrapped onto a second line, and the extra
    # depth the alignment block needs while the robot is waiting to engage.
    footer_h = 108
    if robot_state:
        if robot_state.get("state") == "ENGAGED":
            footer_h += 74
        else:
            # header + one row per metric + one per stalled joint + 2 message lines
            footer_h += 74 + 15 * (len(METRICS) + len(robot_state.get("stalled") or []))
    else:
        footer_h += 30
    row_fixed = 22 + 14 + bar_h + 15 + 8
    slack = (height - footer_h - 16) - y - len(METRICS) * row_fixed
    # Negative slack is allowed to *compress* the rows rather than being
    # clamped to zero: with the alignment block on, the footer is tall enough
    # that the rows would otherwise overrun it and print the last state label
    # through the divider. The floor stops the compression eating the row.
    extra = max(-8, slack // len(METRICS))
    for m in METRICS:
        r = results[m["key"]]
        if r["value"] is None:
            val_text, dir_text, state_color = "--", "No data", MUTED
        else:
            val_text = f"{r['value']:.0f}deg" if m["unit"] == "deg" else f"{r['value']:.2f}"
            dir_text = state_label(m, r["state"])
            state_color = (STATE_HIGH if r["state"] == "high" else
                           STATE_LOW if r["state"] == "low" else TEXT)

        y += 22
        cv2.putText(panel, m["label"], (12, y), font, 0.44, TEXT, 1, cv2.LINE_AA)
        (val_w, _), _ = cv2.getTextSize(val_text, font, 0.95, 2)
        cv2.putText(panel, val_text, (PANEL_W - 12 - val_w, y + 3), font, 0.95,
                    state_color, 2, cv2.LINE_AA)
        y += 14

        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), PANEL_BORDER, 1)
        # The classifier's own band boundaries: below the left tick reads
        # "low", above the right one "high", between them "mid".
        for frac in (band_fraction, 1.0 - band_fraction):
            tx = bar_x + int(bar_w * frac)
            cv2.line(panel, (tx, y - 2), (tx, y + bar_h + 2), MUTED, 1)
        if r["value"] is not None:
            lo, hi = m["range"]
            pct = max(0.0, min(1.0, (r["value"] - lo) / (hi - lo)))
            cv2.rectangle(panel, (bar_x, y), (bar_x + int(bar_w * pct), y + bar_h),
                          state_color, -1)
        y += bar_h + 15
        cv2.putText(panel, dir_text, (12, y), font, 0.42, state_color, 1, cv2.LINE_AA)
        y += 8 + extra

    y = height - footer_h
    if robot_state:
        y = draw_alignment(panel, y, robot_state)
    elif robot_status == "connected":
        cv2.line(panel, (12, y), (PANEL_W - 12, y), PANEL_BORDER, 1)
        cv2.putText(panel, "ROBOT   waiting for state...", (12, y + 20),
                    font, 0.4, MUTED, 1, cv2.LINE_AA)
        y += 30
    else:
        # E does nothing without a link, and silence there is the same
        # symptom as a broken handshake. Say which it is.
        cv2.line(panel, (12, y), (PANEL_W - 12, y), PANEL_BORDER, 1)
        cv2.putText(panel, "ROBOT   not connected  -  press C", (12, y + 20),
                    font, 0.4, STATE_LOW, 1, cv2.LINE_AA)
        y += 30

    cv2.line(panel, (12, y), (PANEL_W - 12, y), PANEL_BORDER, 1)
    y += 20
    cv2.putText(panel, "CALIBRATION", (12, y), font, 0.4, MUTED, 1, cv2.LINE_AA)
    y += 20
    calib_color = STATE_HIGH if calib_status.startswith("loaded") else MUTED
    cv2.putText(panel, calib_status, (12, y), font, 0.42, calib_color, 1, cv2.LINE_AA)

    y += 26
    cv2.putText(panel, "ROBOT LINK", (12, y), font, 0.4, MUTED, 1, cv2.LINE_AA)
    y += 22
    status_color = STATE_HIGH if robot_status == "connected" else (STATE_LOW if robot_status == "error" else MUTED)
    cv2.putText(panel, robot_status, (12, y), font, 0.45, status_color, 1, cv2.LINE_AA)

    return panel


def match_hand_to_side(hand_landmarks_list, pose):
    """Match each detected hand to the LEFT/RIGHT pose wrist by nearest
    image-space distance, since MediaPipe's own handedness label can disagree
    with pose left/right depending on framing.

    Returns {side: index into the hand results}, so the caller can look the
    match up in both the normalized and world landmark lists.
    """
    matches = {"LEFT": None, "RIGHT": None}
    if not hand_landmarks_list:
        return matches

    targets = {
        "LEFT": pose[POSE["LEFT_WRIST"]] if pose else None,
        "RIGHT": pose[POSE["RIGHT_WRIST"]] if pose else None,
    }

    for i, landmarks in enumerate(hand_landmarks_list):
        hand_wrist = landmarks[0]
        best_side, best_dist = None, float("inf")
        for side in ("LEFT", "RIGHT"):
            t = targets[side]
            if t is None:
                continue
            d = math.hypot(hand_wrist.x - t.x, hand_wrist.y - t.y)
            if d < best_dist:
                best_dist, best_side = d, side
        if best_side and best_dist < 0.15:
            matches[best_side] = i
    return matches


class CameraPipeline:
    """One camera together with its own dedicated pose/hand landmarkers.

    The landmarkers cannot be shared between cameras: in VIDEO running mode
    each one carries frame-to-frame tracking state and expects a single
    monotonically increasing timestamp series, so feeding it two interleaved
    streams would corrupt both. Every camera therefore gets its own models,
    its own timestamp counter and its own smoothing/classifier state --
    which is also exactly what "two entirely independent pipelines" means.
    """

    def __init__(self, camera, name, color, pose_model, hand_model, delegate,
                 hands=True):
        self.camera = camera
        self.index = camera.index
        self.name = name
        self.color = color

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        ))
        # Optional, because it is the most expendable quarter of the CPU
        # budget. Grip is the only metric that needs it, and grip is read from
        # one camera (see --metrics-cam) even with both running -- the pose
        # model has no finger joints, so there is no triangulated form of it
        # to fuse. On a second camera the hand model therefore buys nothing
        # but the hand skeleton drawn on that camera's own feed, at the price
        # of a whole extra inference per frame on every frame.
        self.hand_landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )) if hands else None

        self.classifier = JointStateClassifier()
        self.last_ts_ms = -1
        self.pose_norm = None      # 2D, for triangulation
        self.pose_world = None     # this camera's own monocular estimate
        self.hand_world = None     # still needed: grip has no triangulated form
        self.results = None
        self.raw = {}              # unsmoothed, for recordings

    def read(self):
        """Grab one frame. Capture itself lives in camera_io.Camera so this
        and calibrate_cameras.py behave identically."""
        return self.camera.read()

    def process(self, frame, side):
        """Run this camera's models on one frame and return it annotated
        (and mirrored), with the metrics stored on the pipeline."""
        if frame is None:
            return None
        frame = frame.copy()

        ts_ms = max(int(time.perf_counter() * 1000), self.last_ts_ms + 1)
        self.last_ts_ms = ts_ms

        mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_result = self.pose_landmarker.detect_for_video(mp_image, ts_ms)
        hand_result = (self.hand_landmarker.detect_for_video(mp_image, ts_ms)
                       if self.hand_landmarker else None)

        pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        self.pose_norm = pose
        self.pose_world = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None

        hands = (match_hand_to_side(hand_result.hand_landmarks, pose)
                 if hand_result else {"LEFT": None, "RIGHT": None})
        hand_idx = hands[side]
        hand_world = hand_result.hand_world_landmarks[hand_idx] if hand_idx is not None else None
        self.hand_world = hand_world

        frame_h, frame_w = frame.shape[:2]
        # This camera's own monocular estimate. It no longer drives anything
        # when triangulation is running -- it is kept because it is what the
        # feed's overlay shows, and because it is the fallback when the two
        # cameras cannot agree on a pose at all.
        raw = compute_joint_metrics(self.pose_world, hand_world, side)
        # Kept unsmoothed as well: a recording stores both, so a capture can
        # be re-filtered offline without the classifier's state baked in.
        self.raw = raw
        self.results = self.classifier.update(raw)

        drawn_hands = [
            hand_result.hand_landmarks[i] if i is not None else None
            for i in (hands["LEFT"], hands["RIGHT"])
        ] if hand_result else [None, None]
        draw_landmarks(frame, pose, drawn_hands)
        # No angle text on the feeds. These are per-camera monocular numbers,
        # and printing them beside the panel's triangulated ones put two
        # different values for the same joint on screen at once.
        frame = cv2.flip(frame, 1)  # natural selfie view

        # Title bar identifying which feed this is, in the camera's colour.
        cv2.rectangle(frame, (0, 0), (frame_w, 22), (0, 0, 0), -1)
        cv2.putText(frame, f"{self.name}  (index {self.index})", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.color, 1, cv2.LINE_AA)
        if self.pose_world is None:
            cv2.putText(frame, "no pose", (frame_w - 70, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1, cv2.LINE_AA)
        if self.camera.health:
            cv2.putText(frame, self.camera.health, (frame_w - 190, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, STATE_LOW, 1, cv2.LINE_AA)
        return frame

    def close(self):
        self.camera.release()
        self.pose_landmarker.close()
        if self.hand_landmarker:
            self.hand_landmarker.close()


# Jog targets the page may name. Must match main.py's ARM_JOINTS + gripper;
# main.py validates them again, since it is the one holding the arm.
JOG_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper")

# The 3D projection row, when served to a browser. Independent of the loop
# rate: it is redrawn from scratch every time, which is the expensive half.
WEB_VIEWS_HZ = 8
WEB_VIEWS_W = 960

# How often the recordings directory is re-listed for the page. Disk I/O in
# the capture loop, and the only thing that changes it mid-session is a
# recording this process just finished writing.
LISTING_EVERY_S = 3.0

# How many frames of history are replayed through the filter before a frame
# that was jumped to, so it reads the same as it did live. The smoothing
# factor is 0.35, so after 20 frames 0.65**20 = 0.02% of whatever the filter
# held before the jump survives -- below the rounding the value is shown at.
REPLAY_PRIME_FRAMES = 20

KEY_ACTIONS = {
    ord("q"): "quit", 27: "quit", ord(" "): "side", ord("t"): "triangulation",
    ord("v"): "views", ord("c"): "link", ord("e"): "engage", ord("d"): "disengage",
    ord("i"): "trust",
}


def build_web_state(results, side, source, fps, robot_link, tri_status,
                    calib_status, tri_available, tri_enabled, tri_live,
                    cams, show_views, joints=(), trust_inferred=False, ranges=None,
                    cameras=None, capture=None, replay=None):
    """Everything the browser page renders as text, as plain JSON.

    Deliberately the same numbers build_panel draws, from the same dicts, so
    the two front ends cannot drift apart -- the page is a different
    rendering of this state, not a second source of it.
    """
    return {
        "fps": fps,
        "side": side,
        "source": source,
        "views": bool(show_views),
        "link": {"status": robot_link.status, "connected": robot_link.is_connected},
        "robot": robot_link.robot_state if robot_link.is_connected else None,
        "triangulation": {
            "available": bool(tri_available),
            "enabled": bool(tri_enabled),
            "live": bool(tri_live),
            "status": tri_status,
            "calibration": calib_status,
            "trustInferred": bool(trust_inferred),
            # Per-landmark, for the side being tracked: the answer to "both
            # cameras can see me, so why is nothing triangulating".
            "joints": list(joints),
        },
        # Enough per camera to tell a working one from a stuck one without
        # reading the console: which stream to pull, what it is delivering,
        # and whether this camera in particular can see a pose right now.
        # "Both cameras opened" and "both cameras are contributing" are not
        # the same claim, and only the second one matters to triangulation.
        # Human range-of-motion recording: the counterpart of the robot's
        # --recalibrate, for the other end of the map.
        "ranges": ranges or {},
        # Session capture and playback. Both are always present so the page
        # can offer recording before anything has been recorded, and can
        # offer to open a file with no cameras attached at all.
        "capture": capture or {},
        "replay": replay or {},
        # A replayed frame describes the cameras of the recording, which may
        # be nothing like the ones attached now -- or attached at all -- so
        # the caller may supply the list rather than it being read off the
        # live pipelines.
        "cameras": cameras if cameras is not None else [{
            "name": c.name,
            "index": c.index,
            "stream": f"cam{i}",
            "size": list(c.camera.size),
            "health": c.camera.health,
            "stale": c.camera.stale_frames,
            "pose": bool(c.pose_norm),
        } for i, c in enumerate(cams)],
        "metrics": [{
            "key": m["key"],
            "label": m["label"],
            # For the alignment table, where the column is narrow and the
            # direction is already in the reading. Not dropped entirely, as
            # the OpenCV panel does: "Shoulder" alone is ambiguous next to
            # "Shoulder Rotation", and both can be listed at once.
            "short": m["label"].replace(" Ext/Flex", " Flex"),
            "unit": m["unit"],
            "min": m["range"][0],
            "max": m["range"][1],
            "value": results[m["key"]]["value"],
            "state": results[m["key"]]["state"],
            "stateLabel": state_label(m, results[m["key"]]["state"]),
        } for m in METRICS],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDICES[0], help="OpenCV index of the front camera")
    parser.add_argument("--camera2", type=int, default=DEFAULT_CAMERA_INDICES[1],
                        help="OpenCV index of the side camera; -1 to run with one camera")
    parser.add_argument("--side", choices=["LEFT", "RIGHT"], default="RIGHT", help="initially tracked arm")
    parser.add_argument("--metrics-cam", type=int, choices=[1, 2], default=1,
                        help="which camera's estimate drives the metrics panel and the robot")
    parser.add_argument("--robot-host", default="localhost:8765", help="host:port of main.py --input webcam")
    parser.add_argument("--robot", action="store_true", help="connect to the robot bridge on startup")
    parser.add_argument("--delegate", choices=["CPU", "GPU"], default="CPU", help="MediaPipe inference delegate")
    # Four model invocations per frame on two cameras is the bulk of the
    # frame budget, and one of the four is nearly free to drop: see the note
    # in CameraPipeline.__init__.
    parser.add_argument("--hands", choices=["metrics", "all", "off"], default="metrics",
                        help="which cameras run the hand model: the one driving grip "
                             "(default), all of them, or none (disables grip)")
    parser.add_argument("--raw-format", action="store_true", help="don't force MJPEG capture (use if your camera doesn't support it)")
    parser.add_argument("--no-views", action="store_true", help="start with the 3D projection row hidden")
    parser.add_argument("--web", action="store_true",
                        help="serve the UI in a browser instead of opening an OpenCV window")
    parser.add_argument("--web-port", type=int, default=8080, help="port for --web")
    # Localhost by default: this page drives a robot arm, and binding it to
    # every interface is a decision to make deliberately, not a default.
    parser.add_argument("--web-host", default="127.0.0.1",
                        help="interface for --web; 0.0.0.0 to reach it from another device")
    # Detection always runs on the full-resolution frame; this is only how
    # big the picture of yourself is. 640 is plenty to line up against, and
    # the cheapest thing to encode while two MediaPipe graphs want the CPU.
    parser.add_argument("--web-width", type=int, default=640,
                        help="width to stream each camera at in --web mode")
    parser.add_argument("--scale", default="auto",
                        help="window scale: 'auto' fits it to your screen, or give a number like 0.75")
    # MediaPipe runs its landmark model on a crop around the detected person,
    # so a higher-resolution frame puts more real detail into that crop --
    # which is what the arm joints need. Calibrate at whatever is set here:
    # intrinsics from another resolution are adapted automatically, but that
    # adaptation is a model of how the camera changes mode, not a measurement.
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    # The two cameras free-run on independent clocks, so their frames are
    # exposed up to one frame period apart -- 33 ms at 30 fps, 17 ms at 60.
    # That offset is what puts a moving arm in two different places in the two
    # views, and triangulating across it is what max-reproj has to throw out.
    # Asking for a higher rate is the only lever software has on it; genuinely
    # removing it needs hardware-triggered cameras.
    parser.add_argument("--fps", type=int, default=30,
                        help="frame rate to request (higher = less skew between the cameras)")
    parser.add_argument("--calibration", default=CALIBRATION_PATH, help="stereo calibration JSON from calibrate_cameras.py")
    # MediaPipe reports low visibility for joints it is inferring rather than
    # clearly seeing. 0.5 rejects most elbows/wrists when the arm is near the
    # frame edge, so the default is looser and tunable.
    parser.add_argument("--min-visibility", type=float, default=0.3,
                        help="per-view landmark confidence needed to triangulate a joint")
    parser.add_argument("--max-reproj", type=float, default=40.0,
                        help="max reprojection error (px) before a triangulated joint is discarded")
    parser.add_argument("--trust-inferred", action="store_true",
                        help="keep every joint instead of dropping the ones that fail the "
                             "visibility or reprojection gate; they are labelled as inferred "
                             "rather than discarded (also toggleable with I)")
    # --- recording and replay ---
    parser.add_argument("--record", action="store_true",
                        help="start recording the session to a file immediately")
    parser.add_argument("--record-name", metavar="NAME",
                        help="filename for --record (default: session-<timestamp>.jsonl)")
    parser.add_argument("--record-note", default="",
                        help="free-text note stored in the recording's header, "
                             "e.g. what the session was testing")
    # Measured at ~8.8 MB/min uncompressed and ~0.44 MB/min gzipped, so the
    # saving is real. It is not the default because an interrupted plain file
    # is readable with any tool, which for a capture someone stood up to
    # produce is worth more than the disk.
    parser.add_argument("--record-compress", action="store_true",
                        help="gzip the recording (~20x smaller, ~0.4 MB/min)")
    # Off by default: it is most of the bytes. What it buys is the one thing
    # landmarks cannot give back -- the pose model can be re-run on the raw
    # frames, where stored landmarks are already that model's answer.
    parser.add_argument("--record-video", action="store_true",
                        help="also keep each camera's raw frames as video next to the "
                             "recording (~15-40 MB/min per camera; also toggleable on the page)")
    # Identity protection for recorded subjects. 'block' paints the head out
    # solid and is the one to use for that purpose. 'blur' pixelates and
    # smooths instead, which keeps the frame readable -- but it measurably
    # destroys only the fine detail: skin tone, head shape and pose all
    # survive it, so it reduces identifiability rather than removing it.
    # Either mode also strips the face landmarks from the stored data, since
    # eleven facial coordinates per frame beside a masked video would
    # protect nobody.
    parser.add_argument("--mask-face", choices=recording.MASK_MODES, default="off",
                        help="obscure the subject's face in recorded video and drop the "
                             "face landmarks from the data. 'block' blacks the head out "
                             "and is what to use for anonymity; 'blur' pixelates it, "
                             "which hides detail but leaves skin tone and head shape, "
                             "so it is NOT anonymisation; 'off' (default) keeps faces.")
    parser.add_argument("--video-codec", default="avc1",
                        help="fourcc for --record-video (avc1/H.264 default; "
                             "mp4v and XVID are ~2x larger, MJPG far larger)")
    parser.add_argument("--recordings", default=recording.RECORDINGS_DIR,
                        help="directory recordings are written to and listed from")
    parser.add_argument("--replay", metavar="FILE",
                        help="play a recording back instead of the live cameras; "
                             "a bare filename is looked up in --recordings")
    parser.add_argument("--derive", choices=DERIVE_MODES, default="points",
                        help="how a replayed frame is interpreted: 'stored' shows the "
                             "angles as recorded, 'points' recomputes them from the "
                             "recorded 3D, 'cameras' re-triangulates from the recorded "
                             "2D with the current calibration")
    # Replaying on a machine with no rig attached is the normal case for
    # reviewing a capture, and opening cameras that are not there costs
    # seconds of timeouts before failing.
    parser.add_argument("--no-cameras", action="store_true",
                        help="don't open any camera; serve the UI for replay only")
    parser.add_argument("--dump-canvas", metavar="PATH",
                        help="save the composed window to PATH and exit (for diagnosing display problems)")
    args = parser.parse_args()

    # Reviewing a capture is normally done on a machine nowhere near the rig,
    # so --replay opens no cameras: waiting out the driver timeouts for two
    # that are not plugged in would be the slowest part of starting up. To
    # watch a recording while the cameras run, load it from the page instead.
    no_cameras = args.no_cameras or bool(args.replay)
    if no_cameras and not args.web and not args.dump_canvas:
        args.web = True
        print("No cameras to open, so the UI is served in a browser (--web).")

    specs = [(args.camera, "CAM 1 front")]
    if args.camera2 >= 0:
        specs.append((args.camera2, "CAM 2 side"))
    cams = []
    metrics_index = 0
    if not no_cameras:
        pose_model = ensure_model(POSE_MODEL_URL, os.path.join(MODEL_DIR, "pose_landmarker_lite.task"))
        hand_model = ensure_model(HAND_MODEL_URL, os.path.join(MODEL_DIR, "hand_landmarker.task"))
        delegate = BaseOptions.Delegate.GPU if args.delegate == "GPU" else BaseOptions.Delegate.CPU
        devices = open_cameras(specs, args.width, args.height,
                               force_mjpg=not args.raw_format, fps=args.fps)
        metrics_index = min(args.metrics_cam, len(devices)) - 1
        cams = [
            CameraPipeline(dev, name, CAM_COLORS[i], pose_model, hand_model, delegate,
                           hands=args.hands == "all"
                           or (args.hands == "metrics" and i == metrics_index))
            for i, (dev, (_, name)) in enumerate(zip(devices, specs))
        ]

    frame_sizes = [c.camera.size for c in cams]
    # Without cameras there is no capture resolution to adapt the intrinsics
    # to, so they load unscaled; a recording carries its own sizes and the
    # calibration is re-adapted to those when one is loaded.
    calibration = (load_calibration(args.calibration, frame_sizes or None)
                   if len(cams) > 1 or no_cameras else None)
    if calibration:
        rep = calibration["report"]
        calib_status = f"loaded ({rep.get('stereo_rms_px', float('nan')):.2f} px RMS)"
        print(f"Calibration: {args.calibration}")
        print(f"  baseline {rep.get('baseline_mm', 0):.0f} mm, "
              f"cameras {rep.get('angle_deg', 0):.1f} deg apart")
        for cam, size in zip(calibration["cameras"], frame_sizes):
            if tuple(cam["calib_size"]) != tuple(size):
                print(f"  NOTE: {cam['name']} calibrated at {cam['calib_size'][0]}x{cam['calib_size'][1]} "
                      f"but capturing at {size[0]}x{size[1]}; intrinsics rescaled, which assumes "
                      f"the camera keeps the same field of view. Prefer matching resolutions.")
    else:
        calib_status = "none - run calibrate_cameras.py"
        print(f"No calibration at {args.calibration} (tracking still works; "
              f"triangulation will need it).")

    metrics_cam = cams[metrics_index] if cams else None
    robot_link = RobotLink()
    side = args.side
    show_views = not args.no_views
    window = "Arm Motion Tracker"
    scale = None if args.scale == "auto" else float(args.scale)
    window_sized = False
    # --dump-canvas exists to diagnose the composed window, so it always
    # takes the window path even with --web, which never composes one.
    web = None
    if args.web and not args.dump_canvas:
        from webui import WebUI

        web = WebUI(host=args.web_host, port=args.web_port, max_width=args.web_width)
        try:
            web.start()
        except OSError as exc:
            raise SystemExit(
                f"Cannot serve the web UI on {args.web_host}:{args.web_port}: {exc}\n"
                f"Another tracker may still be running, or pick a port with --web-port.")
    else:
        # NORMAL rather than AUTOSIZE so the window can be resized and can be
        # smaller than the canvas; AUTOSIZE forces it to the image size, which a
        # window manager then clips off-screen with no way to shrink it.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    if args.robot:
        robot_link.connect(f"ws://{args.robot_host}/ws")

    if cams:
        print(f"Running {len(cams)} independent pipeline(s): "
              + ", ".join(f"{c.name}={c.index}" for c in cams))
        print(f"Metrics panel and robot output follow {metrics_cam.name}.")
        without = [c.name for c in cams if c.hand_landmarker is None]
        if without:
            print(f"Hand model off for {', '.join(without)} (--hands all to enable). "
                  f"{'Grip is disabled.' if args.hands == 'off' else 'Grip still comes from ' + metrics_cam.name + '.'}")
    else:
        print("No cameras opened. Load a recording from the page to review one.")
    if web:
        print(f"\nOpen the tracker UI at {web.url}")
        if args.web_host not in ("127.0.0.1", "localhost"):
            print(f"  (reachable from other devices on this network at port {args.web_port})")
        print("Buttons for every control; the keys below work on the page too.")
    print("Controls: SPACE = toggle tracked side, T = toggle triangulation, "
          "I = keep/drop inferred joints, V = toggle 3D views, "
          "C = toggle robot connection, Q/Esc = quit.")
    if args.trust_inferred:
        print("Inferred joints are being KEPT: every landmark triangulates, and the ones\n"
              "neither camera clearly saw are labelled rather than dropped. Readings built\n"
              "on them can be steady and wrong -- check the Triangulation panel.")
    print("Robot handshake: E = engage once matched, D = disengage.")

    fps_ema = None
    last_views = 0.0
    recorder = RangeRecorder()
    range_note = ""
    tri_classifier = JointStateClassifier()
    use_triangulation = calibration is not None
    trust_inferred = args.trust_inferred
    tri_reproj_ema = None
    pool = ThreadPoolExecutor(max_workers=len(cams)) if len(cams) > 1 else None

    # Two different recorders, and they record different things: `recorder`
    # above measures how far each of your joints travels, to calibrate the
    # map, and keeps only the extremes. `session` here keeps every frame, to
    # be replayed and re-analysed later.
    session = recording.Recorder(directory=args.recordings)
    session_note = ""
    session_t0 = 0.0
    session_video = None        # VideoRecorder while capturing footage
    want_video = args.record_video
    mask_mode = args.mask_face
    player = None
    replay_video = None         # VideoSource while replaying one that has it
    replay_calib = None
    replay_classifier = JointStateClassifier()
    derive_mode = args.derive
    replay_note = ""
    replay_cam_state = []
    replay_last = None       # frame index the classifier's state belongs to
    replay_cache = None
    listing = recording.list_recordings(args.recordings)
    last_listing = 0.0

    def invalidate_replay():
        """Forget the filter state, so the next frame rebuilds its history."""
        nonlocal replay_last, replay_cache
        replay_classifier.reset()
        replay_last, replay_cache = None, None

    def replay_frame(idx):
        """(points, info, results) for one replayed frame, filtered as live.

        Smoothing is a running filter, so the reading for a frame depends on
        the frames before it. Two things break that on replay, and both show
        up as a scrubbed frame disagreeing with what the same frame read
        live -- which would make the whole point of the derive modes
        (comparing stored readings against recomputed ones) meaningless,
        since the difference would be filter history rather than code.

        Jumping: after a seek, a step, or playback at 4x skipping frames,
        the filter holds state from somewhere else entirely. So the frames
        just before the target are pushed through it first.

        Holding: while paused the loop keeps running, and re-filtering the
        same frame 30 times a second walks the smoothed value steadily
        toward the raw one -- a paused reading would visibly drift. So a
        frame is derived once and reused until the index actually moves.
        """
        nonlocal replay_last, replay_cache
        if idx == replay_last and replay_cache is not None:
            return replay_cache

        def derive(j):
            return derive_replay_frame(
                player.recording, j, derive_mode, replay_calib or calibration,
                args.min_visibility, args.max_reproj, trust_inferred, side)

        smooth = derive_mode != "stored"
        if replay_last is None or not 0 < idx - replay_last <= 1:
            replay_classifier.reset()
            for j in range(max(0, idx - REPLAY_PRIME_FRAMES), idx):
                replay_results(replay_classifier, derive(j)[2], smooth)
        points, info, raw = derive(idx)
        replay_cache = (points, info,
                        replay_results(replay_classifier, raw, smooth))
        replay_last = idx
        return replay_cache

    def session_header():
        """Provenance for a capture: enough to interpret it without this code.

        The calibration report goes in whole rather than by reference,
        because the file it came from is the one thing most likely to have
        been replaced by the time anyone reads the recording back -- and a
        capture whose geometry cannot be identified is not evidence of
        anything.
        """
        shape = [{"name": c.name, "index": c.index, "size": list(c.camera.size),
                  "hands": c.hand_landmarker is not None} for c in cams]
        return recording.build_header(
            side=side,
            cameras=shape or [{"name": c[1], "index": c[0], "size": []} for c in specs],
            metrics=METRICS, landmark_names=POSE,
            calibration_report=(calibration or {}).get("report"),
            calibration_path=args.calibration,
            video=({"files": session_video.paths, "codec": args.video_codec,
                    "fps": args.fps, "sizes": [list(c.camera.size) for c in cams],
                    "faceMask": mask_mode}
                   if session_video else None),
            privacy={"faceMask": mask_mode,
                     "faceLandmarksRemoved": mask_mode != "off",
                     "faceLandmarkIndices": list(recording.FACE_LANDMARKS)},
            settings={
                "min_visibility": args.min_visibility,
                "max_reproj": args.max_reproj,
                "trust_inferred": trust_inferred,
                "hands": args.hands,
                "metrics_cam": metrics_cam.name if metrics_cam else None,
                "fps_requested": args.fps,
            },
            note=args.record_note)

    def session_action(name):
        """Start or stop capturing this session. Returns a status line."""
        nonlocal session_t0, session_video, want_video, mask_mode
        if name.startswith("rec-mask:"):
            mode = name.split(":", 1)[1]
            if mode not in recording.MASK_MODES:
                return session_note
            if session.active:
                return ("Already recording — stop first to change face masking. "
                        "Changing it mid-capture would leave one file with two "
                        "different privacy guarantees.")
            mask_mode = mode
            return {"off": "Faces will NOT be masked.",
                    "block": "Faces will be blacked out, and face landmarks dropped.",
                    "blur": "Faces will be pixelated, and face landmarks dropped. "
                            "Pixelation leaves skin tone and head shape, so it is "
                            "not anonymisation — use 'blacked out' for that."}[mode]
        if name == "rec-video":
            # Footage is decided when a capture opens its files, so this is
            # a preference for the next one rather than a live switch.
            if session.active:
                return ("Already recording — stop first to change whether "
                        "footage is kept.")
            want_video = not want_video
            return ("The next recording will keep the raw video too."
                    if want_video else "The next recording will keep landmarks only.")
        if name == "rec-start":
            if session.active:
                return session_note
            # The filename is fixed first, because the video files are named
            # after it and the header has to name them in turn.
            base = args.record_name or f"session-{time.strftime('%Y%m%d-%H%M%S')}"
            note = ""
            if want_video and cams:
                session_video = recording.VideoRecorder(
                    os.path.join(args.recordings, base),
                    [c.camera.size for c in cams], fps=args.fps,
                    codec=args.video_codec, mask=mask_mode)
                try:
                    os.makedirs(args.recordings, exist_ok=True)
                    session_video.start()
                except OSError as exc:
                    # A missing codec must not cost the capture itself --
                    # the landmarks are the part that cannot be re-made.
                    session_video = None
                    note = (f" Video is off: {exc}. Try --video-codec mp4v.")
            try:
                path = session.start(session_header(), name=base,
                                     compress=args.record_compress)
            except OSError as exc:
                if session_video:
                    session_video.stop()
                    session_video = None
                return f"Could not start recording: {exc}"
            session_t0 = time.monotonic()
            kept = (f" with video ({', '.join(session_video.paths)})"
                    if session_video else "")
            return f"Recording to {os.path.basename(path)}{kept}.{note}"
        # rec-stop
        path = session.stop()
        vid = session_video.stop() if session_video else None
        session_video = None
        if not path:
            return session_note
        size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
        if vid:
            here = os.path.dirname(path)
            size += sum(os.path.getsize(os.path.join(here, f)) / 1e6
                        for f in vid["files"] if os.path.exists(os.path.join(here, f)))
        lost = sum(vid["dropped"]) if vid else 0
        return (f"Saved {os.path.basename(path)} — {session.frames} frames, "
                f"{session.elapsed:.0f}s, {size:.1f} MB"
                + (f" including video" if vid else "")
                + (f"; {lost} video frames dropped under load" if lost else "") + ".")

    def load_replay(name):
        """Open a recording for playback. Returns a status line."""
        nonlocal player, replay_calib, derive_mode, replay_video
        path = name if os.path.isabs(name) else os.path.join(args.recordings, name)
        try:
            rec = recording.Recording.load(path)
        except (OSError, ValueError) as exc:
            return f"Cannot open {os.path.basename(path)}: {exc}"
        if not rec.count:
            return f"{rec.name} has no frames."
        player = recording.Player(rec)
        if replay_video:
            replay_video.close()
        # Footage lives in separate files beside the .jsonl, so it may not
        # have travelled with it. A recording without its video is still a
        # recording; it just replays as the stick figure.
        replay_video = rec.open_video()
        invalidate_replay()
        # Intrinsics belong to the resolution they were captured at, so a
        # recording made at a different size than the cameras currently
        # report needs its own adaptation -- otherwise re-triangulation is
        # done with a focal length for the wrong frame.
        sizes = rec.sizes()
        replay_calib = (load_calibration(args.calibration, sizes)
                        if all(s and len(s) == 2 and all(s) for s in sizes) and len(sizes) == 2
                        else calibration)
        if derive_mode == "cameras" and not replay_calib:
            derive_mode = "points"
        player.play()
        return (f"Playing {rec.name} — {rec.count} frames, {rec.duration:.1f}s "
                f"at {rec.fps:.0f} fps, {rec.header.get('side', '?')} arm.")

    def replay_action(name):
        """Transport controls for the loaded recording."""
        nonlocal player, derive_mode, replay_calib, replay_video
        if name.startswith("replay-load:"):
            return load_replay(name.split(":", 1)[1])
        if name == "replay-close":
            player = None
            replay_calib = None
            if replay_video:
                replay_video.close()
                replay_video = None
            invalidate_replay()
            return "Replay closed." if cams else "Replay closed. No cameras are running."
        if name.startswith("replay-derive:"):
            mode = name.split(":", 1)[1]
            if mode not in DERIVE_MODES:
                return replay_note
            if mode == "cameras" and not (replay_calib or calibration):
                return "No calibration loaded, so the camera views cannot be re-triangulated."
            derive_mode = mode
            invalidate_replay()
            return f"Showing angles {DERIVE_LABELS[mode]}."
        if player is None:
            return replay_note
        if name == "replay-play":
            player.toggle()
            return replay_note
        if name == "replay-loop":
            player.loop = not player.loop
            return replay_note
        # Parsed defensively: these carry a number from the page, and a
        # malformed one must not take down a loop that may be holding a
        # robot arm.
        try:
            if name.startswith("replay-seek:"):
                player.seek_fraction(float(name.split(":", 1)[1]))
            elif name.startswith("replay-step:"):
                player.step(int(name.split(":", 1)[1]))
            elif name.startswith("replay-speed:"):
                player.set_speed(float(name.split(":", 1)[1]))
        except ValueError:
            return f"Ignored a malformed replay command: {name}"
        return replay_note

    def record_action(name):
        """Range recording: start, stop, save, clear. Returns a status line."""
        nonlocal range_note
        if name == "range-start":
            recorder.start()
            return ("Recording your range of motion. Move every joint through "
                    "its full comfortable travel, then stop.")
        if name == "range-stop":
            recorder.stop()
            short = [k for k, v in recorder.spans().items() if not v["ok"]]
            return ("Stopped. Ready to save." if not short else
                    f"Stopped. Not enough movement in: {', '.join(short)}.")
        if name == "range-clear":
            recorder.clear()
            return "Recording discarded."
        # range-save
        recorder.stop()
        path, payload = save_arm_ranges(recorder)
        if path is None:
            return f"Nothing saved: {payload}."
        if robot_link.is_connected:
            robot_link.send_ranges({k: v for k, v in payload.items()
                                    if isinstance(v, list)})
            return f"Saved to {os.path.basename(path)} and sent to the robot."
        return (f"Saved to {os.path.basename(path)}. The robot is not "
                f"connected, so it takes effect when it next starts.")

    def do_action(name):
        """Apply one control action, whoever asked for it.

        Both front ends funnel through here -- an OpenCV keypress and a
        button in the browser are the same six commands, and they touch
        state (the tracked side, the classifiers, the robot link) that only
        this loop may own. Returns False to stop the loop.
        """
        nonlocal side, use_triangulation, show_views, scale, window_sized
        nonlocal trust_inferred, session_note, replay_note
        if name == "quit":
            return False
        if name == "side":
            side = "LEFT" if side == "RIGHT" else "RIGHT"
            for cam in cams:
                cam.classifier.reset()
            tri_classifier.reset()
            invalidate_replay()
            print(f"Tracking side: {side}")
        elif name == "triangulation" and calibration:
            use_triangulation = not use_triangulation
            tri_classifier.reset()
            print(f"triangulated metrics {'on' if use_triangulation else 'off'}")
        elif name == "trust":
            trust_inferred = not trust_inferred
            tri_classifier.reset()
            # Re-triangulation reads this gate, so a replayed frame derived
            # under the old setting is no longer what this setting produces.
            invalidate_replay()
            print(f"inferred joints {'kept' if trust_inferred else 'dropped'}")
        elif name == "views":
            show_views = not show_views
            if args.scale == "auto":
                scale = None  # canvas changed shape; refit on the next frame
            window_sized = False
        elif name == "link":
            if robot_link.is_connected:
                robot_link.disconnect()
            else:
                robot_link.connect(f"ws://{args.robot_host}/ws")
        elif name.startswith("range-"):
            range_note = record_action(name)
            print(range_note)
        elif name.startswith("rec-"):
            session_note = session_action(name)
            print(session_note)
        elif name.startswith("replay-"):
            replay_note = replay_action(name)
            if replay_note:
                print(replay_note)
        elif name.startswith("jog:") and robot_link.is_connected:
            # "jog:<joint>:<+|->", forwarded to main.py, which owns the arm
            # and every limit on it. Validated here too so a stray action
            # name never reaches the bridge as a joint it has to reject.
            _, joint, direction = name.split(":", 2)
            if joint in JOG_JOINTS:
                robot_link.jog(joint, 1 if direction == "+" else -1)
        elif name == "save-pose" and robot_link.is_connected:
            robot_link.save_pose()
        elif name in ("engage", "disengage") and robot_link.is_connected:
            # The engage handshake is driven from the UI you are looking at
            # while you line yourself up, not from main.py's terminal.
            robot_link.request(name)
        return True

    if args.record:
        session_note = session_action("rec-start")
        print(session_note)
    if args.replay:
        replay_note = load_replay(args.replay)
        print(replay_note)
        if player is None:
            raise SystemExit(replay_note)

    try:
        while True:
            loop_start = time.perf_counter()
            replaying = player is not None

            if replaying:
                # A replayed frame replaces the whole capture stage: no
                # cameras are read and no model runs. What each camera saw is
                # redrawn from the recorded landmarks, so the panes still show
                # which view lost the arm -- the one thing about a capture
                # worth looking at that the numbers do not say.
                idx = player.tick()
                tri_points, tri_info, replay_results_now = replay_frame(idx)
                rec_cams = player.recording.header.get("cameras", [])
                poses = player.recording.poses_norm(idx)
                at = player.recording.video_index(idx) or []
                frames = [
                    replay_camera_view(poses[i] if i < len(poses) else None,
                                       c.get("name", f"cam{i + 1}"), c.get("index", i),
                                       CAM_COLORS[i % len(CAM_COLORS)],
                                       footage=(replay_video.frame(i, at[i])
                                                if replay_video and i < len(at) else None))
                    for i, c in enumerate(rec_cams)
                ] if rec_cams else []
                replay_cam_state = [{
                    "name": c.get("name", f"cam{i + 1}"),
                    "index": c.get("index", i),
                    "stream": f"cam{i}",
                    "size": list(c.get("size") or REPLAY_PANE),
                    "health": "from the recording",
                    "stale": 0,
                    "pose": bool(poses[i]) if i < len(poses) else False,
                } for i, c in enumerate(rec_cams)]
            elif cams:
                # Capture first, both cameras concurrently, so neither driver
                # queue waits on the other's inference and the two frames are as
                # close together in time as free-running cameras allow -- every
                # millisecond between them is arm displacement that triangulation
                # sees as disagreement. Only then run the models, also in parallel:
                # MediaPipe releases the GIL during native inference, which is
                # worth ~1.6x on two cameras.
                raw = read_all([cam.camera for cam in cams])
                frames = (list(pool.map(lambda cf: cf[0].process(cf[1], side), zip(cams, raw)))
                          if pool else [cams[0].process(raw[0], side)])
                # The two cameras need not agree on a resolution -- they have
                # different maximum modes, and each may fall back independently.
                # Detection has already run at full resolution, so matching them
                # here is purely so the panes can sit side by side.
                if frames and all(f is not None for f in frames):
                    h_min = min(f.shape[0] for f in frames)
                    frames = [f if f.shape[0] == h_min else
                              cv2.resize(f, (round(f.shape[1] * h_min / f.shape[0]), h_min),
                                         interpolation=cv2.INTER_AREA)
                              for f in frames]
                if any(f is None for f in frames):
                    print("A camera stopped delivering frames.")
                    break
            else:
                # No cameras and nothing loaded: the page is up so a recording
                # can be opened from it. Idle at a low rate rather than
                # spinning a core on an empty loop.
                frames = []
                time.sleep(0.05)

            # Triangulate from the two views. This supersedes the per-camera
            # monocular estimates when it succeeds, because its depth is
            # measured rather than inferred.
            if not replaying:
                tri_points, tri_info = (None, None)
                if calibration and len(cams) == 2:
                    tri_points, tri_info = triangulate_pose(
                        calibration,
                        [c.pose_norm for c in cams],
                        [c.camera.size for c in cams],
                        min_visibility=args.min_visibility,
                        max_reproj_px=args.max_reproj,
                        trust_inferred=trust_inferred,
                    )

            triangulated = False
            frame_raw = {}
            if replaying:
                results = replay_results_now
                triangulated = tri_points is not None
                source = f"REPLAY · {DERIVE_LABELS[derive_mode]}"
            elif not cams:
                results = replay_classifier.update({})
                source = "no cameras"
            else:
                source = metrics_cam.name
                results = metrics_cam.results
                frame_raw = metrics_cam.raw
            if tri_points and use_triangulation and not replaying:
                triangulated = True
                # Five of the six metrics now come entirely from the
                # triangulated points. Only grip still needs the hand model,
                # which has no triangulated equivalent -- Pose carries no
                # finger joints, and grip is a ratio along each finger, so it
                # is the metric least troubled by monocular depth anyway.
                tri_raw = compute_joint_metrics(tri_points, metrics_cam.hand_world, side)
                frame_raw = tri_raw
                results = tri_classifier.update(tri_raw)
                source = "TRIANGULATED"

            # Capture, if recording. Only ever the live path: re-recording a
            # replay would write a file derived from another one, which looks
            # identical afterwards and is not a measurement of anything.
            if session.active and not replaying and cams:
                # The RAW frames, before draw_landmarks and the mirror flip:
                # burnt-in overlays are exactly what would stop the footage
                # being re-runnable through a pose model later.
                video_at = (session_video.add(raw, [c.pose_norm for c in cams])
                            if session_video else None)
                session.add(recording.build_frame(
                    index=session.frames, t=time.monotonic() - session_t0,
                    points=tri_points, info=tri_info, results=results,
                    raw=frame_raw, poses_norm=[c.pose_norm for c in cams],
                    hand_world=metrics_cam.hand_world if metrics_cam else None,
                    triangulated=triangulated, source=source,
                    stale=[c.camera.stale_frames for c in cams],
                    video=video_at, drop_face=mask_mode != "off"))
                if session.error:
                    session_note = f"Recording stopped: {session.error}"
                    print(session_note)

            recorder.update(results)
            active = [
                f"{m['label'].replace(' Ext/Flex', '')}: {state_label(m, results[m['key']]['state'])}"
                for m in METRICS if results[m["key"]]["state"] in ("high", "low")
            ]
            if tri_info and tri_info.get("reproj") is not None:
                r = tri_info["reproj"]
                tri_reproj_ema = r if tri_reproj_ema is None else 0.9 * tri_reproj_ema + 0.1 * r
            if replaying:
                # During replay this line describes the recording, not the
                # rig: the cameras it names may not even be plugged in.
                tri_status = (f"{tri_info['n_valid']}/33 joints from the recording"
                              if tri_points and tri_info else
                              (tri_info or {}).get("reason") or "no points in this frame")
            elif not cams:
                # Not a failure to report: there is nothing attached to fail.
                tri_status = "idle - no cameras"
            elif not calibration:
                tri_status = calib_status
            elif not use_triangulation:
                tri_status = "off (press T)"
            elif tri_points:
                tri_status = f"{tri_info['n_valid']}/33 joints"
                if tri_reproj_ema is not None:
                    tri_status += f", {tri_reproj_ema:.0f}px"
                dropped = tri_info["lost_visibility"] + tri_info["lost_reproj"]
                if dropped and trust_inferred:
                    tri_status += f" ({dropped} inferred)"
                elif dropped:
                    tri_status += f" (-{tri_info['lost_visibility']}vis -{tri_info['lost_reproj']}err)"
            else:
                tri_status = tri_info["reason"] if tri_info else "no data"
            robot_link.poll()
            # The browser renders all of this as text from the state feed, so
            # there is nothing to draw -- and nothing to squeeze into 430px.
            panel = None if web or not frames else build_panel(
                frames[0].shape[0], results, active, robot_link.status,
                title=f"METRICS - {source}", fps=fps_ema,
                calib_status=tri_status,
                band_fraction=tri_classifier.band_fraction,
                robot_state=robot_link.robot_state)

            # Continuous angles, not bands. Only send what was actually
            # measured this frame: a metric that came back None is absent
            # rather than zero, so the robot holds its last target for that
            # joint instead of being commanded to the middle of its range.
            angles = {m["key"]: results[m["key"]]["value"]
                      for m in METRICS if results[m["key"]]["value"] is not None}
            # "Tracking" means the fused skeleton is driving these numbers.
            # A monocular fallback is not good enough to hand a robot: its
            # depth is guessed, which is the whole reason for triangulating.
            robot_link.send(angles, triangulated and tri_points is not None,
                            int(loop_start * 1000))

            def make_views(width):
                """The front/side/top row.

                Only the fused skeleton, with its own readings beside each
                joint. When it is missing the views say why rather than
                falling back to the monocular traces, which look plausible
                and are exactly what triangulation exists to stop trusting.

                Before engagement, also draw the posture that matches the
                arm's current pose: the answer to "how do I line my body up
                with the robot" when the arm is parked somewhere that looks
                nothing like a human standing at rest. Once engaged the ghost
                would just be the stale pose you already matched, so it stops.
                """
                ghost = None
                robot_state = robot_link.robot_state
                if (use_triangulation and tri_points and robot_state
                        and robot_state.get("state") in ("HOMING", "READY")):
                    ghost = ghost_arm_points(tri_points, side, results, robot_state.get("errors"))
                return build_pose_views(width, VIEW_ROW_H,
                                        tri_points if (use_triangulation or replaying) else None,
                                        side, metrics=results, status=tri_status, ghost=ghost)

            if web:
                # One stream per camera rather than a composed row. The page
                # lays them out itself, at whatever size the screen has; more
                # to the point, a camera that has stopped delivering is then
                # visibly its own dead pane instead of half of a picture.
                for i, frame in enumerate(frames):
                    web.publish_frame(f"cam{i}", frame)
                # The projections are a diagnostic you glance at, not
                # something you track your own motion in, and drawing them
                # costs more than encoding them does -- so they go out at a
                # fraction of the loop rate, and only for a browser that is
                # actually pulling that stream.
                if (show_views and web.wants("views")
                        and loop_start - last_views >= 1.0 / WEB_VIEWS_HZ):
                    last_views = loop_start
                    web.publish_frame("views", make_views(WEB_VIEWS_W))
                if loop_start - last_listing >= LISTING_EVERY_S:
                    # Re-read the directory on a timer, not every frame: it
                    # is disk I/O in the capture loop, and the only thing
                    # that changes it is a recording this process just made.
                    last_listing = loop_start
                    listing = recording.list_recordings(args.recordings)
                web.publish_state(build_web_state(
                    results, side, source, fps_ema, robot_link,
                    tri_status=tri_status, calib_status=calib_status,
                    tri_available=calibration is not None,
                    tri_enabled=use_triangulation, tri_live=triangulated,
                    cams=cams, show_views=show_views, trust_inferred=trust_inferred,
                    cameras=replay_cam_state if replaying else None,
                    ranges={"recording": recorder.active,
                            "seconds": round(recorder.elapsed, 1),
                            "note": range_note,
                            "file": os.path.basename(ARM_RANGES_FILE),
                            "spans": recorder.spans()},
                    capture={"recording": session.active,
                             "file": os.path.basename(session.path or ""),
                             "frames": session.frames,
                             "seconds": round(session.elapsed, 1),
                             "megabytes": round(session.bytes_written / 1e6, 2),
                             "note": session_note,
                             "available": bool(cams) and not replaying,
                             "directory": args.recordings,
                             "wantVideo": want_video,
                             "video": bool(session_video),
                             "videoFiles": session_video.paths if session_video else [],
                             "videoDropped": (sum(session_video.dropped)
                                              if session_video else 0),
                             "maskModes": list(recording.MASK_MODES),
                             "mask": mask_mode,
                             # Frames where no face could be located and the
                             # whole frame was obscured instead. A rising
                             # count means detection is dropping out, which
                             # is worth seeing while you can still fix it.
                             "maskBlanked": (sum(session_video.blanked)
                                             if session_video else 0)},
                    replay={"open": replaying,
                            "note": replay_note,
                            "video": bool(replay_video),
                            "derive": derive_mode,
                            "deriveLabel": DERIVE_LABELS[derive_mode],
                            "modes": list(DERIVE_MODES),
                            "canRetriangulate": bool(replay_calib or calibration),
                            "speeds": list(recording.Player.SPEEDS),
                            "files": listing,
                            **(player.state() if player else {})},
                    # Per-landmark diagnosis only when something is actually
                    # producing landmarks. With no cameras attached it would
                    # report every joint as "no pose detected", which reads
                    # as a rig that is failing rather than one that is absent.
                    joints=joint_diagnosis(tri_info, side, args.min_visibility,
                                           args.max_reproj)
                           if (replaying or (calibration and cams)) else ()))
                if not all(do_action(a) for a in web.take_commands()):
                    break
                dt = time.perf_counter() - loop_start
                fps = 1.0 / dt if dt > 0 else 0.0
                fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps
                continue

            top_row = np.hstack(frames + [panel])
            canvas = top_row
            if show_views:
                canvas = np.vstack([top_row, make_views(top_row.shape[1])])

            if args.dump_canvas:
                for i, f in enumerate(frames):
                    print(f"  frame {i}: {f.shape[1]}x{f.shape[0]}")
                print(f"  panel: {panel.shape[1]}x{panel.shape[0]}")
                print(f"  composed canvas: {canvas.shape[1]}x{canvas.shape[0]}")
                cv2.imwrite(args.dump_canvas, canvas)
                print(f"wrote {args.dump_canvas}")
                break

            if scale is None:
                scale = fit_scale(canvas.shape[1], canvas.shape[0])
                if scale < 1.0:
                    print(f"Canvas {canvas.shape[1]}x{canvas.shape[0]} exceeds the screen; "
                          f"displaying at {scale:.2f}x. Override with --scale.")
            if scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if not window_sized:
                cv2.resizeWindow(window, canvas.shape[1], canvas.shape[0])
                window_sized = True
            cv2.imshow(window, canvas)

            dt = time.perf_counter() - loop_start
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps

            action = KEY_ACTIONS.get(cv2.waitKey(1) & 0xFF)
            if action and not do_action(action):
                break
    except KeyboardInterrupt:
        # Ctrl+C reaches every process sharing this console, so this is the
        # normal way run_teleop.py stops the tracker. Taking the same exit
        # path as Q means the cameras and the robot link close properly
        # instead of the process being killed for not finishing in time.
        print("\nStopping the tracker...")
    finally:
        if session.active:
            # Whatever stopped the loop -- Q, Ctrl+C, a camera dying -- the
            # frames already captured are worth more than the tidy exit, so
            # the file is closed properly before anything else is torn down.
            print(session_action("rec-stop"))
        if replay_video:
            replay_video.close()
        if pool:
            pool.shutdown()
        if web:
            web.stop()
        cv2.destroyAllWindows()
        robot_link.disconnect()
        for cam in cams:
            cam.close()


if __name__ == "__main__":
    main()
