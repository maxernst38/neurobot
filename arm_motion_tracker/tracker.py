"""Webcam pose/hand tracking -> predicted shoulder/elbow/wrist/grip motion.

OpenCV window replacement for the old browser app (index.html/main.js/motion.js/
robotLink.js). Same MediaPipe models, same joint-angle geometry, same six
tracked metrics, and the same websocket wire format to main.py's
`--input webcam` bridge, so main.py needs no changes.

Layout: a top row of one annotated feed per camera plus the metrics panel,
over a row of front/side/top orthographic projections. Each camera runs its
own independent detection pipeline, and every estimate is drawn in each 3D
view in its own colour.

With a stereo calibration present (see calibrate_cameras.py) the two views
are additionally triangulated into one fused skeleton, and the metrics
switch to it. That is the point of calibrating: a monocular estimate has to
guess depth, and MediaPipe's guess is poor enough to invert joint angles,
whereas triangulated depth is measured from known geometry.

Controls: SPACE = toggle tracked side, T = toggle triangulation,
V = toggle the 3D views, C = toggle robot connection, Q/Esc = quit.
"""

import argparse
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

from camera_io import Camera, fit_scale, is_torn, open_cameras

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


# --- geometry + classification (port of motion.js) --------------------------

POSE = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
}

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
    {"key": "shoulder_rotation", "label": "Shoulder Rotation", "unit": "deg", "range": (-180, 180), "highLabel": "Rotated Out", "midLabel": "Neutral", "lowLabel": "Rotated In"},
    {"key": "shoulder_flexion", "label": "Shoulder Ext/Flex", "unit": "deg", "range": (0, 180), "highLabel": "Flexed (raised)", "midLabel": "Neutral", "lowLabel": "Extended (lowered)"},
    {"key": "elbow_flexion", "label": "Elbow Ext/Flex", "unit": "deg", "range": (0, 180), "highLabel": "Extended", "midLabel": "Neutral", "lowLabel": "Flexed"},
    {"key": "wrist_flexion", "label": "Wrist Ext/Flex", "unit": "deg", "range": (-90, 90), "highLabel": "Extended", "midLabel": "Neutral", "lowLabel": "Flexed"},
    {"key": "wrist_rotation", "label": "Wrist Rotation", "unit": "deg", "range": (-180, 180), "highLabel": "Supinated", "midLabel": "Neutral", "lowLabel": "Pronated"},
    {"key": "grip", "label": "Hand Grip", "unit": "", "range": (0, 1), "highLabel": "Gripping", "midLabel": "Half", "lowLabel": "Open"},
]

POSE_CONNECTIONS = [
    (POSE["LEFT_SHOULDER"], POSE["RIGHT_SHOULDER"]),
    (POSE["LEFT_SHOULDER"], POSE["LEFT_ELBOW"]), (POSE["LEFT_ELBOW"], POSE["LEFT_WRIST"]),
    (POSE["RIGHT_SHOULDER"], POSE["RIGHT_ELBOW"]), (POSE["RIGHT_ELBOW"], POSE["RIGHT_WRIST"]),
    (POSE["LEFT_SHOULDER"], POSE["LEFT_HIP"]), (POSE["RIGHT_SHOULDER"], POSE["RIGHT_HIP"]),
    (POSE["LEFT_HIP"], POSE["RIGHT_HIP"]),
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


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


def compute_joint_metrics(pose_world, hand_world, side, pose_norm=None, hand_norm=None, aspect=1.0):
    """Six joint metrics for one side of the body.

    Angle math runs on MediaPipe's *world* landmarks (metres, isotropic axes,
    origin at the hip midpoint) rather than the normalized image landmarks
    used for drawing -- see _angle_at for why that distinction matters.
    pose_norm/hand_norm/aspect are only needed for wrist_flexion, which is a
    signed in-image-plane angle spanning two different landmark models.

    Any of the landmark lists may be None when nothing was detected.
    """
    shoulder = pose_world[POSE[f"{side}_SHOULDER"]] if pose_world else None
    elbow = pose_world[POSE[f"{side}_ELBOW"]] if pose_world else None
    wrist = pose_world[POSE[f"{side}_WRIST"]] if pose_world else None
    hip = pose_world[POSE[f"{side}_HIP"]] if pose_world else None

    out = {
        "shoulder_rotation": None, "shoulder_flexion": None, "elbow_flexion": None,
        "wrist_flexion": None, "wrist_rotation": None, "grip": None,
    }

    # Gated per metric rather than all-or-nothing: the hip is only needed to
    # define the torso axis for shoulder flexion, and with two cameras it is
    # often out of frame in one of them while the arm is fully visible.
    if shoulder and elbow and wrist:
        out["elbow_flexion"] = _angle_at(shoulder, elbow, wrist)
        # Axial-rotation proxy: swing angle of the forearm through the depth
        # (z) axis while the upper arm is roughly fixed. From a single camera
        # this leans on guessed depth and is only an approximation; once the
        # points are triangulated the depth is measured, so it becomes a real
        # geometric quantity.
        out["shoulder_rotation"] = math.degrees(math.atan2(wrist.z - elbow.z, wrist.x - elbow.x))

    if shoulder and elbow and hip:
        out["shoulder_flexion"] = _angle_at(hip, shoulder, elbow)

    # Wrist flexion needs the forearm (pose model) and the hand (hand model)
    # in one frame of reference. The two models' world landmarks use
    # different origins, so this one stays in normalized image space, where
    # both are expressed against the same frame -- x rescaled by the aspect
    # ratio so it shares y's units, and z left out entirely (this is a signed
    # in-plane bend, and normalized z is too noisy to help).
    elbow_norm = pose_norm[POSE[f"{side}_ELBOW"]] if pose_norm else None
    if elbow_norm and hand_norm:
        hand_wrist_n = hand_norm[HAND["WRIST"]]
        middle_mcp_n = hand_norm[HAND["MIDDLE_MCP"]]
        # Signed deviation from "straight" (0 deg), not the unsigned
        # _angle_at(): bending the wrist palm-down and bending it back both
        # shrink the unsigned angle the same way, so that form can't tell
        # flexion from extension apart. The cross-product sign gives the
        # bend direction.
        fx = (hand_wrist_n.x - elbow_norm.x) * aspect
        fy = hand_wrist_n.y - elbow_norm.y
        hx = (middle_mcp_n.x - hand_wrist_n.x) * aspect
        hy = middle_mcp_n.y - hand_wrist_n.y
        out["wrist_flexion"] = math.degrees(math.atan2(fx * hy - fy * hx, fx * hx + fy * hy))

    if hand_world:
        hand_wrist = hand_world[HAND["WRIST"]]
        index_mcp = hand_world[HAND["INDEX_MCP"]]
        pinky_mcp = hand_world[HAND["PINKY_MCP"]]
        if hand_wrist and index_mcp and pinky_mcp:
            # Palm normal, i.e. the direction the palm faces. Index and pinky
            # sit on opposite sides of a left vs. right hand, so this cross
            # product follows the palm on a right hand but the back of the
            # hand on a left one -- mirror it so both mean "palm".
            u, v = _vec(hand_wrist, index_mcp), _vec(hand_wrist, pinky_mcp)
            normal = np.cross(u, v)
            if side == "LEFT":
                normal = -normal
            # 0 deg = palm facing the camera. MediaPipe places the camera
            # along -z, so negating z puts the zero there; +-180 is the palm
            # turned fully away, +90 palm toward the right of the image.
            out["wrist_rotation"] = math.degrees(math.atan2(normal[0], -normal[2]))

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


# --- robot link (port of robotLink.js) ---------------------------------------

class RobotLink:
    """Thin websocket client forwarding per-joint state to main.py's
    `--input webcam` control bridge, at the same ~15Hz as the browser client."""

    def __init__(self):
        self.ws = None
        self.status = "disconnected"
        self.last_send_ms = 0
        self.send_interval_ms = 66

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

    def send(self, states, now_ms):
        if self.ws is None:
            return
        if now_ms - self.last_send_ms < self.send_interval_ms:
            return
        self.last_send_ms = now_ms
        try:
            self.ws.send(json.dumps(states))
        except Exception:
            self.disconnect()
            self.status = "error"


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
    if abs(fw / fh - tw / th) < 0.01:
        s = tw / fw
        return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], float) @ K

    if tw / th < fw / fh:
        # Target is narrower: vertical FOV survives, width is cropped.
        s = th / fh
        crop_x, crop_y = (fw - tw / s) / 2, 0.0
    else:
        # Target is wider: horizontal FOV survives, height is cropped.
        s = tw / fw
        crop_x, crop_y = 0.0, (fh - th / s) / 2

    K = K.copy()
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y
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


def triangulate_pose(calibration, poses_norm, sizes, min_visibility=0.3, max_reproj_px=40.0):
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
    valid &= seen
    valid &= pts_mm[:, 2] > 0  # must be in front of camera 1

    # Reproject every point back into both views. A joint that lands far from
    # where either camera actually saw it was not really the same joint --
    # usually because the two cameras grabbed at different instants and the
    # arm moved between them, or because one view mislocalised it. Those
    # points are dropped rather than averaged into the result.
    reproj = None
    if valid.any():
        per_point = np.full(len(pts_mm), np.inf)
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
        valid &= keep

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


def ensure_model(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {os.path.basename(dest_path)}...")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


# --- drawing --------------------------------------------------------------

def to_px(landmark, w, h):
    """True (unmirrored) pixel position, matching the raw camera frame that
    detection ran on."""
    return int(landmark.x * w), int(landmark.y * h)


def to_px_mirrored(landmark, w, h):
    """Pixel position on the mirrored (selfie-view) display frame, i.e. the
    same landmark after the frame has been flipped for display. Used only
    for text, which must be drawn post-flip so it isn't itself mirrored."""
    return int(w - landmark.x * w), int(landmark.y * h)


def draw_label(frame, anchor_px, lines):
    lines = [l for l in lines if l]
    if not lines or anchor_px is None:
        return
    x, y = anchor_px
    line_h, pad_x, pad_y = 18, 6, 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    text_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    box_w, box_h = text_w + pad_x * 2, len(lines) * line_h + pad_y
    box_x, box_y = x + 10, y - box_h // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    for i, line in enumerate(lines):
        ty = box_y + pad_y + line_h * (i + 1) - 5
        cv2.putText(frame, line, (box_x + pad_x, ty), font, scale, (102, 224, 255), thick, cv2.LINE_AA)


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


def draw_angle_labels(frame, pose, metrics, side, active_hand):
    """Draws the joint-angle text labels onto the already-mirrored display
    frame, at mirrored anchor positions, so the text itself reads normally
    instead of coming out backwards."""
    h, w = frame.shape[:2]

    if pose and metrics and side:
        shoulder = pose[POSE[f"{side}_SHOULDER"]]
        elbow = pose[POSE[f"{side}_ELBOW"]]
        wrist = pose[POSE[f"{side}_WRIST"]]

        draw_label(frame, to_px_mirrored(shoulder, w, h), [
            f"Shldr flex {metrics['shoulder_flexion']:.0f}deg" if metrics["shoulder_flexion"] is not None else None,
            f"Shldr rot {metrics['shoulder_rotation']:.0f}deg" if metrics["shoulder_rotation"] is not None else None,
        ])
        draw_label(frame, to_px_mirrored(elbow, w, h), [
            f"Elbow {metrics['elbow_flexion']:.0f}deg" if metrics["elbow_flexion"] is not None else None,
        ])
        draw_label(frame, to_px_mirrored(wrist, w, h), [
            f"Wrist flex {metrics['wrist_flexion']:.0f}deg" if metrics["wrist_flexion"] is not None else None,
            f"Wrist rot {metrics['wrist_rotation']:.0f}deg" if metrics["wrist_rotation"] is not None else None,
        ])

    if active_hand and metrics.get("grip") is not None:
        draw_label(frame, to_px_mirrored(active_hand[HAND["WRIST"]], w, h), [f"Grip {metrics['grip'] * 100:.0f}%"])


PANEL_W = 340
PANEL_BG = (33, 27, 23)
PANEL_BORDER = (53, 44, 38)
TEXT = (239, 233, 230)
MUTED = (161, 146, 138)
ACCENT = (255, 140, 79)
STATE_HIGH = (129, 196, 51)
STATE_LOW = (74, 158, 255)
TRI_COLOR = (120, 255, 160)   # the fused, triangulated skeleton

VIEW_BG = (26, 21, 18)
# Per-camera skeleton colours, reused for that camera's feed title and for
# its trace in every 3D view, so a line is always attributable to a camera.
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


def draw_pose_view(width, height, skeletons, mode, side):
    """Orthographic projection of one or more world-landmark skeletons.

    "front" looks down camera 1's axis, "side" looks along the body's
    left/right axis, "top" is a bird's-eye view. Every mode is expressed in
    camera 1's frame.

    skeletons is a list of (label, colour, pose_world); each is drawn in its
    own colour so two independent estimates stay distinguishable.

    World landmarks are metres about the hip midpoint, so this uses a fixed
    metres-per-pixel scale anchored on that origin -- a skeleton keeps a
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

    tracked = {
        (POSE[f"{side}_SHOULDER"], POSE[f"{side}_ELBOW"]),
        (POSE[f"{side}_ELBOW"], POSE[f"{side}_WRIST"]),
    }

    legend_y = 18
    drew_any = False
    for label, color, pose_world in skeletons:
        cv2.putText(view, label, (width - 62, legend_y), font, 0.38,
                    color if pose_world else PANEL_BORDER, 1, cv2.LINE_AA)
        legend_y += 15
        if not pose_world:
            continue
        drew_any = True

        for a, b in POSE_CONNECTIONS:
            if pose_world[a] is None or pose_world[b] is None:
                continue
            highlight = (a, b) in tracked or (b, a) in tracked
            cv2.line(view, project(pose_world[a]), project(pose_world[b]),
                     color, 2 if highlight else 1, cv2.LINE_AA)
        for idx in (POSE["LEFT_SHOULDER"], POSE["RIGHT_SHOULDER"], POSE["LEFT_ELBOW"],
                    POSE["RIGHT_ELBOW"], POSE["LEFT_WRIST"], POSE["RIGHT_WRIST"]):
            if pose_world[idx] is not None:
                cv2.circle(view, project(pose_world[idx]), 3, TEXT, -1, cv2.LINE_AA)

    if not drew_any:
        cv2.putText(view, "no pose", (ox - 26, oy - 6), font, 0.4, MUTED, 1, cv2.LINE_AA)

    return view


def build_pose_views(width, height, skeletons, side):
    """The front/side/top projections laid out as one row."""
    w = width // 3
    views = [draw_pose_view(w, height, skeletons, mode, side) for mode in ("front", "side", "top")]
    # Last pane absorbs any rounding remainder so the row matches `width`.
    if width - 3 * w:
        views[-1] = draw_pose_view(width - 2 * w, height, skeletons, "top", side)
    row = np.hstack(views)
    for i in (1, 2):
        cv2.line(row, (i * w, 0), (i * w, height), PANEL_BORDER, 1)
    return row


def build_panel(height, results, active, robot_status, title="METRICS", fps=None,
                calib_status="none"):
    panel = np.full((height, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, title, (12, 16), font, 0.4, MUTED, 1, cv2.LINE_AA)
    if fps is not None:
        fps_text = f"{fps:4.1f} fps"
        (fw, _), _ = cv2.getTextSize(fps_text, font, 0.4, 1)
        cv2.putText(panel, fps_text, (PANEL_W - 12 - fw, 16), font, 0.4, MUTED, 1, cv2.LINE_AA)
    y = 46

    banner = "  ·  ".join(active) if active else "All joints neutral"
    banner_color = STATE_HIGH if active else MUTED
    cv2.putText(panel, banner[:38], (12, y), font, 0.5, banner_color, 1, cv2.LINE_AA)
    y += 30
    cv2.line(panel, (12, y - 12), (PANEL_W - 12, y - 12), PANEL_BORDER, 1)

    for m in METRICS:
        r = results[m["key"]]
        cv2.putText(panel, m["label"], (12, y), font, 0.45, TEXT, 1, cv2.LINE_AA)
        if r["value"] is None:
            val_text, dir_text = "--", "No data"
        else:
            val_text = f"{r['value']:.0f}deg" if m["unit"] == "deg" else f"{r['value']:.2f}"
            dir_text = state_label(m, r["state"])
        (val_w, _), _ = cv2.getTextSize(val_text, font, 0.45, 1)
        cv2.putText(panel, val_text, (PANEL_W - 12 - val_w, y), font, 0.45, MUTED, 1, cv2.LINE_AA)
        y += 18

        lo, hi = m["range"]
        bar_x, bar_w, bar_h = 12, PANEL_W - 24, 6
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), PANEL_BORDER, 1)
        if r["value"] is not None:
            pct = max(0.0, min(1.0, (r["value"] - lo) / (hi - lo)))
            fill_color = STATE_HIGH if r["state"] == "high" else STATE_LOW if r["state"] == "low" else PANEL_BORDER
            cv2.rectangle(panel, (bar_x, y), (bar_x + int(bar_w * pct), y + bar_h), fill_color, -1)
        y += bar_h + 6
        cv2.putText(panel, dir_text, (12, y), font, 0.4, MUTED, 1, cv2.LINE_AA)
        y += 24

    y = height - 108
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

    def __init__(self, camera, name, color, pose_model, hand_model, delegate):
        self.camera = camera
        self.index = camera.index
        self.name = name
        self.color = color

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        ))
        self.hand_landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        ))

        self.classifier = JointStateClassifier()
        self.last_ts_ms = -1
        self.pose_norm = None
        self.pose_world = None
        self.hand_norm = None
        self.hand_world = None
        self.results = None

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
        hand_result = self.hand_landmarker.detect_for_video(mp_image, ts_ms)

        pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        self.pose_norm = pose
        self.pose_world = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None

        hands = match_hand_to_side(hand_result.hand_landmarks, pose)
        hand_idx = hands[side]
        hand_norm = hand_result.hand_landmarks[hand_idx] if hand_idx is not None else None
        hand_world = hand_result.hand_world_landmarks[hand_idx] if hand_idx is not None else None
        self.hand_norm, self.hand_world = hand_norm, hand_world

        frame_h, frame_w = frame.shape[:2]
        raw = compute_joint_metrics(
            self.pose_world, hand_world, side,
            pose_norm=pose, hand_norm=hand_norm, aspect=frame_w / frame_h,
        )
        self.results = self.classifier.update(raw)

        drawn_hands = [
            hand_result.hand_landmarks[i] if i is not None else None
            for i in (hands["LEFT"], hands["RIGHT"])
        ]
        draw_landmarks(frame, pose, drawn_hands)
        frame = cv2.flip(frame, 1)  # natural selfie view
        draw_angle_labels(frame, pose, raw, side, hand_norm)

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
        self.hand_landmarker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV index of the front camera")
    parser.add_argument("--camera2", type=int, default=2,
                        help="OpenCV index of the side camera; -1 to run with one camera")
    parser.add_argument("--side", choices=["LEFT", "RIGHT"], default="RIGHT", help="initially tracked arm")
    parser.add_argument("--metrics-cam", type=int, choices=[1, 2], default=1,
                        help="which camera's estimate drives the metrics panel and the robot")
    parser.add_argument("--robot-host", default="localhost:8765", help="host:port of main.py --input webcam")
    parser.add_argument("--robot", action="store_true", help="connect to the robot bridge on startup")
    parser.add_argument("--delegate", choices=["CPU", "GPU"], default="CPU", help="MediaPipe inference delegate")
    parser.add_argument("--raw-format", action="store_true", help="don't force MJPEG capture (use if your camera doesn't support it)")
    parser.add_argument("--no-views", action="store_true", help="start with the 3D projection row hidden")
    parser.add_argument("--scale", default="auto",
                        help="window scale: 'auto' fits it to your screen, or give a number like 0.75")
    parser.add_argument("--width", type=int, default=640, help="capture width (match your calibration resolution)")
    parser.add_argument("--height", type=int, default=480, help="capture height (match your calibration resolution)")
    parser.add_argument("--calibration", default=CALIBRATION_PATH, help="stereo calibration JSON from calibrate_cameras.py")
    # MediaPipe reports low visibility for joints it is inferring rather than
    # clearly seeing. 0.5 rejects most elbows/wrists when the arm is near the
    # frame edge, so the default is looser and tunable.
    parser.add_argument("--min-visibility", type=float, default=0.3,
                        help="per-view landmark confidence needed to triangulate a joint")
    parser.add_argument("--max-reproj", type=float, default=40.0,
                        help="max reprojection error (px) before a triangulated joint is discarded")
    parser.add_argument("--dump-canvas", metavar="PATH",
                        help="save the composed window to PATH and exit (for diagnosing display problems)")
    args = parser.parse_args()

    pose_model = ensure_model(POSE_MODEL_URL, os.path.join(MODEL_DIR, "pose_landmarker_lite.task"))
    hand_model = ensure_model(HAND_MODEL_URL, os.path.join(MODEL_DIR, "hand_landmarker.task"))
    delegate = BaseOptions.Delegate.GPU if args.delegate == "GPU" else BaseOptions.Delegate.CPU

    specs = [(args.camera, "CAM 1 front")]
    if args.camera2 >= 0:
        specs.append((args.camera2, "CAM 2 side"))
    devices = open_cameras(specs, args.width, args.height, force_mjpg=not args.raw_format)
    cams = [
        CameraPipeline(dev, name, CAM_COLORS[i], pose_model, hand_model, delegate)
        for i, (dev, (_, name)) in enumerate(zip(devices, specs))
    ]

    frame_sizes = [c.camera.size for c in cams]
    calibration = load_calibration(args.calibration, frame_sizes) if len(cams) > 1 else None
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

    metrics_cam = cams[min(args.metrics_cam, len(cams)) - 1]
    robot_link = RobotLink()
    side = args.side
    show_views = not args.no_views
    window = "Arm Motion Tracker"
    # NORMAL rather than AUTOSIZE so the window can be resized and can be
    # smaller than the canvas; AUTOSIZE forces it to the image size, which a
    # window manager then clips off-screen with no way to shrink it.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = None if args.scale == "auto" else float(args.scale)
    window_sized = False

    if args.robot:
        robot_link.connect(f"ws://{args.robot_host}/ws")

    print(f"Running {len(cams)} independent pipeline(s): " + ", ".join(f"{c.name}={c.index}" for c in cams))
    print(f"Metrics panel and robot output follow {metrics_cam.name}.")
    print("Controls: SPACE = toggle tracked side, T = toggle triangulation, "
          "V = toggle 3D views, C = toggle robot connection, Q/Esc = quit.")

    fps_ema = None
    tri_classifier = JointStateClassifier()
    use_triangulation = calibration is not None
    tri_reproj_ema = None
    pool = ThreadPoolExecutor(max_workers=len(cams)) if len(cams) > 1 else None
    try:
        while True:
            loop_start = time.perf_counter()

            # Capture first, from both cameras back to back, so neither
            # driver queue waits on the other's inference. Only then run the
            # models, in parallel -- MediaPipe releases the GIL during native
            # inference, which is worth ~1.6x on two cameras.
            raw = [cam.read() for cam in cams]
            frames = (list(pool.map(lambda cf: cf[0].process(cf[1], side), zip(cams, raw)))
                      if pool else [cams[0].process(raw[0], side)])
            if any(f is None for f in frames):
                print("A camera stopped delivering frames.")
                break

            # Triangulate from the two views. This supersedes the per-camera
            # monocular estimates when it succeeds, because its depth is
            # measured rather than inferred.
            tri_points, tri_info = (None, None)
            if calibration and len(cams) == 2:
                tri_points, tri_info = triangulate_pose(
                    calibration,
                    [c.pose_norm for c in cams],
                    [c.camera.size for c in cams],
                    min_visibility=args.min_visibility,
                    max_reproj_px=args.max_reproj,
                )

            source = metrics_cam.name
            results = metrics_cam.results
            if tri_points and use_triangulation:
                mc = metrics_cam
                tri_raw = compute_joint_metrics(
                    tri_points, mc.hand_world, side,
                    pose_norm=mc.pose_norm, hand_norm=mc.hand_norm,
                    aspect=mc.camera.size[0] / mc.camera.size[1],
                )
                results = tri_classifier.update(tri_raw)
                source = "TRIANGULATED"

            active = [
                f"{m['label'].replace(' Ext/Flex', '')}: {state_label(m, results[m['key']]['state'])}"
                for m in METRICS if results[m["key"]]["state"] in ("high", "low")
            ]
            if tri_info and tri_info.get("reproj") is not None:
                r = tri_info["reproj"]
                tri_reproj_ema = r if tri_reproj_ema is None else 0.9 * tri_reproj_ema + 0.1 * r
            if not calibration:
                tri_status = calib_status
            elif not use_triangulation:
                tri_status = "off (press T)"
            elif tri_points:
                tri_status = f"{tri_info['n_valid']}/33 joints"
                if tri_reproj_ema is not None:
                    tri_status += f", {tri_reproj_ema:.0f}px"
                dropped = tri_info["lost_visibility"] + tri_info["lost_reproj"]
                if dropped:
                    tri_status += f" (-{tri_info['lost_visibility']}vis -{tri_info['lost_reproj']}err)"
            else:
                tri_status = tri_info["reason"] if tri_info else "no data"
            panel = build_panel(frames[0].shape[0], results, active, robot_link.status,
                                title=f"METRICS - {source}", fps=fps_ema,
                                calib_status=tri_status)

            states = {m["key"]: results[m["key"]]["state"] for m in METRICS}
            robot_link.send(states, int(loop_start * 1000))

            top_row = np.hstack(frames + [panel])
            canvas = top_row
            if show_views:
                skeletons = [(" ".join(c.name.split()[:2]), c.color, c.pose_world) for c in cams]
                if tri_points:
                    skeletons.append(("3D FUSED", TRI_COLOR, tri_points))
                views = build_pose_views(top_row.shape[1], 360, skeletons, side)
                canvas = np.vstack([top_row, views])

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

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                side = "LEFT" if side == "RIGHT" else "RIGHT"
                for cam in cams:
                    cam.classifier.reset()
                tri_classifier.reset()
                print(f"Tracking side: {side}")
            elif key == ord("t") and calibration:
                use_triangulation = not use_triangulation
                tri_classifier.reset()
                print(f"triangulated metrics {'on' if use_triangulation else 'off'}")
            elif key == ord("v"):
                show_views = not show_views
                if args.scale == "auto":
                    scale = None  # canvas changed shape; refit on the next frame
                window_sized = False
            elif key == ord("c"):
                if robot_link.is_connected:
                    robot_link.disconnect()
                else:
                    robot_link.connect(f"ws://{args.robot_host}/ws")
    finally:
        if pool:
            pool.shutdown()
        cv2.destroyAllWindows()
        robot_link.disconnect()
        for cam in cams:
            cam.close()


if __name__ == "__main__":
    main()
