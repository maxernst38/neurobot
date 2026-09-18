"""Stereo-calibrate the two tracker cameras from a held calibration board.

Two steps, always in this order:

    python calibrate_cameras.py capture      # collect images from both cameras
    python calibrate_cameras.py calibrate    # solve from those images

Collecting and solving are separate on purpose. Capture then costs almost
nothing per frame, which matters on a bandwidth-limited link where starving
the capture loop is what corrupts frames; the images can be re-processed
with different board or detector settings without standing in front of the
cameras again; and a disappointing calibration stays diagnosable afterwards
because the evidence is still on disk.

Capture records its board settings into session.json beside the images and
calibrate reads them back, so the two steps cannot silently disagree about
the board -- which would otherwise produce a confidently wrong result.

Two patterns are supported, and they behave differently on a widely
separated camera pair:

  --pattern charuco       (default) every corner carries an identity, so a
                          view still counts when the board is half out of
                          frame or steeply oblique.
  --pattern checkerboard  all-or-nothing: every inner corner must be visible
                          before a view contributes anything. Simpler to
                          print, but it discards far more views here, and
                          its corners have no identity -- so if the two
                          cameras ever order them differently the stereo
                          solve fails loudly with a huge RMS.

Defaults describe a 10x7 board. Sizes are what you MEASURED on the printout:

    python calibrate_cameras.py capture --pattern checkerboard --square-mm 22.5

Views where only one camera sees the board are NOT wasted: they still feed
that camera's own intrinsics. Only the extrinsics need simultaneous views.

Capture controls: SPACE = capture, A = auto-capture, U = undo, Q/Esc = done.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from camera_io import DEFAULT_CAMERA_INDICES, Camera, fit_scale, open_cameras, read_all

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, "calibration", "stereo_calibration.json")
DEFAULT_CAPTURE_DIR = os.path.join(_HERE, "calibration", "captures")
SESSION_FILE = "session.json"

# Defaults describe the 10x7 board already in use here. Its squares measure
# 22.5 mm rather than the nominal 25 because that print came out at 90%
# scale -- pass the sizes you measure on your own printout.
BOARD_DEFAULTS = {"cols": 10, "rows": 7, "square_mm": 22.5,
                  "marker_mm": 16.2, "dictionary": "DICT_5X5_100",
                  "pattern": "charuco"}

# A view contributes to a camera's intrinsics only if it pins down enough of
# the board; too few corners makes the per-view pose ambiguous and poisons
# the solve rather than helping it.
MIN_CORNERS_INTRINSIC = 12
MIN_CORNERS_STEREO = 8

TEXT = (239, 233, 230)
MUTED = (161, 146, 138)
OK_COLOR = (129, 196, 51)
WARN_COLOR = (74, 158, 255)
CAM_COLORS = [(255, 140, 79), (120, 130, 255)]


class BoardDetector:
    orientation_ambiguous = False   # charuco corners carry their own ids

    def __init__(self, cols, rows, square_mm, marker_mm, dict_name, refine=True):
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        self.board = cv2.aruco.CharucoBoard((cols, rows), square_mm, marker_mm, self.dictionary)

        params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        if refine:
            # OpenCV defaults to CORNER_REFINE_NONE, which locates marker
            # corners only to the nearest contour pixel. Every charuco corner
            # is interpolated from those, so without refinement the whole
            # board's corners jitter by a pixel or more between frames -- the
            # calibration then fits that jitter as if it were geometry.
            # Sub-pixel refinement is the single biggest stability win here.
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 50
            params.cornerRefinementMinAccuracy = 0.01
            charuco_params.tryRefineMarkers = True

        self.detector = cv2.aruco.CharucoDetector(self.board, charuco_params, params)
        # Every inner corner's 3D position on the board plane, indexed by the
        # same charuco id the detector reports, so ids index straight into it.
        self.all_corners = self.board.getChessboardCorners()

    def detect(self, frame):
        """Return (corners Nx2, ids N) of detected inner corners, or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) == 0:
            return None
        return np.asarray(corners, np.float32).reshape(-1, 2), np.asarray(ids, np.int32).reshape(-1)


class CheckerboardDetector:
    """Plain checkerboard, exposing the same interface as BoardDetector.

    Corners are reported with sequential ids so everything downstream --
    the shared-corner intersection, the intrinsics gather, stereoCalibrate --
    works unchanged. Unlike ChArUco those ids carry no identity of their own:
    a checkerboard is all-or-nothing, so either the whole grid is found or
    the view contributes nothing.

    That difference matters for a widely-separated pair. ChArUco tolerates a
    board that is half out of frame or steeply oblique; a checkerboard needs
    every inner corner visible in a view before that view counts at all.
    """

    # Anonymous corners: the same grid detected half a turn round is an
    # equally valid answer, so stereo pairs need resolve_corner_flips.
    orientation_ambiguous = True

    def __init__(self, cols, rows, square_mm, refine=True):
        self.pattern = (cols - 1, rows - 1)          # inner corners, OpenCV's convention
        self.refine = refine
        w, h = self.pattern
        grid = np.zeros((w * h, 3), np.float32)
        grid[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        self.all_corners = grid * float(square_mm)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # findChessboardCornersSB is the newer sector-based detector: it is
        # markedly better than the classic one under blur and uneven
        # lighting, and refines to sub-pixel itself.
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if self.refine:
            flags |= cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, self.pattern, flags)
        if not ok or corners is None:
            ok, corners = cv2.findChessboardCorners(
                gray, self.pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok or corners is None:
                return None
            if self.refine:
                cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
        corners = np.asarray(corners, np.float32).reshape(-1, 2)
        return corners, np.arange(len(corners), dtype=np.int32)


def make_detector(args):
    """Build the detector for the requested pattern."""
    if args.pattern == "checkerboard":
        return CheckerboardDetector(args.cols, args.rows, args.square_mm,
                                    refine=not args.no_refine)
    return BoardDetector(args.cols, args.rows, args.square_mm, args.marker_mm,
                         args.dictionary, refine=not args.no_refine)


def frame_sharpness(frame):
    """Variance of the Laplacian: high for crisp edges, low for blur."""
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


# An ArUco marker from a 5x5 dictionary carries a 7x7 bit grid once its
# mandatory black border is counted, so its apparent size sets a hard floor on
# whether it can be decoded at all: below roughly 14 px across there are not
# enough pixels to hold the bits, and no amount of lighting or focus recovers
# it. Detection is reliable from about 20 px. This is usually what "the camera
# can't see the board" actually means -- the answer is a larger printed board,
# a higher capture resolution, or holding it closer, not a better sensor.
MARKER_PX_GOOD = 20.0
MARKER_PX_FLOOR = 14.0


def board_scale_px(detector, detection):
    """Pixels per millimetre of board, measured from one detection.

    Each detected corner is paired with its nearest neighbour *on the board*,
    which is a known distance away; the same pair measured in the image gives
    the scale. Taking the median over all corners makes it robust to a stray
    misdetection, and it needs no calibration -- which is the point, since it
    is used to decide whether a calibration is even possible.
    """
    corners, ids = detection
    if len(ids) < 2:
        return None
    obj = detector.all_corners[ids][:, :2].astype(np.float64)
    d_obj = np.linalg.norm(obj[:, None] - obj[None, :], axis=2)
    d_img = np.linalg.norm(corners[:, None].astype(np.float64) - corners[None, :], axis=2)
    np.fill_diagonal(d_obj, np.inf)
    rows = np.arange(len(ids))
    nn = np.argmin(d_obj, axis=1)
    scales = d_img[rows, nn] / d_obj[rows, nn]
    return float(np.median(scales))


def capture_burst(cams, n, mode="sharpest"):
    """Take one frame per camera, optionally choosing from a short burst.

    n = 1 simply grabs a frame, which is the right default for a handheld
    board.

    Averaging a burst suppresses random noise, but only if nothing moved.
    A board held by hand drifts between frames, and blending that drift is
    motion blur -- which destroys corner accuracy far more thoroughly than
    the noise it removes, because a blurred edge has no well-defined
    position at all. "sharpest" instead keeps the single crispest frame of
    the burst, so a burst can still reject the worst hand-shake without ever
    mixing two positions together.
    """
    n = max(1, n)
    if n == 1:
        return read_all(cams)

    best = [None] * len(cams)
    best_score = [-1.0] * len(cams)
    for _ in range(n):
        for i, f in enumerate(read_all(cams)):
            if f is None:
                continue
            score = frame_sharpness(f)
            if score > best_score[i]:
                best[i], best_score[i] = f, score

    if mode != "average":
        return best

    stacks = [[] for _ in cams]
    for _ in range(n):
        for i, f in enumerate(read_all(cams)):
            if f is not None:
                stacks[i].append(f.astype(np.float32))
    return [p if not s else np.mean(s, axis=0).astype(np.uint8)
            for p, s in zip(best, stacks)]


def save_pair(save_dir, index, frames):
    """Write one capture's frames as PNG.

    PNG rather than JPEG deliberately: JPEG's ringing around the board's
    high-contrast edges lands exactly where corners are measured, so a
    lossy round-trip would bake detection error into the archive.
    """
    os.makedirs(save_dir, exist_ok=True)
    for cam_i, frame in enumerate(frames):
        if frame is not None:
            cv2.imwrite(os.path.join(save_dir, f"cam{cam_i + 1}_{index:04d}.png"), frame)


def write_session(args, count):
    """Record the board and camera settings the images were shot with."""
    payload = {k: getattr(args, k) for k in BOARD_DEFAULTS}
    payload.update(camera=args.camera, camera2=args.camera2,
                   captures=count, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(args.dir, SESSION_FILE), "w") as fh:
        json.dump(payload, fh, indent=2)


def read_session(save_dir):
    path = os.path.join(save_dir, SESSION_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_pairs(save_dir):
    """Read back capture sets written by save_pair, newest run included.

    A missing file for one camera is fine -- that view simply contributes to
    the other camera's intrinsics, exactly as in the live path.
    """
    import glob
    indices = sorted({os.path.basename(p).split("_")[1].split(".")[0]
                      for p in glob.glob(os.path.join(save_dir, "cam[12]_*.png"))})
    pairs = []
    for idx in indices:
        frames = []
        for cam_i in (1, 2):
            path = os.path.join(save_dir, f"cam{cam_i}_{idx}.png")
            frames.append(cv2.imread(path) if os.path.exists(path) else None)
        if any(f is not None for f in frames):
            pairs.append((idx, frames))
    return pairs


def draw_detection(frame, detection, color):
    if detection is None:
        return 0
    corners, _ = detection
    for (x, y) in corners:
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    hull = cv2.convexHull(corners.reshape(-1, 1, 2).astype(np.int32))
    cv2.polylines(frame, [hull], True, color, 1, cv2.LINE_AA)
    return len(corners)


def common_points(detector, det1, det2):
    """Object/image point triple for one stereo view.

    The two cameras rarely resolve the same set of corners, so the view is
    reduced to the ids both actually saw, sorted so the three arrays stay
    row-aligned -- stereoCalibrate requires exact correspondence.
    """
    (c1, i1), (c2, i2) = det1, det2
    shared = np.intersect1d(i1, i2)
    if len(shared) < MIN_CORNERS_STEREO:
        return None
    p1 = c1[np.argsort(i1)][np.isin(np.sort(i1), shared)]
    p2 = c2[np.argsort(i2)][np.isin(np.sort(i2), shared)]
    obj = detector.all_corners[shared]
    return (obj.reshape(-1, 1, 3).astype(np.float32),
            p1.reshape(-1, 1, 2).astype(np.float32),
            p2.reshape(-1, 1, 2).astype(np.float32))


def resolve_corner_flips(obj_s, p1_s, p2_s, cam1, cam2):
    """Undo 180-degree ordering disagreements between the two views.

    A checkerboard's corners are anonymous, so detecting the same physical
    grid rotated half a turn is an equally valid answer, and which one comes
    back depends on the viewpoint. When the two cameras disagree, corner k in
    one view is matched against corner N-1-k in the other and stereoCalibrate
    fits nonsense -- visible as a stereo RMS an order of magnitude worse than
    either camera's own.

    The board pose is recovered per camera per view, which makes the relative
    camera transform observable for each view independently. That transform
    is a fixed property of the rig, so views that disagree with the consensus
    are the flipped ones; reversing their second-view points restores the
    correspondence. ChArUco never needs this -- its corners are identified.
    """
    def relative_rvec(obj, pts1, pts2):
        ok1, r1, _ = cv2.solvePnP(obj, pts1, cam1[0], cam1[1])
        ok2, r2, _ = cv2.solvePnP(obj, pts2, cam2[0], cam2[1])
        if not (ok1 and ok2):
            return None
        R1, R2 = cv2.Rodrigues(r1)[0], cv2.Rodrigues(r2)[0]
        return cv2.Rodrigues(R2 @ R1.T)[0].ravel()

    options = [(relative_rvec(o, a, b), relative_rvec(o, a, b[::-1]))
               for o, a, b in zip(obj_s, p1_s, p2_s)]
    usable = [i for i, o in enumerate(options) if o[0] is not None and o[1] is not None]
    if not usable:
        return p2_s, 0

    flips = [False] * len(options)
    for _ in range(5):  # converges immediately once the consensus is right
        ref = np.median([options[i][int(flips[i])] for i in usable], axis=0)
        for i in usable:
            flips[i] = bool(np.linalg.norm(options[i][1] - ref)
                            < np.linalg.norm(options[i][0] - ref))

    fixed = [p[::-1] if f else p for p, f in zip(p2_s, flips)]
    return fixed, int(sum(flips))


def check_intrinsics(K, dist, image_size):
    """Sanity-check a solved camera model. Returns (stats, warnings).

    Reprojection RMS only says how well the model fits the views it was
    given; it says nothing about whether the model is physically sensible or
    whether it extrapolates. A high-order radial polynomial fitted to too
    few views, or to views that all share a similar tilt, can score a
    flattering RMS while being wildly wrong a few pixels outside the corners
    the board happened to occupy -- and it is used everywhere, on every
    landmark, every frame.

    The three checks below each catch a distinct failure that has actually
    happened on these cameras:

      fx/fy      a webcam has square pixels, so this is ~1.000. A ratio well
                 off 1 means fy absorbed error instead of the geometry being
                 solved, which skews every reconstructed angle.
      radial     the distortion factor at the frame corner. Should be near
                 1.0 (a few percent of correction). A large or negative
                 value means the polynomial has diverged, and undistortPoints
                 will move points by hundreds of pixels near the edges.
      centre     the principal point sits close to the image centre on a
                 normal lens; far off usually means it traded against fy for
                 want of views with varied tilt.
    """
    w, h = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, _, _, k3 = dist.ravel()[:5]
    x, y = (w - cx) / fx, (h - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

    stats = {"fx_over_fy": float(fx / fy), "corner_radial": float(radial),
             "principal_offset_pct": [float(100 * (cx - w / 2) / w), float(100 * (cy - h / 2) / h)]}
    warnings = []
    if not 0.98 <= fx / fy <= 1.02:
        warnings.append(f"fx/fy = {fx / fy:.3f}, expected ~1.000 for square pixels")
    if not 0.8 <= radial <= 1.2:
        warnings.append(f"radial factor at the frame corner = {radial:+.2f}, expected ~1.0 "
                        f"-- the distortion model has diverged")
    if abs(cx - w / 2) > 0.1 * w or abs(cy - h / 2) > 0.1 * h:
        warnings.append(f"principal point ({cx:.0f}, {cy:.0f}) is far from the image "
                        f"centre ({w / 2:.0f}, {h / 2:.0f})")
    return stats, warnings


def calibrate(detector, captures, image_sizes, fix_aspect=True, fix_k3=True):
    """Per-camera intrinsics, then extrinsics with those intrinsics held fixed.

    image_sizes is one (w, h) per camera -- they need not match, and a
    camera's intrinsics are only meaningful against the size they were
    measured at.

    fix_aspect forces fx == fy, and fix_k3 pins the third radial term at 0.
    Both constrain the fit to what a webcam can physically be, and both are
    on by default because leaving them free is how the solver ends up
    describing noise: with 5 free distortion terms and a limited spread of
    views it can drive k2/k3 to large opposing values that cancel over the
    board and diverge outside it. Removing k3 leaves a model that cannot do
    that, and a fixed aspect ratio removes fy's freedom to absorb error that
    belongs to the pose. Turn either off only to compare.
    """
    report = {}
    intrinsics = []
    for cam in (0, 1):
        image_size = image_sizes[cam]
        obj_pts, img_pts = [], []
        for cap in captures:
            det = cap[cam]
            if det is None or len(det[1]) < MIN_CORNERS_INTRINSIC:
                continue
            corners, ids = det
            obj_pts.append(detector.all_corners[ids].reshape(-1, 1, 3).astype(np.float32))
            img_pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
        if len(obj_pts) < 6:
            raise SystemExit(
                f"Camera {cam + 1}: only {len(obj_pts)} usable views "
                f"(need >= 6, ideally 20+). Capture more."
            )

        flags = 0
        K_init = None
        if fix_aspect:
            # CALIB_FIX_ASPECT_RATIO holds fx/fy at whatever ratio the *input*
            # camera matrix has, so it only means "fx == fy" if the initial
            # guess is seeded that way -- hence initCameraMatrix2D with
            # aspectRatio 1.0, and USE_INTRINSIC_GUESS to make it the start
            # point. Passing None here would leave the constrained ratio
            # undefined rather than equal.
            K_init = cv2.initCameraMatrix2D(obj_pts, img_pts, image_size, 1.0)
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
        if fix_k3:
            flags |= cv2.CALIB_FIX_K3

        rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, image_size,
                                                 K_init, None, flags=flags)
        intrinsics.append((K, dist))
        stats, warnings = check_intrinsics(K, dist, image_size)
        report[f"cam{cam + 1}_views"] = len(obj_pts)
        report[f"cam{cam + 1}_rms_px"] = float(rms)
        report[f"cam{cam + 1}_checks"] = stats
        if warnings:
            report[f"cam{cam + 1}_warnings"] = warnings
        print(f"  camera {cam + 1}: {len(obj_pts):3d} views   reprojection RMS = {rms:.3f} px"
              f"   fx/fy = {stats['fx_over_fy']:.3f}"
              f"   corner radial = {stats['corner_radial']:+.2f}")
        for w_ in warnings:
            print(f"    WARNING: {w_}")

    obj_s, p1_s, p2_s = [], [], []
    for cap in captures:
        if cap[0] is None or cap[1] is None:
            continue
        pts = common_points(detector, cap[0], cap[1])
        if pts is None:
            continue
        obj_s.append(pts[0]), p1_s.append(pts[1]), p2_s.append(pts[2])
    if len(obj_s) < 6:
        raise SystemExit(
            f"Only {len(obj_s)} views where both cameras saw enough of the board "
            f"(need >= 6, ideally 15+). Hold the board where both can see it."
        )

    (K1, d1), (K2, d2) = intrinsics
    if getattr(detector, "orientation_ambiguous", False):
        p2_s, n_flipped = resolve_corner_flips(obj_s, p1_s, p2_s, (K1, d1), (K2, d2))
        report["stereo_flips_fixed"] = n_flipped
        if n_flipped:
            print(f"  fixed {n_flipped}/{len(obj_s)} views where the two cameras "
                  f"ordered the checkerboard corners oppositely")
    # With CALIB_FIX_INTRINSIC the imageSize argument is only used for
    # initialisation, so camera 1's is fine even if the two differ.
    rms, K1, d1, K2, d2, R, T, _, _ = cv2.stereoCalibrate(
        obj_s, p1_s, p2_s, K1, d1, K2, d2, image_sizes[0],
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    baseline = float(np.linalg.norm(T))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    report.update(stereo_views=len(obj_s), stereo_rms_px=float(rms),
                  baseline_mm=baseline, angle_deg=angle)
    print(f"  stereo:    {len(obj_s):3d} views   reprojection RMS = {rms:.3f} px")
    print(f"  recovered baseline {baseline:.1f} mm, cameras {angle:.1f} deg apart")
    return (K1, d1), (K2, d2), R, T, report


def save(path, detector, args, image_sizes, cam1, cam2, R, T, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "units": "millimetres (from the board square size)",
        "board": {
            "pattern": args.pattern,
            "squares": [args.cols, args.rows],
            "square_mm": args.square_mm,
            "marker_mm": args.marker_mm if args.pattern == "charuco" else None,
            "dictionary": args.dictionary if args.pattern == "charuco" else None,
        },
        # What the intrinsics were constrained to, so a calibration stays
        # interpretable later -- "fx != fy" means something quite different
        # in a run that allowed it than in one that should have forbidden it.
        "solver": {
            "fix_aspect_ratio": not getattr(args, "free_aspect", False),
            "fix_k3": not getattr(args, "free_k3", False),
        },
        "cameras": [
            {"index": args.camera, "name": "CAM 1 front", "image_size": list(image_sizes[0]),
             "K": cam1[0].tolist(), "dist": cam1[1].ravel().tolist()},
            {"index": args.camera2, "name": "CAM 2 side", "image_size": list(image_sizes[1]),
             "K": cam2[0].tolist(), "dist": cam2[1].ravel().tolist()},
        ],
        # Maps a point in camera 2's frame into camera 1's: X1 = R @ X2 + T.
        # Camera 1 is therefore the reference ("world") frame.
        "stereo": {"R": R.tolist(), "T": T.ravel().tolist()},
        "report": report,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {path}")


def cmd_capture(args):
    """Collect board images. No calibration, no heavy detection."""
    detector = make_detector(args)
    cams = open_cameras([(args.camera, "CAM 1"), (args.camera2, "CAM 2")],
                        args.width, args.height, force_mjpg=not args.raw_format,
                        fps=args.fps)
    print()
    print(f"Saving to {args.dir}")
    if args.pattern == "checkerboard":
        print(f"Board: {args.cols}x{args.rows} checkerboard "
              f"({args.cols - 1}x{args.rows - 1} inner corners), square {args.square_mm} mm")
    else:
        print(f"Board: {args.cols}x{args.rows} ChArUco, square {args.square_mm} mm, "
              f"marker {args.marker_mm} mm, {args.dictionary}")
    print("If those are not the sizes you MEASURED on the printout, quit and pass the real ones.\n")
    print("Collect 30-40 views: tilt the board 30-45 degrees in BOTH axes, vary distance,")
    print("and work it into all four corners of each frame. Views with too little tilt")
    print("variety are what leave fy and the principal point poorly determined.")
    print("Holding it still during a capture matters more than anything else.")
    if args.pattern == "charuco":
        print(f"\nEach feed reports its apparent marker size. Below {MARKER_PX_FLOOR:.0f} px "
              f"the markers cannot be")
        print(f"decoded at all and the label turns blue; {MARKER_PX_GOOD:.0f} px or more is "
              f"comfortable. If you")
        print("cannot reach that at the distance both cameras need, print a larger board.")
        print("Intrinsics do not need both cameras, so shoot those close up where the")
        print("markers are large; only the stereo views have to be held further back.")
    print("\nControls: SPACE = capture, A = auto-capture, U = undo last, Q/Esc = done.\n")

    saved = 0
    auto = False
    last_auto = 0.0
    last_centroid = None
    window = "Calibration capture"
    # NORMAL, not AUTOSIZE: two 1280x720 feeds side by side is 2560 px wide,
    # which AUTOSIZE would push off the edge of the screen unresizably.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    display_scale = None
    window_sized = False
    pool = ThreadPoolExecutor(max_workers=2)
    dropped = frame_no = 0
    dets = None

    def store(frames):
        nonlocal saved
        saved += 1
        save_pair(args.dir, saved, frames)
        return saved

    try:
        while True:
            # Concurrent, exactly as tracker.py does it, so the two frames
            # are as close together in time as free-running cameras allow.
            frames = read_all(cams)
            if any(f is None for f in frames):
                dropped += 1
                if dropped % 30 == 1:
                    print(f"waiting for frames ({dropped} misses so far)...")
                if dropped > 300:
                    print("Cameras stopped delivering frames; giving up.")
                    break
                continue
            frames = [f.copy() for f in frames]  # about to be drawn on
            sizes = [(f.shape[1], f.shape[0]) for f in frames]

            # Preview detection is only an aiming aid, so it is throttled and
            # its results are never what gets calibrated -- the calibrate step
            # re-detects every saved image properly.
            frame_no += 1
            if frame_no % max(1, args.detect_every) == 0 or dets is None:
                dets = list(pool.map(detector.detect, frames))
            counts = [draw_detection(frames[i], dets[i], CAM_COLORS[i]) for i in (0, 1)]
            both = dets[0] is not None and dets[1] is not None

            shared = 0
            if both:
                pts = common_points(detector, dets[0], dets[1])
                shared = 0 if pts is None else len(pts[0])

            now = time.time()
            if auto and both and shared >= MIN_CORNERS_STEREO and now - last_auto > args.auto_interval:
                # Require the board to have actually moved, so holding it
                # still does not fill the set with near-duplicate views that
                # add no new constraints.
                centroid = dets[0][0].mean(axis=0)
                if last_centroid is None or np.linalg.norm(centroid - last_centroid) > 25:
                    n = store(capture_burst(cams, args.burst, args.burst_mode))
                    print(f"auto-captured #{n}  (preview corners {counts[0]}/{counts[1]}, shared {shared})")
                    last_centroid, last_auto = centroid, now

            for i in (0, 1):
                cv2.rectangle(frames[i], (0, 0), (sizes[i][0], 24), (0, 0, 0), -1)
                label = f"CAM {i + 1}   corners {counts[i]:3d}"
                # Apparent marker size, which is what decides whether the board
                # is decodable at this distance -- see MARKER_PX_FLOOR.
                feature_px = None
                if dets[i] is not None:
                    px_per_mm = board_scale_px(detector, dets[i])
                    if px_per_mm:
                        if args.pattern == "charuco":
                            feature_px = px_per_mm * args.marker_mm
                            label += f"   marker {feature_px:.0f}px"
                        else:
                            label += f"   square {px_per_mm * args.square_mm:.0f}px"
                if cams[i].health:
                    label += f"   [{cams[i].health}]"
                color = CAM_COLORS[i] if dets[i] is not None else MUTED
                if feature_px is not None and feature_px < MARKER_PX_FLOOR:
                    color = WARN_COLOR   # too small to decode; move closer or print bigger
                cv2.putText(frames[i], label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)

            canvas = np.hstack(frames)
            bar = np.zeros((72, canvas.shape[1], 3), np.uint8)
            cv2.putText(bar, f"saved {saved} captures  ->  {args.dir}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
            hint = f"shared corners {shared}" if both else "board not visible in both cameras"
            cv2.putText(bar, hint, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        OK_COLOR if shared >= MIN_CORNERS_STEREO else WARN_COLOR, 1, cv2.LINE_AA)
            cv2.putText(bar, f"AUTO {'ON' if auto else 'off'}   SPACE capture   U undo   Q done",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)
            shown = np.vstack([canvas, bar])
            if display_scale is None:
                display_scale = fit_scale(shown.shape[1], shown.shape[0])
                if display_scale < 1.0:
                    print(f"Preview shown at {display_scale:.2f}x to fit the screen "
                          f"(images are saved at full resolution).")
            if display_scale != 1.0:
                shown = cv2.resize(shown, None, fx=display_scale, fy=display_scale,
                                   interpolation=cv2.INTER_AREA)
            if not window_sized:
                cv2.resizeWindow(window, shown.shape[1], shown.shape[0])
                window_sized = True
            cv2.imshow(window, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                n = store(capture_burst(cams, args.burst, args.burst_mode))
                print(f"captured #{n}  (preview corners {counts[0]}/{counts[1]}, shared {shared})")
            elif key == ord("a"):
                auto = not auto
                print(f"auto-capture {'on' if auto else 'off'}")
            elif key == ord("u") and saved:
                for cam_i in (1, 2):
                    path = os.path.join(args.dir, f"cam{cam_i}_{saved:04d}.png")
                    if os.path.exists(path):
                        os.remove(path)
                saved -= 1
                print(f"undo -> {saved} captures")
    finally:
        pool.shutdown()
        for c in cams:
            c.release()
        cv2.destroyAllWindows()

    if saved:
        # Record what the board actually was, so the calibrate step cannot be
        # run against different dimensions than the images were shot with --
        # a silent way to get a confidently wrong calibration.
        write_session(args, saved)
        print(f"\nSaved {saved} captures to {args.dir}")
        print(f"Now run:  python {os.path.basename(__file__)} calibrate {args.dir}")
    else:
        print("\nNothing captured.")


def cmd_calibrate(args):
    """Detect the board in saved images and solve the calibration."""
    session = read_session(args.dir)
    for key in ("cols", "rows", "square_mm", "marker_mm", "dictionary", "pattern"):
        # Command-line values win; otherwise fall back to what capture recorded.
        if getattr(args, key, None) is None:
            setattr(args, key, session.get(key, BOARD_DEFAULTS[key]))
    if session:
        print(f"Board settings from {os.path.join(args.dir, SESSION_FILE)}: "
              f"{args.pattern} {args.cols}x{args.rows}, square {args.square_mm} mm")

    detector = make_detector(args)
    pairs = load_pairs(args.dir)
    if not pairs:
        raise SystemExit(f"No cam1_*.png / cam2_*.png images found in {args.dir}")

    print(f"Detecting the board in {len(pairs)} captures...")
    captures, sizes = [], [None, None]
    for n, (idx, frames) in enumerate(pairs, 1):
        dets = []
        for cam_i, frame in enumerate(frames):
            if frame is None:
                dets.append(None)
                continue
            sizes[cam_i] = (frame.shape[1], frame.shape[0])
            dets.append(detector.detect(frame))
        captures.append(tuple(dets))
        got = [0 if d is None else len(d[1]) for d in dets]
        shared = 0
        if all(d is not None for d in dets):
            pts = common_points(detector, dets[0], dets[1])
            shared = 0 if pts is None else len(pts[0])
        print(f"  [{n:3d}/{len(pairs)}] {idx}: cam1={got[0]:3d} cam2={got[1]:3d} shared={shared:3d}")

    if any(s is None for s in sizes):
        raise SystemExit("Need images from both cameras to solve the extrinsics.")

    print(f"\nCalibrating from {len(captures)} captures...")
    cam1, cam2, R, T, report = calibrate(detector, captures, sizes,
                                         fix_aspect=not args.free_aspect,
                                         fix_k3=not args.free_k3)
    args.camera = session.get("camera", 0)
    args.camera2 = session.get("camera2", 2)
    save(args.out, detector, args, sizes, cam1, cam2, R, T, report)

    if any(k.endswith("_warnings") for k in report):
        print("\nThis calibration did not pass its sanity checks. The reprojection RMS")
        print("above only measures fit to the views you captured, so a suspect model can")
        print("still score well -- capture more views (30-40), with the board tilted 30-45")
        print("degrees in both axes and pushed into all four corners of each frame.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def board_args(sp, defaults=True):
        d = BOARD_DEFAULTS if defaults else {k: None for k in BOARD_DEFAULTS}
        sp.add_argument("--cols", type=int, default=d["cols"], help="squares across")
        sp.add_argument("--rows", type=int, default=d["rows"], help="squares down")
        sp.add_argument("--square-mm", type=float, default=d["square_mm"],
                        help="MEASURED printed square size")
        sp.add_argument("--marker-mm", type=float, default=d["marker_mm"],
                        help="MEASURED printed marker size")
        sp.add_argument("--dictionary", default=d["dictionary"],
                        help="ArUco dictionary (charuco only)")
        sp.add_argument("--pattern", choices=["charuco", "checkerboard"], default=d["pattern"],
                        help="charuco tolerates a partly-visible board; checkerboard "
                             "needs every inner corner in view")
        sp.add_argument("--no-refine", action="store_true",
                        help="disable sub-pixel corner refinement (not recommended)")

    c = sub.add_parser("capture", help="collect board images from both cameras")
    c.add_argument("dir", nargs="?", default=DEFAULT_CAPTURE_DIR,
                   help="directory to write the images into")
    board_args(c)
    c.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDICES[0])
    c.add_argument("--camera2", type=int, default=DEFAULT_CAMERA_INDICES[1])
    # Match tracker.py's capture mode. Calibrating at a different resolution
    # than you track at means adapt_intrinsics has to model how the camera
    # changes mode -- a sensor crop across the 4:3/16:9 boundary, not a plain
    # rescale -- and it also extrapolates the distortion model into normalised
    # radii the calibration never observed. Both go away if the two match.
    #
    # 1280x720 also doubles the pixels across a ChArUco marker, which is what
    # decides whether the board detects at all: a 5x5 marker needs a 7x7 bit
    # grid, so below ~14 px across it cannot be decoded at any exposure. A
    # 16.2 mm marker at 1.5 m is 6.8 px at 640x480 and 10.3 px at 1280x720 --
    # so if detection is still poor, the fix is a physically larger board
    # (or shooting the intrinsics closer), not a better camera.
    c.add_argument("--width", type=int, default=1280, help="capture width")
    c.add_argument("--height", type=int, default=720, help="capture height")
    # Only matters for the stereo pairs, and only a little: the board is held
    # still, so skew between the two cameras barely registers here. It is
    # worth asking for anyway so capture runs in the same mode as tracking.
    c.add_argument("--fps", type=int, default=30,
                   help="frame rate to request (higher = less skew between the cameras)")
    c.add_argument("--raw-format", action="store_true", help="don't force MJPEG capture")
    c.add_argument("--auto-interval", type=float, default=1.0, help="seconds between auto-captures")
    # 1 by default: a handheld board moves between frames, so anything that
    # combines several of them trades sensor noise for motion blur, and a
    # blurred edge has no well-defined corner position at all. Raise --burst
    # only if you can hold the board genuinely still (or rest it on
    # something), and leave the mode on "sharpest" unless it is propped up.
    c.add_argument("--burst", type=int, default=1,
                   help="frames to take per capture (1 = single shot)")
    c.add_argument("--burst-mode", choices=["sharpest", "average"], default="sharpest",
                   help="with --burst > 1: keep the crispest frame, or blend them")
    # Preview detection is the expensive part of the capture loop and is only
    # an aiming aid, so it is throttled by default.
    c.add_argument("--detect-every", type=int, default=2,
                   help="run preview detection every Nth frame (higher = cheaper)")

    k = sub.add_parser("calibrate", help="solve the calibration from saved images")
    k.add_argument("dir", nargs="?", default=DEFAULT_CAPTURE_DIR,
                   help="directory of images from the capture step")
    board_args(k, defaults=False)   # None means "use what capture recorded"
    k.add_argument("--out", default=DEFAULT_OUT, help="where to write the calibration JSON")
    # Both default to constrained; see calibrate()'s docstring for why.
    k.add_argument("--free-aspect", action="store_true",
                   help="let fx and fy differ (default: held equal, as square pixels require)")
    k.add_argument("--free-k3", action="store_true",
                   help="let the third radial distortion term vary (default: pinned at 0)")

    args = p.parse_args()
    (cmd_capture if args.command == "capture" else cmd_calibrate)(args)


if __name__ == "__main__":
    main()
