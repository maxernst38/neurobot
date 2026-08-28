"""Stereo-calibrate the two tracker cameras from a held ChArUco board.

Hold the board so BOTH cameras see it, and capture 25-40 views at varied
angles, distances and screen positions. The script then solves for:

  * each camera's intrinsics (focal lengths, principal point, distortion)
  * the rigid transform (R, T) from camera 2's frame into camera 1's

and writes them to a JSON file that tracker.py loads.

Views where only one camera sees the board are NOT wasted: they still feed
that camera's own intrinsics. Only the extrinsics need simultaneous views,
so the both-cameras-see-it constraint applies to the stereo step alone --
which is why the script tracks the two counts separately.

Board defaults match make_calibration_board.py's nominal output. If your
print came out scaled, pass the MEASURED sizes:

    python calibrate_cameras.py --square-mm 22.5 --marker-mm 16.2

Controls: SPACE = capture, A = auto-capture, U = undo last,
          C = calibrate and save, Q/Esc = quit.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from camera_io import Camera, fit_scale, open_cameras

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "stereo_calibration.json")

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
    def __init__(self, cols, rows, square_mm, marker_mm, dict_name):
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        self.board = cv2.aruco.CharucoBoard((cols, rows), square_mm, marker_mm, self.dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)
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


def calibrate(detector, captures, image_sizes):
    """Per-camera intrinsics, then extrinsics with those intrinsics held fixed.

    image_sizes is one (w, h) per camera -- they need not match, and a
    camera's intrinsics are only meaningful against the size they were
    measured at.
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
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)
        intrinsics.append((K, dist))
        report[f"cam{cam + 1}_views"] = len(obj_pts)
        report[f"cam{cam + 1}_rms_px"] = float(rms)
        print(f"  camera {cam + 1}: {len(obj_pts):3d} views   reprojection RMS = {rms:.3f} px")

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
            "squares": [args.cols, args.rows],
            "square_mm": args.square_mm,
            "marker_mm": args.marker_mm,
            "dictionary": args.dictionary,
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--camera2", type=int, default=2)
    p.add_argument("--cols", type=int, default=10, help="squares across")
    p.add_argument("--rows", type=int, default=7, help="squares down")
    p.add_argument("--square-mm", type=float, default=25.0, help="MEASURED printed square size")
    p.add_argument("--marker-mm", type=float, default=18.0, help="MEASURED printed marker size")
    p.add_argument("--dictionary", default="DICT_5X5_100")
    p.add_argument("--out", default=DEFAULT_OUT)
    # Defaults match tracker.py. Two reasons to keep them matched: the
    # usbipd link only sustains ~1.7 FPS for two 720p streams against ~4.7
    # at VGA, and intrinsics measured at one resolution have to be adapted
    # to be used at another -- across the 4:3/16:9 boundary these cameras
    # have, that adaptation involves a sensor crop rather than a plain
    # rescale. Calibrating at the tracking resolution avoids both.
    #
    # Raise to --width 1280 --height 720 if the board will not detect at
    # your working distance; the resolution is recorded per camera, so the
    # result stays correct either way.
    p.add_argument("--width", type=int, default=640, help="capture width used for calibration")
    p.add_argument("--height", type=int, default=480, help="capture height used for calibration")
    p.add_argument("--raw-format", action="store_true", help="don't force MJPEG capture")
    p.add_argument("--auto-interval", type=float, default=1.0, help="seconds between auto-captures")
    args = p.parse_args()

    detector = BoardDetector(args.cols, args.rows, args.square_mm, args.marker_mm, args.dictionary)

    cams = open_cameras([(args.camera, "CAM 1"), (args.camera2, "CAM 2")],
                        args.width, args.height, force_mjpg=not args.raw_format)
    print()

    print(f"Board: {args.cols}x{args.rows} squares, square {args.square_mm} mm, "
          f"marker {args.marker_mm} mm, {args.dictionary}")
    print("If those sizes are not what you actually measured on the printout, quit and pass the real ones.\n")
    print("Capture 25-40 views: tilt the board, vary distance, and work it into")
    print("all corners of BOTH frames. Hold still at each capture.")
    print("Controls: SPACE = capture, A = auto-capture, U = undo, C = calibrate+save, Q/Esc = quit.\n")

    captures = []
    auto = False
    last_auto = 0.0
    last_centroid = None
    window = "Stereo Calibration"
    # NORMAL, not AUTOSIZE: two 1280x720 feeds side by side is 2560 px wide,
    # which AUTOSIZE would push off the edge of the screen unresizably.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    display_scale = None
    window_sized = False
    pool = ThreadPoolExecutor(max_workers=2)
    image_sizes = None
    dropped = 0

    try:
        while True:
            # Sequential, exactly as tracker.py does it: two simultaneous
            # reads is the peak demand the usbipd link is least able to
            # serve, and doing that is what tore the frames.
            frames = [cam.read() for cam in cams]
            if any(f is None for f in frames):
                dropped += 1
                if dropped % 30 == 1:
                    print(f"waiting for frames ({dropped} misses so far)...")
                if dropped > 300:
                    print("Cameras stopped delivering frames; giving up.")
                    break
                continue
            frames = [f.copy() for f in frames]  # about to be drawn on
            image_sizes = [(f.shape[1], f.shape[0]) for f in frames]

            # Detection is the slow part and is pure CPU, so it still runs in
            # parallel -- it just no longer competes with the capture.
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
                    captures.append(tuple(dets))
                    last_centroid, last_auto = centroid, now

            n_both = sum(1 for c in captures if c[0] is not None and c[1] is not None)
            n_cam = [sum(1 for c in captures if c[i] is not None) for i in (0, 1)]

            for i in (0, 1):
                cv2.rectangle(frames[i], (0, 0), (image_sizes[i][0], 24), (0, 0, 0), -1)
                label = f"CAM {i + 1}   corners {counts[i]:3d}"
                if cams[i].health:
                    label += f"   [{cams[i].health}]"
                cv2.putText(frames[i], label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            CAM_COLORS[i] if dets[i] is not None else MUTED, 1, cv2.LINE_AA)

            canvas = np.hstack(frames)
            bar = np.zeros((72, canvas.shape[1], 3), np.uint8)
            status = f"captured {len(captures)}   both-cameras {n_both}   cam1 {n_cam[0]}  cam2 {n_cam[1]}"
            cv2.putText(bar, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
            hint = (f"shared corners {shared}" if both else "board not visible in both cameras")
            cv2.putText(bar, hint, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        OK_COLOR if shared >= MIN_CORNERS_STEREO else WARN_COLOR, 1, cv2.LINE_AA)
            cv2.putText(bar, f"AUTO {'ON' if auto else 'off'}   SPACE capture   C calibrate   U undo   Q quit",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)
            shown = np.vstack([canvas, bar])
            if display_scale is None:
                display_scale = fit_scale(shown.shape[1], shown.shape[0])
                if display_scale < 1.0:
                    print(f"Preview shown at {display_scale:.2f}x to fit the screen "
                          f"(detection still runs on the full-resolution frames).")
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
                if dets[0] is None and dets[1] is None:
                    print("No board detected in either camera - not captured.")
                else:
                    captures.append(tuple(dets))
                    print(f"captured #{len(captures)}  cam1={counts[0]} cam2={counts[1]} shared={shared}")
            elif key == ord("a"):
                auto = not auto
                print(f"auto-capture {'on' if auto else 'off'}")
            elif key == ord("u") and captures:
                captures.pop()
                print(f"undo -> {len(captures)} captures")
            elif key == ord("c"):
                if not captures:
                    print("Nothing captured yet.")
                    continue
                print(f"\nCalibrating from {len(captures)} captures...")
                cam1, cam2, R, T, report = calibrate(detector, captures, image_sizes)
                save(args.out, detector, args, image_sizes, cam1, cam2, R, T, report)
                break
    finally:
        pool.shutdown()
        for c in cams:
            c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
