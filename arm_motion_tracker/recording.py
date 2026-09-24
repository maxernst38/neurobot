"""Recording and replay of tracked poses.

WHAT IS RECORDED, AND WHY THAT MUCH
-----------------------------------
A session file holds three layers of the same instant, because they answer
different questions and only one of them can be recovered from the others:

  1. what each camera saw      -- the 2D landmarks and the model's confidence
  2. where that put the joint  -- the triangulated xyz, in metres
  3. what that measured        -- the joint angles driving the robot

Keeping layer 1 is the decision that makes these files research data rather
than a screen recording. The xyz in layer 2 is a *derived* quantity: it is
what this calibration, these visibility gates and this reprojection limit
made of the detections. Recalibrate the rig, change --max-reproj, or fix a
bug in compute_joint_metrics, and every published number changes -- with the
raw detections kept, an old capture can simply be re-processed against the
new code and the two compared. Without them the capture is frozen at the
moment it was taken, and a calibration error found later invalidates it
permanently instead of merely requiring a re-run.

Layer 3 is stored even though replay recomputes it, for the same reason: it
is the record of what the robot was actually told at the time, which is not
reproducible from anything else once the code has moved on.

FORMAT
------
JSON Lines: one header object, then one object per frame. Chosen over a
binary or a single JSON array because a recording is appended to for minutes
at a time and may be interrupted -- a truncated JSONL file loses its last
line and stays readable, where a truncated JSON array loses everything. It
also needs no library to inspect: head, grep and `json.loads` are enough,
and pandas reads it with `read_json(..., lines=True)`.

Files ending .gz are read and written through gzip, which costs roughly a
factor of five in size. The default stays uncompressed, so an interrupted
recording is still readable.

UNITS AND FRAMES are stated in the header rather than assumed: points in
metres, hip-centred, in MediaPipe's world convention (x right, y down, z
away from camera 1); angles in degrees; time in seconds from the first
frame. Anything reading these files should read them from the header.
"""

import gzip
import json
import os
import sys
import time
import zlib

SCHEMA = "arm-motion-tracker/pose-recording"
VERSION = 1

# Rounding applied on write. Deliberately finer than the measurement:
# these numbers are a record of a computation, not of a length, and any
# rounding shows up later as a discrepancy indistinguishable from a code
# change when an old capture is re-run against new code. Measured on a
# 40-frame synthetic capture, storing points at 0.1 mm moved the replayed
# angles by up to 0.17 deg -- with shoulder_rotation, built from the short
# shoulder-hip lever, amplifying it about 4x more than the other metrics.
# A micrometre costs 7 characters per frame (0.4% of the file) and brings
# that below 0.001 deg, so replay differences mean the code, not the file.
POINT_DP = 6        # metres -> micrometres
LANDMARK_DP = 6     # normalised frame coordinates -> ~0.001 px at 1280 wide
CONF_DP = 3         # a model's own confidence; display-grade
PIXEL_DP = 1        # reprojection error, in pixels
ANGLE_DP = 4        # degrees

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "recordings")


def _open(path, mode):
    """Text handle for a path, transparently gzipped when it ends .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return open(path, mode, encoding="utf-8", newline="\n")


def _tolerant_lines(fh):
    """Iterate a handle, stopping quietly where the data runs out.

    A recording is written while someone is moving in front of the cameras,
    so being interrupted is ordinary rather than exceptional -- a killed
    process, a full disk, a pulled USB drive. Plain text simply ends
    mid-line; gzip raises EOFError or BadGzipFile out of the decompressor at
    the same point. Both mean "this is everything that was written", and
    neither should lose the minutes that were.
    """
    while True:
        try:
            line = fh.readline()
        except (EOFError, OSError, gzip.BadGzipFile, zlib.error):
            return
        if not line:
            return
        yield line


def _round(value, dp):
    return None if value is None else round(float(value), dp)


def _round_list(values, dp):
    return None if values is None else [_round(v, dp) for v in values]


class _Point:
    """Landmark-shaped holder, so replayed points flow through the same
    geometry code as live ones. Mirrors tracker._Point deliberately: this
    module must stay importable without MediaPipe, which tracker is not."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _Landmark:
    """A replayed 2D detection, shaped like MediaPipe's NormalizedLandmark.

    Carries visibility and presence separately because tracker._confidence
    reads both and takes the smaller: collapsing them on write would quietly
    change which joints pass the gate on replay.
    """
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=None, presence=None):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.visibility, self.presence = visibility, presence


def _landmarks_to_json(landmarks):
    """One camera's 2D detection as [x, y, z, visibility, presence] rows.

    A list of rows rather than a list of dicts: 33 landmarks x 5 keys of
    JSON object overhead per frame is most of the file, and the column order
    is fixed by this function and named in the header.
    """
    if landmarks is None:
        return None
    out = []
    for lm in landmarks:
        out.append([
            _round(lm.x, LANDMARK_DP), _round(lm.y, LANDMARK_DP),
            _round(getattr(lm, "z", 0.0), LANDMARK_DP),
            _round(getattr(lm, "visibility", None), CONF_DP),
            _round(getattr(lm, "presence", None), CONF_DP),
        ])
    return out


def _world_to_json(points):
    """Triangulated or world points as [x, y, z] rows, null where missing."""
    if points is None:
        return None
    return [None if p is None else
            [_round(p.x, POINT_DP), _round(p.y, POINT_DP), _round(p.z, POINT_DP)]
            for p in points]


LANDMARK_COLUMNS = ["x", "y", "z", "visibility", "presence"]
POINT_COLUMNS = ["x", "y", "z"]


def build_header(side, cameras, metrics, landmark_names, calibration_report=None,
                 calibration_path=None, settings=None, note="", video=None,
                 privacy=None):
    """The self-describing preamble. Everything a reader needs to interpret
    the frames without this codebase."""
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "side": side,
        "units": {
            "points": "metres, hip-centred, MediaPipe world convention "
                      "(x right, y down, z away from camera 1)",
            "landmarks": "normalised to frame width/height, origin top-left",
            "angles": "degrees",
            "grip": "0 open .. 1 closed",
            "time": "seconds since the first recorded frame",
            "reprojection": "pixels",
        },
        "columns": {"landmarks": LANDMARK_COLUMNS, "points": POINT_COLUMNS},
        "landmarks": landmark_names,
        "metrics": metrics,
        "cameras": cameras,
        "calibration": {"path": calibration_path, "report": calibration_report},
        "settings": settings or {},
        # Present only when footage was kept alongside. The declared fps is
        # the rate that was asked of the cameras, used so players show the
        # clip at roughly the right speed -- the authoritative timing is the
        # per-frame "t" in the rows, which is measured rather than assumed.
        "video": video,
        # What was removed, and therefore what this file can and cannot be
        # used for. Stated at the top level rather than inside "video"
        # because it applies to a landmarks-only capture too: face masking
        # drops the face landmarks whether or not footage was kept. A
        # consumer that needs faces must read this and stop, rather than
        # find zeros and guess at why.
        "privacy": privacy or {"faceMask": "off", "faceLandmarksRemoved": False},
    }


def build_frame(index, t, points, info, results, raw, poses_norm, hand_world,
                triangulated, source, stale=None, video=None, drop_face=False):
    """One instant, as the dict written to the file.

    Only what was actually produced: a key whose value is null means the
    tracker had nothing for it that frame, which is a different claim from
    the key being absent in an older schema version.
    """
    frame = {
        "n": index,
        "t": round(float(t), 4),
        "tri": bool(triangulated),
        "source": source,
        # Each camera's running count of frames it repeated because the
        # driver had nothing new. A row whose count is higher than the
        # previous row's is the same observation seen twice, not motion
        # that stopped -- an analysis measuring velocity or counting
        # samples has to be able to tell those apart, and nothing else in
        # the row says which it is.
        "stale": list(stale) if stale is not None else None,
        # Which frame of each camera's video file this row's landmarks came
        # from. Stored rather than assumed equal to "n" because the encoder
        # drops a frame under load, after which the two would silently
        # diverge and replay would draw a skeleton over the wrong picture.
        # null for a camera whose frame was dropped, or when not recording
        # video at all.
        "v": list(video) if video is not None else None,
        # Layer 1: what each camera saw, before any geometry.
        "cams": [(blank_face_landmarks(_landmarks_to_json(p)) if drop_face
                  else _landmarks_to_json(p)) for p in poses_norm],
        "hand": _world_to_json(hand_world),
        # Layer 2: where the two views put each joint. The face points are
        # dropped along with the 2D ones when masking: a triangulated set of
        # eye, ear and nose positions is a 3D measurement of someone's face,
        # and removing it from one layer while leaving it in another
        # protects nobody. None here already means "not resolved", so
        # nothing downstream needs to learn a new case.
        "xyz": (_blank_rows(_world_to_json(points), FACE_LANDMARKS) if drop_face
                else _world_to_json(points)),
        # Layer 3: what that measured, smoothed as the robot received it,
        # and unsmoothed so a different filter can be tried offline.
        "m": {k: _round(v["value"], ANGLE_DP) for k, v in (results or {}).items()},
        "raw": {k: _round(v, ANGLE_DP) for k, v in (raw or {}).items()},
    }
    if info:
        frame["q"] = {
            "reproj": _round(info.get("reproj"), PIXEL_DP),
            "n_valid": info.get("n_valid"),
            "inferred": info.get("inferred"),
            "valid": info.get("valid"),
            "per_joint_reproj": _round_list(info.get("per_joint_reproj"), PIXEL_DP),
            "confidence": [_round_list(c, CONF_DP) for c in info["confidence"]]
                          if info.get("confidence") else None,
            "reason": info.get("reason"),
        }
    return frame


class Recorder:
    """Appends frames to a session file while the tracker runs.

    Writes as it goes rather than buffering to the end: a capture is minutes
    of someone's time to produce, and losing one to a crash at the end is
    worse than the cost of a line of I/O per frame. The file is flushed on a
    timer rather than every frame, which keeps the syscall off most passes of
    the loop while bounding what an abrupt kill can lose.
    """

    FLUSH_EVERY_S = 2.0

    def __init__(self, directory=RECORDINGS_DIR):
        self.directory = directory
        self.path = None
        self._fh = None
        self.frames = 0
        self.started = 0.0
        self.elapsed = 0.0
        self.error = ""
        self._last_flush = 0.0

    @property
    def active(self):
        return self._fh is not None

    @property
    def bytes_written(self):
        if not self.path or not os.path.exists(self.path):
            return 0
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def start(self, header, name=None, compress=False):
        """Open a new session file. Returns its path, or raises OSError."""
        self.stop()
        os.makedirs(self.directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = name or f"session-{stamp}"
        if not base.endswith(".jsonl") and not base.endswith(".jsonl.gz"):
            base += ".jsonl.gz" if compress else ".jsonl"
        self.path = os.path.join(self.directory, base)
        self._fh = _open(self.path, "w")
        self._fh.write(json.dumps(header) + "\n")
        self.frames = 0
        self.started = time.monotonic()
        self.elapsed = 0.0
        self.error = ""
        self._last_flush = self.started
        return self.path

    def add(self, frame):
        """Write one frame. Never raises: a recording failing mid-session
        must not take the tracker down with it."""
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(frame) + "\n")
            self.frames += 1
            now = time.monotonic()
            self.elapsed = now - self.started
            if now - self._last_flush >= self.FLUSH_EVERY_S:
                self._fh.flush()
                self._last_flush = now
        except (OSError, ValueError, TypeError) as exc:
            self.error = str(exc)
            self.stop()

    def stop(self):
        """Close the file. Returns the path written, or None."""
        path = self.path if self._fh else None
        if self._fh:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
        self._fh = None
        return path


# MediaPipe's first eleven pose landmarks are the face: nose, the six eye
# points, both ears, and the two mouth corners. Nothing downstream uses any
# of them -- the arm metrics read shoulders, elbows, wrists, hips and the
# coarse hand points, all of which are index 11 upward -- so they can be
# removed from a capture at no cost to anything it is for.
FACE_LANDMARKS = tuple(range(11))

MASK_MODES = ("off", "block", "blur")


def face_region(pose_norm, width, height):
    """Pixel box covering the head, from the face landmarks, or None.

    Deliberately much larger than the landmarks themselves. They span only
    eyes to mouth and ear to ear, which is the middle of a face -- the skull,
    hair, jaw and chin all lie outside that hull, and a mask drawn to it
    would leave most of a recognisable head visible.
    """
    if not pose_norm:
        return None
    xs, ys = [], []
    for i in FACE_LANDMARKS:
        if i >= len(pose_norm):
            continue
        lm = pose_norm[i]
        # Visibility is ignored on purpose: a landmark the model is unsure
        # of is still roughly where the face is, and over-covering costs
        # nothing while under-covering defeats the point.
        xs.append(lm.x * width)
        ys.append(lm.y * height)
    if not xs:
        return None
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    # Ear-to-ear is about a head wide; eyes-to-mouth is only about a third
    # of a head tall, so the vertical expansion has to be far larger.
    half_w = max(span_x * 0.95, 30.0)
    half_h = max(span_y * 2.0, half_w * 1.25)
    return (int(max(0, cx - half_w)), int(max(0, cy - half_h)),
            int(min(width, cx + half_w)), int(min(height, cy + half_h)))


def mask_face(frame, pose_norm, mode="block"):
    """Obscure the face in `frame`, in place. Returns True if a face was found.

    FAILS CLOSED. When there is no pose to locate a face from -- the model
    lost the person, they turned away, they were half out of shot -- the
    whole frame is obscured rather than written through. A privacy feature
    whose failure mode is publishing the face is worse than no feature,
    because it is trusted. The cost is a visibly blanked frame, which is
    also the honest signal that detection dropped out.
    """
    import cv2

    if mode == "off":
        return True
    h, w = frame.shape[:2]
    box = face_region(pose_norm, w, h)
    if box is None or box[2] <= box[0] or box[3] <= box[1]:
        # Solid black whatever the mode, including 'blur'. Blurring a whole
        # frame is not a fallback: blur preserves colour and low-frequency
        # structure, so a face filling much of the frame survives it well
        # enough to recognise. The fallback has to be the strongest action
        # available, not the gentler one the mode usually asks for --
        # this is precisely the moment the subject's position is unknown.
        frame[:] = 0
        return False
    x0, y0, x1, y1 = box
    if mode == "blur":
        roi = frame[y0:y1, x0:x1]
        # Pixelate first, then blur: a plain blur is partially invertible,
        # and destroying the detail before smoothing leaves much less to
        # recover. Still weaker than a solid block -- see --mask-face.
        small = cv2.resize(roi, (max(1, (x1 - x0) // 16), max(1, (y1 - y0) // 16)),
                           interpolation=cv2.INTER_AREA)
        roi[:] = cv2.GaussianBlur(
            cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST),
            (31, 31), 0)
    else:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    return True


def _blank_rows(rows, indices):
    """Set the given rows to None (the file's "not resolved")."""
    if rows is None:
        return None
    for i in indices:
        if i < len(rows):
            rows[i] = None
    return rows


def blank_face_landmarks(rows):
    """Zero the face landmarks of one camera's row, keeping its shape.

    Zeroed rather than deleted so the array stays 33 long and every index
    downstream still means the same joint; visibility goes to 0 so the
    normal gates drop them instead of anything having to know they are
    special. Masking the video but leaving eleven face coordinates per frame
    in the session file would not be identity protection.
    """
    if rows is None:
        return None
    for i in FACE_LANDMARKS:
        if i < len(rows):
            rows[i] = [0.0, 0.0, 0.0, 0.0, 0.0]
    return rows


class VideoRecorder:
    """Writes each camera's raw frames beside the session file.

    Optional, and off by default, because it changes the size of a capture
    by roughly an order of magnitude -- measured on this machine at 720p30,
    landmarks alone are 8.8 MB/min for both cameras while H.264 video is
    15-40 MB/min *per* camera. What it buys is the one thing the landmarks
    cannot: re-running the pose model itself. The stored landmarks are a
    model's output, so a better landmarker, or a question about whether a
    joint was genuinely occluded, has nothing to work from without this.

    RAW frames, not the annotated ones shown in the feeds: overlays burned
    into the pixels are exactly what makes footage useless to re-run.

    Encoding happens on a worker thread. It is only 1-2 ms a frame, but the
    capture loop is already sharing a CPU with two MediaPipe graphs, and a
    disk hiccup should cost frames rather than frame rate. The queue is
    therefore bounded and non-blocking: when it fills, that frame is dropped
    and counted, and the row in the session file records no video index for
    it rather than pointing at the wrong picture.
    """

    # ~0.5 s of slack per camera. Bigger is pointless when encoding takes
    # 2 ms against a 33 ms budget, and 16 frames of 720p is already 44 MB.
    QUEUE_FRAMES = 16

    def __init__(self, base_path, sizes, fps=30.0, codec="avc1", mask="off"):
        self.base = os.path.splitext(base_path)[0]
        if self.base.endswith(".jsonl"):
            self.base = self.base[:-len(".jsonl")]
        self.sizes = [tuple(s) for s in sizes]
        self.fps = float(fps) or 30.0
        self.codec = codec
        # Masking belongs to the writer rather than the caller so that there
        # is no code path through this class that writes an unmasked frame.
        # A privacy guarantee that depends on every call site remembering to
        # apply it is not a guarantee.
        self.mask = mask if mask in MASK_MODES else "off"
        self.paths = []
        self.written = [0] * len(self.sizes)
        self.dropped = [0] * len(self.sizes)
        self.blanked = [0] * len(self.sizes)   # frames with no face to locate
        self.error = ""
        self._writers = []
        self._queue = None
        self._thread = None

    @property
    def active(self):
        return bool(self._writers)

    def start(self):
        """Open one video per camera. Returns the filenames, or raises."""
        import queue as _queue
        import threading

        import cv2

        ext = ".avi" if self.codec in ("MJPG", "XVID") else ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        for i, size in enumerate(self.sizes):
            path = f"{self.base}.cam{i}{ext}"
            w = cv2.VideoWriter(path, fourcc, self.fps, (int(size[0]), int(size[1])))
            if not w.isOpened():
                for done in self._writers:
                    done.release()
                self._writers = []
                raise OSError(f"codec {self.codec!r} is not available for "
                              f"{size[0]}x{size[1]} video")
            self._writers.append(w)
            self.paths.append(os.path.basename(path))
        self._queue = _queue.Queue(maxsize=self.QUEUE_FRAMES * max(1, len(self.sizes)))
        self._thread = threading.Thread(target=self._drain, name="video", daemon=True)
        self._thread.start()
        return list(self.paths)

    def add(self, frames, poses_norm=None):
        """Queue one frame per camera; returns the index each was given.

        The index is assigned here rather than by the writer thread, so the
        session row can name it immediately. None means the frame was
        dropped and this row has no footage.

        `poses_norm` locates the face to mask, one per camera. With masking
        on and no pose for a camera, that frame is obscured entirely rather
        than written through -- see mask_face.
        """
        import queue as _queue

        out = []
        for i, frame in enumerate(frames):
            if i >= len(self._writers) or frame is None:
                out.append(None)
                continue
            try:
                # Copied: the camera hands back a buffer it may reuse, and
                # the encoder reads it later on another thread. Masking is
                # applied to the copy, before it is queued, so the frame
                # the tracker keeps using is untouched and the frame that
                # reaches the encoder has never been unmasked.
                private = frame.copy()
                if self.mask != "off":
                    pose = (poses_norm[i] if poses_norm and i < len(poses_norm)
                            else None)
                    if not mask_face(private, pose, self.mask):
                        self.blanked[i] += 1
                self._queue.put_nowait((i, private))
            except _queue.Full:
                self.dropped[i] += 1
                out.append(None)
                continue
            out.append(self.written[i])
            self.written[i] += 1
        return out

    def _drain(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            i, frame = item
            try:
                self._writers[i].write(frame)
            except Exception as exc:               # noqa: BLE001 - reported, never fatal
                self.error = str(exc)

    def stop(self):
        """Flush and close. Returns per-camera stats."""
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            # Bounded: a wedged encoder must not hold up the tracker's exit,
            # and the frames already written are on disk either way.
            self._thread.join(timeout=5)
        for w in self._writers:
            try:
                w.release()
            except Exception:                      # noqa: BLE001
                pass
        self._writers = []
        self._thread, self._queue = None, None
        return {"files": list(self.paths), "written": list(self.written),
                "dropped": list(self.dropped), "blanked": list(self.blanked),
                "mask": self.mask, "error": self.error}


class VideoSource:
    """Reads back the footage of a recording, for replay.

    Sequential reads are the common case -- playback -- and are just
    read(). A jump asks the decoder to seek, which for H.264 means finding
    the nearest keyframe and rolling forward, so it is much slower; that is
    a scrub, where one slow frame is invisible.
    """

    def __init__(self, directory, files):
        self.paths = [os.path.join(directory, f) for f in files]
        self._caps = [None] * len(self.paths)
        self._next = [None] * len(self.paths)

    def _cap(self, i):
        import cv2

        if self._caps[i] is None and i < len(self.paths):
            if not os.path.exists(self.paths[i]):
                return None
            cap = cv2.VideoCapture(self.paths[i])
            if not cap.isOpened():
                return None
            self._caps[i] = cap
            self._next[i] = 0
        return self._caps[i]

    def frame(self, i, index):
        import cv2

        if index is None or i >= len(self.paths):
            return None
        cap = self._cap(i)
        if cap is None:
            return None
        if self._next[i] != index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            self._next[i] = index
        ok, frame = cap.read()
        self._next[i] = index + 1 if ok else None
        return frame if ok else None

    def close(self):
        for cap in self._caps:
            if cap is not None:
                cap.release()
        self._caps = [None] * len(self.paths)


class Recording:
    """A loaded session file."""

    def __init__(self, header, frames, path=""):
        self.header = header
        self.frames = frames
        self.path = path

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def count(self):
        return len(self.frames)

    @property
    def duration(self):
        return self.frames[-1]["t"] if self.frames else 0.0

    @property
    def fps(self):
        return (self.count - 1) / self.duration if self.duration > 0 else 0.0

    @classmethod
    def load(cls, path, limit=None):
        """Read a session file.

        Tolerates a truncated final line, which is what an interrupted
        recording leaves behind and is the case this format was chosen to
        survive -- so it must not be an error here.
        """
        frames, header = [], None
        with _open(path, "r") as fh:
            # A truncated .gz raises from the decompressor rather than
            # returning a short line, so the same interruption has to be
            # caught in two places to mean the same thing.
            lines = _tolerant_lines(fh)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    if header is None:
                        raise ValueError(f"{os.path.basename(path)} is not a "
                                         f"recording: its first line is not JSON")
                    break        # truncated tail; keep what was read
                if header is None:
                    if obj.get("schema") != SCHEMA:
                        raise ValueError(f"{os.path.basename(path)} is not an "
                                         f"arm motion recording")
                    if obj.get("version", 0) > VERSION:
                        raise ValueError(f"{os.path.basename(path)} was written by a "
                                         f"newer version ({obj['version']} > {VERSION})")
                    header = obj
                    continue
                frames.append(obj)
                if limit and len(frames) >= limit:
                    break
        if header is None:
            raise ValueError(f"{os.path.basename(path)} is empty")
        return cls(header, frames, path)

    # --- reconstruction -----------------------------------------------------

    def points(self, i):
        """Triangulated points of frame i, as landmark-shaped objects."""
        rows = self.frames[i].get("xyz")
        if not rows:
            return None
        return [None if r is None else _Point(*r) for r in rows]

    def hand(self, i):
        rows = self.frames[i].get("hand")
        if not rows:
            return None
        return [None if r is None else _Point(*r) for r in rows]

    def poses_norm(self, i):
        """Per-camera 2D detections of frame i, for re-triangulation."""
        out = []
        for cam in self.frames[i].get("cams") or []:
            out.append(None if cam is None else
                       [_Landmark(r[0], r[1], r[2], r[3], r[4]) for r in cam])
        return out

    def info(self, i):
        """The stored triangulation diagnostics, in triangulate_pose's shape."""
        q = self.frames[i].get("q")
        if not q:
            return None
        return {
            "reason": q.get("reason"), "reproj": q.get("reproj"),
            "n_valid": q.get("n_valid") or 0,
            "lost_visibility": 0, "lost_reproj": 0,
            "inferred": q.get("inferred") or [],
            "valid": q.get("valid") or [],
            "per_joint_reproj": q.get("per_joint_reproj") or [],
            "confidence": q.get("confidence") or [],
            "trust_inferred": bool(self.header.get("settings", {}).get("trust_inferred")),
        }

    def sizes(self):
        """Capture resolutions, needed to re-triangulate from the 2D rows."""
        return [tuple(c.get("size", (0, 0))) for c in self.header.get("cameras", [])]

    @property
    def video(self):
        """The footage block, or {} for a landmarks-only recording."""
        return self.header.get("video") or {}

    @property
    def has_video(self):
        """True only if the files are actually still next to the session.

        Video is written as separate files, so it can be deleted, moved, or
        simply not copied along with the .jsonl -- which is a likely thing
        to happen given it is most of the bytes. A recording whose footage
        has gone is still a perfectly good recording, so this is a question
        to ask rather than an error to raise.
        """
        files = self.video.get("files") or []
        here = os.path.dirname(os.path.abspath(self.path))
        return bool(files) and all(os.path.exists(os.path.join(here, f)) for f in files)

    def open_video(self):
        """A VideoSource over this recording's footage, or None."""
        if not self.has_video:
            return None
        return VideoSource(os.path.dirname(os.path.abspath(self.path)),
                           self.video.get("files") or [])

    def video_index(self, i):
        """Per-camera video frame numbers for row i, or None."""
        return self.frames[i].get("v")

    def index_at(self, t):
        """Index of the last frame at or before time t. Linear scan is fine:
        this is called once per loop pass over a few thousand frames."""
        if not self.frames:
            return 0
        lo, hi = 0, len(self.frames) - 1
        if t <= self.frames[0]["t"]:
            return 0
        if t >= self.frames[hi]["t"]:
            return hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.frames[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def summary(self):
        h = self.header
        return {
            "name": self.name, "path": self.path,
            "frames": self.count, "duration": round(self.duration, 1),
            "fps": round(self.fps, 1), "side": h.get("side"),
            "created": h.get("created"), "note": h.get("note", ""),
            "cameras": [c.get("name") for c in h.get("cameras", [])],
            "calibrated": bool((h.get("calibration") or {}).get("report")),
            "video": bool(self.video),
            "videoPresent": self.has_video,
        }


class Player:
    """Playback clock over a loaded Recording.

    Holds position in recording-time rather than frame index, so playback
    speed and a variable loop rate do not accumulate drift, and so a file
    recorded at 12 fps plays at 12 fps rather than as fast as the loop runs.
    """

    SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(self, recording):
        self.recording = recording
        self.playing = False
        self.speed = 1.0
        self.loop = True
        self.t = 0.0
        self.index = 0
        self._wall = None

    def play(self):
        self.playing = True
        self._wall = time.monotonic()

    def pause(self):
        self.playing = False
        self._wall = None

    def toggle(self):
        self.pause() if self.playing else self.play()

    def set_speed(self, speed):
        self.speed = max(0.05, min(16.0, float(speed)))

    def seek_fraction(self, fraction):
        self.seek(max(0.0, min(1.0, float(fraction))) * self.recording.duration)

    def seek(self, t):
        self.t = max(0.0, min(self.recording.duration, float(t)))
        self.index = self.recording.index_at(self.t)
        self._wall = time.monotonic()

    def step(self, frames=1):
        """Move a whole frame at a time, paused. The only way to look at a
        single instant of a capture, which is most of what review is."""
        self.pause()
        self.index = max(0, min(self.recording.count - 1, self.index + frames))
        self.t = self.recording.frames[self.index]["t"] if self.recording.frames else 0.0

    @property
    def finished(self):
        return not self.loop and self.t >= self.recording.duration

    def tick(self):
        """Advance the clock to now and return the current frame index."""
        now = time.monotonic()
        if self.playing and self._wall is not None:
            self.t += (now - self._wall) * self.speed
            if self.t >= self.recording.duration:
                if self.loop and self.recording.duration > 0:
                    self.t = self.t % self.recording.duration
                else:
                    self.t = self.recording.duration
                    self.playing = False
            self.index = self.recording.index_at(self.t)
        self._wall = now
        return self.index

    def state(self):
        rec = self.recording
        return {
            "name": rec.name,
            "playing": self.playing,
            "loop": self.loop,
            "speed": self.speed,
            "t": round(self.t, 2),
            "duration": round(rec.duration, 2),
            "index": self.index,
            "frames": rec.count,
            "fraction": (self.t / rec.duration) if rec.duration > 0 else 0.0,
            "summary": rec.summary(),
        }


def list_recordings(directory=RECORDINGS_DIR):
    """Every readable session file in a directory, newest first.

    Reads only the header of each, so listing a directory of long captures
    costs one line of I/O per file rather than loading them.
    """
    out = []
    if not os.path.isdir(directory):
        return out
    for name in os.listdir(directory):
        if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz")):
            continue
        path = os.path.join(directory, name)
        entry = {"name": name, "size": 0, "frames": None, "duration": None,
                 "created": None, "side": None, "note": "", "error": ""}
        try:
            entry["size"] = os.path.getsize(path)
            with _open(path, "r") as fh:
                header = json.loads(fh.readline())
            if header.get("schema") != SCHEMA:
                raise ValueError("not an arm motion recording")
            entry.update(created=header.get("created"), side=header.get("side"),
                         note=header.get("note", ""))
        except (OSError, ValueError, EOFError, zlib.error) as exc:
            entry["error"] = str(exc) or exc.__class__.__name__
        out.append(entry)
    out.sort(key=lambda e: e.get("created") or "", reverse=True)
    return out


def to_csv(rec, fh, layer="angles"):
    """Flatten a recording into CSV for a spreadsheet or a plot.

    Two layers, because they are different shapes and mixing them gives a
    table that is mostly empty: "angles" is one row per frame, which is what
    a time series of joint angles wants; "points" is one row per frame per
    landmark, which is what a 3D trajectory wants.
    """
    import csv

    keys = [m["key"] for m in rec.header.get("metrics", [])]
    names = {v: k for k, v in (rec.header.get("landmarks") or {}).items()}
    writer = csv.writer(fh, lineterminator="\n")
    if layer == "angles":
        writer.writerow(["frame", "t", "source", "triangulated", "reproj_px", "n_valid",
                         "repeated"] + keys + [f"{k}_raw" for k in keys])
        prev_stale = None
        for f in rec.frames:
            q = f.get("q") or {}
            m, raw = f.get("m") or {}, f.get("raw") or {}
            # The stored counts are cumulative, so a camera repeated a frame
            # on this row exactly when its count rose. 1 here means at least
            # one view is the previous instant's, which is a sample to drop
            # rather than count -- and, triangulated against a fresh view
            # from the other camera, a position from two different moments.
            stale = f.get("stale")
            repeated = ("" if stale is None or prev_stale is None
                        else int(any(s > p for s, p in zip(stale, prev_stale))))
            prev_stale = stale if stale is not None else prev_stale
            writer.writerow([f["n"], f["t"], f.get("source", ""), int(bool(f.get("tri"))),
                             q.get("reproj", ""), q.get("n_valid", ""), repeated]
                            + [m.get(k, "") for k in keys]
                            + [raw.get(k, "") for k in keys])
        return
    writer.writerow(["frame", "t", "landmark", "index", "x", "y", "z",
                     "inferred", "reproj_px", "cam1_conf", "cam2_conf"])
    for f in rec.frames:
        q = f.get("q") or {}
        conf = q.get("confidence") or [[], []]
        inferred = q.get("inferred") or []
        errs = q.get("per_joint_reproj") or []
        at = lambda seq, i: seq[i] if i < len(seq) and seq[i] is not None else ""
        for i, row in enumerate(f.get("xyz") or []):
            if row is None:
                continue
            writer.writerow([
                f["n"], f["t"], names.get(i, i), i, row[0], row[1], row[2],
                int(bool(at(inferred, i))), at(errs, i),
                at(conf[0], i), at(conf[1], i),
            ])


def _main(argv=None):
    """Inspect a recording without starting the tracker."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="recording.py", description=_main.__doc__)
    ap.add_argument("path", nargs="?", help="recording to read; omit to list them")
    ap.add_argument("--dir", default=RECORDINGS_DIR, help="directory to list")
    ap.add_argument("--csv", metavar="FILE", help="write CSV here ('-' for stdout)")
    ap.add_argument("--layer", choices=["angles", "points"], default="angles",
                    help="one row per frame (angles) or per landmark (points)")
    args = ap.parse_args(argv)

    if not args.path:
        rows = list_recordings(args.dir)
        if not rows:
            print(f"No recordings in {args.dir}")
            return 0
        for e in rows:
            print(f"{e['name']:<34} {e['size'] / 1e6:7.2f} MB  {e['created'] or '?'}"
                  f"  {e.get('note') or ''}{'  [' + e['error'] + ']' if e['error'] else ''}")
        return 0

    path = args.path if os.path.exists(args.path) else os.path.join(args.dir, args.path)
    rec = Recording.load(path)
    s = rec.summary()
    print(f"{s['name']}: {s['frames']} frames, {s['duration']}s at {s['fps']} fps, "
          f"{s['side']} arm, recorded {s['created']}")
    if s["note"]:
        print(f"  note: {s['note']}")
    cal = (rec.header.get("calibration") or {}).get("report") or {}
    if cal:
        print(f"  calibration: {cal.get('stereo_rms_px', float('nan')):.2f} px RMS, "
              f"baseline {cal.get('baseline_mm', 0):.0f} mm, "
              f"{cal.get('angle_deg', 0):.1f} deg apart")
    print(f"  settings: {rec.header.get('settings')}")
    priv = rec.header.get("privacy") or {}
    if priv.get("faceMask", "off") != "off":
        print(f"  privacy: faces {priv['faceMask']}"
              + (", face landmarks removed" if priv.get("faceLandmarksRemoved") else "")
              + ("  (pixelation leaves skin tone and head shape)"
                 if priv["faceMask"] == "blur" else ""))
    elif rec.video:
        # Said plainly rather than by omission: this file contains faces,
        # which is what governs how it may be stored and shared.
        print("  privacy: NOT masked — this recording shows the subject's face")
    if rec.video:
        here = os.path.dirname(os.path.abspath(rec.path))
        sizes = [(f, os.path.getsize(os.path.join(here, f)))
                 for f in rec.video.get("files", [])
                 if os.path.exists(os.path.join(here, f))]
        if sizes:
            print(f"  video: {rec.video.get('codec')}, "
                  + ", ".join(f"{f} ({n / 1e6:.1f} MB)" for f, n in sizes))
        else:
            # Worth saying plainly: the rows still point at frame numbers in
            # files that are not here, so replay will fall back to the
            # skeleton rather than failing, and the footage is not lost
            # because of anything in the data.
            print(f"  video: declared ({', '.join(rec.video.get('files', []))}) "
                  f"but not found next to this file")
    tracked = sum(1 for f in rec.frames if f.get("tri"))
    print(f"  {tracked}/{rec.count} frames triangulated "
          f"({100 * tracked / max(1, rec.count):.0f}%)")
    for m in rec.header.get("metrics", []):
        vals = [f["m"].get(m["key"]) for f in rec.frames if (f.get("m") or {}).get(m["key"]) is not None]
        if vals:
            print(f"  {m['key']:<20} {len(vals):5d} readings   "
                  f"{min(vals):8.2f} .. {max(vals):8.2f}  (span {max(vals) - min(vals):.2f})")
        else:
            print(f"  {m['key']:<20}     0 readings")

    if args.csv:
        if args.csv == "-":
            to_csv(rec, sys.stdout, args.layer)
        else:
            with open(args.csv, "w", encoding="utf-8", newline="") as fh:
                to_csv(rec, fh, args.layer)
            print(f"wrote {args.csv} ({args.layer})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
