"""Shared camera capture and window sizing for tracker.py / calibrate_cameras.py.

These two scripts kept drifting apart in ways that only showed up as
"the camera works in one and not the other": different resolutions, one
setting CAP_PROP_BUFFERSIZE and the other not, one reading its cameras
sequentially and the other in parallel, only one rejecting torn frames.
Capture lives here so both get identical behaviour by construction.

This used to be written around a WSL2 usbipd link shared by both cameras,
whose limited bandwidth degraded by delivering *partial* JPEGs rather than
by failing outright. On a native USB connection that pressure is gone: the
cameras can hold 1280x720, and reads no longer have to be serialised to
keep from starving the link. The torn-frame check stays as a cheap safety
net rather than as a load-bearing defence.

What matters instead now is *skew*. Two UVC webcams free-run on independent
clocks, so the two frames a round of reads returns were exposed up to one
frame period apart -- about 11 ms on average at 30 fps, 5.6 ms at 60. No
amount of software fixes that (it needs a hardware trigger), but two things
shrink it: ask for a higher frame rate, and read the cameras concurrently
through read_all so the duration of one read is not added on top of it.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

IS_WINDOWS = sys.platform.startswith("win")

# Linux exposes two /dev/video nodes per UVC camera and only the first of
# each streams, so the second camera lands on index 2. Windows enumerates
# one index per camera.
DEFAULT_CAMERA_INDICES = (0, 1) if IS_WINDOWS else (0, 2)


def open_capture(index):
    """Open a capture with the backend that actually works on this platform.

    Windows defaults to MSMF, which opens slowly and quietly ignores
    resolution and FOURCC requests -- so a camera asked for 1280x720 MJPEG
    silently delivers something else. DirectShow honours both, so it is
    tried first and MSMF kept only as a fallback.
    """
    if IS_WINDOWS:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
        return cv2.VideoCapture(index, cv2.CAP_MSMF)
    return cv2.VideoCapture(index)


def describe_video_devices(wanted_index=None):
    """Explain *why* a camera would not open, by looking at what exists.

    "Missing" and "busy" need opposite fixes and look identical in OpenCV's
    return value, so it is worth checking which one applies.
    """
    if IS_WINDOWS:
        return "\n".join([
            "  On Windows, check that:",
            "    - no other app holds the camera (Teams, Zoom, the Camera app)",
            "    - Settings > Privacy & security > Camera allows desktop apps",
            "    - the index is right: cameras are 0, 1, 2 ... with no gaps",
            "  If the camera is passed through to WSL via usbipd, Windows cannot",
            "  see it until you detach it there.",
        ])

    import glob
    nodes = sorted(glob.glob("/dev/video*"))
    lines = [f"  present: {', '.join(nodes) if nodes else 'NO /dev/video* devices at all'}"]

    node = f"/dev/video{wanted_index}" if wanted_index is not None else None
    if node and node not in nodes:
        lines += [
            f"  {node} does not exist, so nothing is holding it -- the camera is",
            "  detached from WSL. Re-attach it from an Administrator PowerShell:",
            "      usbipd list                       # find the BUSID",
            "      usbipd attach --wsl --busid <ID>  # add --auto-attach if it keeps dropping",
            "  A forced bind is reclaimable by Windows, so an integrated camera can",
            "  disappear mid-session if another app (or Windows Hello) grabs it.",
        ]
    elif node:
        lines += [
            f"  {node} exists but would not open, so something else holds it:",
            "      ps -eo pid,cmd | grep -i python   # look for a leftover tracker/calibrate",
            "  Each camera also exposes two nodes and only the first streams,",
            "  so the capture indices are typically 0 and 2, not 0 and 1.",
        ]
    return "\n".join(lines)


def screen_size(fallback=(1600, 900)):
    """Usable screen size, or a conservative guess if it can't be queried."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return fallback


def fit_scale(canvas_w, canvas_h, margin=0.92):
    """Largest scale <= 1 that keeps a canvas inside the screen.

    Layouts here grow with capture resolution -- two 1280x720 feeds plus a
    panel is 2900 px wide -- and a window manager responds by clipping,
    which reads as the video being cut off rather than the window being
    oversized.
    """
    sw, sh = screen_size()
    return min(1.0, (sw * margin) / canvas_w, (sh * margin) / canvas_h)


def is_torn(frame, threshold=0.15):
    """True if this frame is a truncated MJPEG decode.

    When the link delivers an incomplete JPEG, the decoder emits the rows it
    managed to read and fills the rest with an exactly-zero green channel
    under a strong blue/red cast. That magenta fill cannot occur in real
    imagery, which makes it a reliable signature -- measured on a torn frame
    here, the green channel read 0.0 across 86% of rows. Sampled on a coarse
    grid so the check is essentially free.
    """
    if frame is None or frame.ndim != 3:
        return False
    s = frame[::8, ::8].astype(np.int16)
    filled = (s[:, :, 1] == 0) & ((s[:, :, 0] + s[:, :, 2]) > 80)
    return float(filled.mean()) > threshold


class Camera:
    """One capture device.

    Two cameras cannot be synchronised in software, so the aim here is only
    to keep the unavoidable skew small: ask for the highest frame rate the
    camera will give (a shorter frame period is a smaller worst-case skew)
    and keep the driver queue shallow so a read returns the newest frame
    rather than a banked one. read_all then does the reads concurrently.
    """

    def __init__(self, index, width=1280, height=720, force_mjpg=True, warmup=6,
                 name=None, fps=None):
        self.index = index
        self.name = name or f"camera {index}"
        self.torn_frames = 0
        self.dropped_frames = 0
        self.stale_frames = 0
        self.last_good = None
        self.frame_time = None
        self.fps = None
        self._read_times = []

        self.cap = open_capture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {self.name} (index {index}).\n" + describe_video_devices(index))

        # Fall back to VGA if the requested mode opens but never streams.
        for attempt, (w, h) in enumerate([(width, height), (640, 480)]):
            self._configure(w, h, force_mjpg, fps)
            frame = None
            for _ in range(warmup):
                ok, f = self.cap.read()
                if ok and f is not None and not is_torn(f):
                    frame = f
            if frame is not None:
                self.size = (frame.shape[1], frame.shape[0])
                self.last_good = frame
                self.frame_time = time.perf_counter()
                # What the driver says it settled on, which is not always what
                # was asked for -- a camera that cannot do 720p60 will quietly
                # give 720p30 instead, and that doubles the skew budget.
                reported = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = reported if reported and reported > 0 else None
                rate = f" @ {self.fps:.0f} fps" if self.fps else ""
                if attempt:
                    print(f"  {self.name}: {width}x{height} would not stream cleanly; "
                          f"using {self.size[0]}x{self.size[1]}{rate}")
                elif self.size != (w, h):
                    print(f"  {self.name}: asked {w}x{h}, camera chose {self.size[0]}x{self.size[1]}{rate}")
                else:
                    print(f"  {self.name}: streaming {self.size[0]}x{self.size[1]}{rate}")
                if fps and self.fps and self.fps < fps - 1:
                    print(f"    (asked for {fps} fps; at {self.fps:.0f} fps the worst-case skew "
                          f"between the two cameras is {1000.0 / self.fps:.0f} ms)")
                return
            if (w, h) == (640, 480):
                break
            print(f"  {self.name}: no usable frames at {w}x{h}, retrying at 640x480...")

        self.cap.release()
        raise SystemExit(f"{self.name} (index {index}) opened but never delivered a usable frame.")

    def _configure(self, w, h, force_mjpg, fps=None):
        if force_mjpg:
            # MJPEG is compressed on-device, so far less has to cross the wire.
            # Raw YUYV simply has no bandwidth headroom for 720p at 30+ fps on
            # USB 2.0, which is exactly the mode worth having. Must be set
            # before the frame size.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps:
            # After the frame size: a UVC camera's available rates depend on
            # the mode, so asking before the size is set picks from the wrong
            # list. Halving the frame period halves the worst-case skew
            # between two free-running cameras, which is the one lever
            # software has on synchronisation.
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not IS_WINDOWS:
            # Shallow queue, so a read returns the newest frame instead of one
            # banked earlier -- a stale frame is skew, and unlike the phase
            # offset between the cameras this part is avoidable. DirectShow
            # has no equivalent and ignores the request.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self, retries=2):
        """Return the newest good frame, or the previous one if this read
        came back torn. Never returns a corrupt frame.

        Falling back to the previous frame is a last resort, and it is
        counted: a repeated frame is silently *old*, and triangulating one
        against a fresh frame from the other camera puts the two views at
        different instants. That was routine over usbipd and should be rare
        on native USB, so a rising stale count means something is wrong
        rather than being the normal cost of the link.
        """
        started = time.perf_counter()
        try:
            for _ in range(retries + 1):
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.dropped_frames += 1
                    continue
                if is_torn(frame):
                    self.torn_frames += 1
                    continue
                self.last_good = frame
                self.frame_time = time.perf_counter()
                return frame
            self.stale_frames += 1
            return self.last_good
        finally:
            # Rolling read time. Cameras on one link can differ enormously --
            # a slow one caps the whole loop, since reads are sequential --
            # and without this the cause is invisible.
            self._read_times.append(time.perf_counter() - started)
            del self._read_times[:-30]

    @property
    def read_ms(self):
        """Mean time a read blocks for, in milliseconds.

        This is what a camera costs the loop, which is the number that
        matters when reads are sequential -- the slowest camera sets the
        pace. It is not the camera's independent frame rate: a camera whose
        frame is already waiting returns in ~1 ms regardless of how fast it
        actually runs.
        """
        if not self._read_times:
            return 0.0
        return 1000.0 * sum(self._read_times) / len(self._read_times)

    @property
    def health(self):
        """Short status string, empty while nothing has gone wrong."""
        bits = [f"{self.read_ms:.0f}ms"] if self._read_times else []
        if self.torn_frames:
            bits.append(f"{self.torn_frames} torn")
        if self.dropped_frames:
            bits.append(f"{self.dropped_frames} dropped")
        if self.stale_frames:
            bits.append(f"{self.stale_frames} stale")
        return "  ".join(bits)

    def release(self):
        self.cap.release()


_read_pool = None
_read_pool_size = 0


def read_all(cams):
    """Read every camera at once, returning one frame per camera in order.

    Concurrent rather than sequential. The cameras still free-run on their
    own clocks, so this cannot remove the phase offset between them -- but
    reading one after the other *adds* the duration of the first read on top
    of that offset, and a read that waits on the driver for a new frame is
    not cheap. Issuing them together removes that added term.

    Reads used to be deliberately serialised because two simultaneous reads
    were the peak demand the usbipd link could least serve, and that is what
    tore frames. Off that link the tradeoff reverses.
    """
    global _read_pool, _read_pool_size
    if len(cams) < 2:
        return [c.read() for c in cams]
    if _read_pool is None or _read_pool_size < len(cams):
        if _read_pool is not None:
            _read_pool.shutdown(wait=False)
        _read_pool = ThreadPoolExecutor(max_workers=len(cams), thread_name_prefix="camread")
        _read_pool_size = len(cams)
    return list(_read_pool.map(lambda c: c.read(), cams))


def read_skew_ms(cams):
    """Spread between when the cameras' current frames came back, in ms.

    An observed lower bound on the real exposure skew, not a measurement of
    it: it times when OpenCV handed each frame over, which is after exposure
    by an unknown amount. Useful for spotting a camera that has fallen out
    of step, not for correcting anything.
    """
    times = [c.frame_time for c in cams if c.frame_time is not None]
    return 1000.0 * (max(times) - min(times)) if len(times) > 1 else 0.0


def open_cameras(specs, width, height, force_mjpg=True, fps=None):
    """Open several cameras in order. specs is a list of (index, name)."""
    print("Opening cameras...")
    cams = [Camera(idx, width, height, force_mjpg, name=name, fps=fps) for idx, name in specs]
    sizes = {c.size for c in cams}
    if len(sizes) > 1:
        print(f"NOTE: cameras ended up at different resolutions {sorted(sizes)}. "
              f"That is fine -- each camera's intrinsics are recorded against its own size.")
    rates = {c.fps for c in cams if c.fps}
    if len(rates) > 1:
        print(f"NOTE: cameras are running at different frame rates {sorted(rates)}. "
              f"The slower one sets the worst-case skew between the two views.")
    return cams
