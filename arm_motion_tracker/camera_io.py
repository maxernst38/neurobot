"""Shared camera capture and window sizing for tracker.py / calibrate_cameras.py.

These two scripts kept drifting apart in ways that only showed up as
"the camera works in one and not the other": different resolutions, one
setting CAP_PROP_BUFFERSIZE and the other not, one reading its cameras
sequentially and the other in parallel, only one rejecting torn frames.
Capture lives here so both get identical behaviour by construction.

The constraint being designed around is a WSL2 usbipd link shared by both
cameras. It has limited, variable bandwidth, and it degrades by delivering
*partial* JPEGs rather than by failing outright -- so the defence is to
keep demand low (MJPEG, shallow queue, no simultaneous reads) and to
recognise a truncated frame when one arrives anyway.
"""

import cv2
import numpy as np


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
    """One capture device, configured for a bandwidth-constrained link.

    Read cameras one after another rather than concurrently: two
    simultaneous reads is the peak demand the link is least able to serve,
    and it is what produces torn frames. Sequential reads are also barely
    slower, since a read is cheap next to whatever the caller does with the
    frame afterwards.
    """

    def __init__(self, index, width=640, height=480, force_mjpg=True, warmup=6, name=None):
        self.index = index
        self.name = name or f"camera {index}"
        self.torn_frames = 0
        self.dropped_frames = 0
        self.last_good = None

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Could not open {self.name} (index {index}).\n"
                f"  - is another script still running and holding it?\n"
                f"  - on WSL2 each camera exposes two /dev/video nodes and only\n"
                f"    the first of each streams, so indices are typically 0 and 2."
            )

        # Fall back to VGA if the requested mode opens but never streams.
        for attempt, (w, h) in enumerate([(width, height), (640, 480)]):
            self._configure(w, h, force_mjpg)
            frame = None
            for _ in range(warmup):
                ok, f = self.cap.read()
                if ok and f is not None and not is_torn(f):
                    frame = f
            if frame is not None:
                self.size = (frame.shape[1], frame.shape[0])
                self.last_good = frame
                if attempt:
                    print(f"  {self.name}: {width}x{height} would not stream cleanly; "
                          f"using {self.size[0]}x{self.size[1]}")
                elif self.size != (w, h):
                    print(f"  {self.name}: asked {w}x{h}, camera chose {self.size[0]}x{self.size[1]}")
                else:
                    print(f"  {self.name}: streaming {self.size[0]}x{self.size[1]}")
                return
            if (w, h) == (640, 480):
                break
            print(f"  {self.name}: no usable frames at {w}x{h}, retrying at 640x480...")

        self.cap.release()
        raise SystemExit(f"{self.name} (index {index}) opened but never delivered a usable frame.")

    def _configure(self, w, h, force_mjpg):
        if force_mjpg:
            # Raw YUYV cannot keep up over the usbipd link and arrives as
            # green/glitchy noise. MJPEG is compressed on-device, so far less
            # has to cross the wire. Must be set before the frame size.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Shallow queue: callers consume slower than the camera produces, so
        # a deep one only banks stale frames and adds buffer pressure.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self, retries=2):
        """Return the newest good frame, or the previous one if this read
        came back torn. Never returns a corrupt frame."""
        for _ in range(retries + 1):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.dropped_frames += 1
                continue
            if is_torn(frame):
                self.torn_frames += 1
                continue
            self.last_good = frame
            return frame
        return self.last_good

    @property
    def health(self):
        """Short status string, empty while nothing has gone wrong."""
        bits = []
        if self.torn_frames:
            bits.append(f"{self.torn_frames} torn")
        if self.dropped_frames:
            bits.append(f"{self.dropped_frames} dropped")
        return "  ".join(bits)

    def release(self):
        self.cap.release()


def open_cameras(specs, width, height, force_mjpg=True):
    """Open several cameras in order. specs is a list of (index, name)."""
    print("Opening cameras...")
    cams = [Camera(idx, width, height, force_mjpg, name=name) for idx, name in specs]
    sizes = {c.size for c in cams}
    if len(sizes) > 1:
        print(f"NOTE: cameras ended up at different resolutions {sorted(sizes)}. "
              f"That is fine -- each camera's intrinsics are recorded against its own size.")
    return cams
