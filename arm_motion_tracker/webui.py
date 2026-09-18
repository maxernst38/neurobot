"""Browser front end for tracker.py, replacing the OpenCV popup window.

The window was the wrong shape for the job. Its panel is drawn with
cv2.putText at whatever size fits, it cannot be scrolled or resized without
rescaling the video with it, its buttons are single keystrokes you have to
remember, and it only exists on the machine with the cameras plugged in --
which is the machine you are standing away from, holding a pose in front of
the lenses. A browser page fixes all four: real text that reflows, real
buttons, and a URL you can open on a phone propped where you can see it.

What moved and what did not: the camera feeds and the 3D projections are
still drawn by OpenCV and arrive here as MJPEG, because they are pixel
renderings of a 3D scene and redoing them in canvas would be a second
implementation to keep in step. Everything textual -- the metrics, the
per-joint alignment errors, the robot state -- is sent as JSON and rendered
as DOM, because that is the part that was hard to read.

Threading: tracker.py's loop is synchronous and owns the cameras, so the
server runs on its own thread with its own event loop. The two sides meet at
two points only, both lock-guarded: publish() hands over the newest frames
and state, take_commands() drains what the browser asked for. Nothing here
ever blocks the tracker loop -- a slow or vanished client affects only its
own response.

Encoding is skipped entirely when nobody is watching a given stream, so
running with the page closed costs the tracker nothing but the JSON -- and
what encoding remains happens here, not in the capture loop. publish_frame
only hands over a reference to the frame; the JPEG is made on a worker
thread when a client is ready for one, so a camera loop already sharing a
CPU with two MediaPipe graphs never waits on cv2.imencode, and a slow client
costs frames rather than frame rate.
"""

import asyncio
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "web", "index.html")

# How long a stream handler keeps waiting on a tracker that has stopped
# publishing before it gives up and closes the response. Only reached if the
# tracker loop dies while a browser is still attached.
STALE_AFTER_S = 10.0


class WebUI:
    """Serves the tracker UI and carries commands back from the browser."""

    def __init__(self, host="127.0.0.1", port=8080, quality=70, max_width=640):
        self.host = host
        self.port = port
        self.quality = quality
        self.max_width = max_width
        self._lock = threading.Lock()
        self._frames = {}        # name -> (seq, BGR frame, not yet encoded)
        self._jpeg = {}          # name -> (seq, jpeg bytes), encoded on demand
        self._viewers = {}       # name -> attached client count
        self._state = {}
        self._state_seq = 0
        self._commands = queue.SimpleQueue()
        self._sockets = set()
        self._thread = None
        self._loop = None
        self._runner = None
        self._encoder = None
        self._ready = threading.Event()
        self._error = None
        self._closing = False

    # --- tracker-facing API (called from the capture thread) --------------

    @property
    def url(self):
        shown = "localhost" if self.host in ("0.0.0.0", "127.0.0.1") else self.host
        return f"http://{shown}:{self.port}/"

    def start(self):
        """Start the server thread. Raises if the port is not available."""
        self._thread = threading.Thread(target=self._run, name="webui", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._error:
            raise self._error

    def wants(self, name):
        """True if any browser is currently pulling this stream.

        Lets the caller skip work it would only throw away -- the 3D
        projection row costs real time to render, and nothing needs it while
        the page is closed or that panel is collapsed.
        """
        with self._lock:
            return self._viewers.get(name, 0) > 0

    def publish_frame(self, name, image):
        """Offer a BGR image as the newest frame of a stream.

        Returns immediately: this stores a reference and nothing else. The
        caller must not write into the array afterwards, which suits how the
        frames arrive -- each pass of the tracker loop annotates a fresh copy.
        """
        if image is None or not self.wants(name):
            return
        with self._lock:
            seq = self._frames.get(name, (0, None))[0] + 1
            self._frames[name] = (seq, image)

    def _encode(self, image):
        if self.max_width and image.shape[1] > self.max_width:
            scale = self.max_width / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    async def _jpeg_for(self, name, since_seq):
        """Newest frame of a stream as JPEG, or (seq, None) if unchanged.

        Encoded at most once per frame however many clients are watching,
        and always on a worker thread: cv2.imencode releases the GIL, so this
        keeps both the capture loop and the event loop out of it.
        """
        with self._lock:
            seq, image = self._frames.get(name, (-1, None))
            cached = self._jpeg.get(name)
        if image is None or seq == since_seq:
            return seq, None
        if cached and cached[0] == seq:
            return cached
        data = await asyncio.get_running_loop().run_in_executor(
            self._encoder, self._encode, image)
        with self._lock:
            self._jpeg[name] = (seq, data)
        return seq, data

    def publish_state(self, state):
        """Offer the newest JSON state. Cheap; always accepted."""
        with self._lock:
            self._state = state
            self._state_seq += 1

    def take_commands(self):
        """Drain everything the browser has asked for since the last call."""
        out = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        """Shut the server down without letting an open page delay exit.

        Every wait here is bounded. aiohttp's graceful cleanup waits for
        in-flight handlers, and an attached browser always has two: a
        websocket parked in `async for`, and an MJPEG response that only ends
        when the client goes away. Closing those first, and capping what is
        left, is the difference between exiting now and run_teleop.py giving
        up on this process and killing it -- which is what puts a
        ConnectionResetError traceback in the robot bridge's console.
        """
        if self._loop is None:
            return
        self._closing = True
        try:
            self._loop.call_soon_threadsafe(self._close_sockets)
        except RuntimeError:
            pass
        if self._thread:
            self._thread.join(timeout=5)
        if self._encoder:
            # Explicit: concurrent.futures joins every live executor thread
            # at interpreter exit, so an abandoned one would hang the process
            # right at the end, past anything that could report why.
            self._encoder.shutdown(wait=False)

    def _close_sockets(self):
        for ws in list(self._sockets):
            asyncio.ensure_future(ws.close(code=1001, message=b"tracker stopping"))
        self._loop.call_later(0.25, self._loop.stop)

    # --- server side (its own thread, its own event loop) -----------------

    def _run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._encoder = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jpeg")
            self._loop.run_until_complete(self._start_site())
        except Exception as exc:                      # noqa: BLE001 - reported to the caller
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    async def _start_site(self):
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._page)
        app.router.add_get("/shutdown", self._shutdown)
        app.router.add_get("/ws", self._ws)
        app.router.add_get("/stream/{name}.mjpg", self._mjpeg)
        # shutdown_timeout, not the 60 s default: handlers are closed
        # explicitly in stop(), and anything still running past a second is
        # not worth delaying the tracker's exit for.
        self._runner = web.AppRunner(app, access_log=None, shutdown_timeout=1.0)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()

    async def _shutdown(self, request):
        """Stop the tracker, for a supervisor that has no signal to send.

        Queues the same command the page's Quit button sends, so the loop
        closes its cameras and its robot link on the way out instead of
        being killed mid-frame. See run_teleop.py.
        """
        from aiohttp import web

        self._commands.put("quit")
        return web.Response(text="stopping")

    async def _page(self, request):
        from aiohttp import web

        # Read per request rather than caching: the page is a plain file next
        # to this one, and editing it then reloading the browser should be
        # the whole edit cycle.
        if not os.path.exists(PAGE):
            return web.Response(status=500, text=f"missing UI page: {PAGE}")
        return web.FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    async def _mjpeg(self, request):
        from aiohttp import web

        name = request.match_info["name"]
        response = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store",
        })
        await response.prepare(request)
        with self._lock:
            self._viewers[name] = self._viewers.get(name, 0) + 1
        last_seq, last_change = -1, time.monotonic()
        try:
            while not self._closing:
                seq, data = await self._jpeg_for(name, last_seq)
                if data is None:
                    # Polling rather than waiting on an event: the producer is
                    # a plain thread with no access to this loop, and 5 ms of
                    # latency is invisible next to a 30 fps camera.
                    if time.monotonic() - last_change > STALE_AFTER_S:
                        break
                    await asyncio.sleep(0.005)
                    continue
                last_seq, last_change = seq, time.monotonic()
                await response.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                    + data + b"\r\n")
        except (ConnectionResetError, ConnectionAbortedError, asyncio.CancelledError):
            pass
        finally:
            with self._lock:
                self._viewers[name] = max(0, self._viewers.get(name, 1) - 1)
        return response

    async def _ws(self, request):
        from aiohttp import web

        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._sockets.add(ws)
        sender = asyncio.create_task(self._push_state(ws))
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                action = data.get("action")
                if action:
                    # Straight onto the queue. Acting on it is the tracker
                    # loop's business -- it owns the cameras, the robot link
                    # and every flag these commands touch.
                    self._commands.put(str(action))
        finally:
            sender.cancel()
            self._sockets.discard(ws)
        return ws

    async def _push_state(self, ws):
        last = -1
        while not self._closing:
            with self._lock:
                seq, state = self._state_seq, self._state
            if seq != last:
                last = seq
                try:
                    await ws.send_str(json.dumps(state))
                except Exception:                     # noqa: BLE001 - client went away
                    return
            await asyncio.sleep(0.03)
