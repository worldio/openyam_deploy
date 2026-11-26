"""A live web page of what the policy sees (scripts/visualize.py): every calibrated camera's view, reprojected as the
server serves it, with the node graph drawn on it from the arm's live joints as omniact-molmospaces'
visualize_node_graph.py draws a WODS frame (nodegraph.py). It only listens to the robot, so it runs beside the server.

``Live`` redraws the newest snapshot at up to ``fps`` while any page watches (and not at all otherwise); ``serve``
serves it: ``/`` the page, ``/stream.mjpg`` the drawing as an MJPEG stream, ``/frame.jpg`` one drawing, ``/status``
what is happening (JSON).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .nodegraph import get_font, strip
from .ros import NotReady

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenYAM live views</title>
<style>
  body { margin: 0; background: #1a1a1e; color: #d8d8de; font: 14px system-ui, sans-serif; }
  header { padding: 8px 16px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; }
  #status { color: #9a9aa6; }
  img { display: block; width: 100%; height: auto; }
</style></head>
<body>
<header><h1>OpenYAM: the policy's views, node graph from the live joints</h1><span id="status">connecting</span></header>
<img src="stream.mjpg" alt="the calibrated views with the node graph">
<script>
async function poll() {
  try {
    const s = await (await fetch('status', {cache: 'no-store'})).json();
    document.getElementById('status').textContent = s.status + (s.fps ? ' · ' + s.fps.toFixed(1) + ' fps' : '');
  } catch (e) { document.getElementById('status').textContent = 'the visualizer is not answering'; }
  setTimeout(poll, 1000);
}
poll();
</script>
</body></html>
"""


class Live:
    """The newest drawing of ``views`` as a JPEG, from ``ros`` snapshots through ``embodiment``."""

    def __init__(self, ros, embodiment, views, fps=10.0, scale=2):
        self.ros, self.emb, self.views = ros, embodiment, sorted(views)
        self.intrinsics = {name: embodiment.views[name].view['K'] for name in self.views}
        self.period, self.scale = 1.0 / float(fps), int(scale)
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.status, self.fps, self.watchers = None, 0, 'waiting for a page', 0.0, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def watch(self, delta):
        with self.cond:
            self.watchers += delta
            self.cond.notify_all()

    def next(self, seq, timeout=5.0):
        """The first drawing after ``seq``: (jpeg, its seq), or (None, seq) after ``timeout``."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq or self._stop.is_set(), timeout)
            return (self.jpeg, self.seq) if self.seq > seq else (None, seq)

    def close(self):
        self._stop.set()
        with self.cond:
            self.cond.notify_all()
        self._thread.join(timeout=2.0)

    def draw(self):
        """One drawing (RGB) of the newest snapshot, or of what is missing."""
        try:
            joints, frames = self.ros.snapshot(self.views)
        except NotReady as error:
            return _message(f'{error}'), f'{error}'
        return strip(self.emb.build(joints, frames, self.views, ''), self.intrinsics, self.scale), 'live'

    def _run(self):
        import cv2
        drawn = []
        while not self._stop.is_set():
            with self.cond:
                self.cond.wait_for(lambda: self.watchers > 0 or self._stop.is_set())
            began = time.monotonic()
            try:
                image, status = self.draw()
            except Exception as error:                       # keep serving: show what went wrong
                image, status = _message(f'{type(error).__name__}: {error}'), f'error: {error}'
            ok, jpeg = cv2.imencode('.jpg', np.ascontiguousarray(image[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
            drawn = [t for t in drawn if t > began - 2.0] + [began]
            with self.cond:
                self.jpeg, self.seq, self.status = jpeg.tobytes(), self.seq + 1, status
                self.fps = (len(drawn) - 1) / max(drawn[-1] - drawn[0], 1e-9) if len(drawn) > 1 else 0.0
                self.cond.notify_all()
            self._stop.wait(max(0.0, self.period - (time.monotonic() - began)))


def serve(live, port, host='0.0.0.0'):
    """An HTTP server for ``live`` on ``host``:``port``, serving from its own thread; returns it (``shutdown()``)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                self._send(PAGE.encode(), 'text/html; charset=utf-8')
            elif path == '/status':
                with live.cond:
                    body = {'status': live.status, 'fps': live.fps, 'watchers': live.watchers}
                self._send(json.dumps(body).encode(), 'application/json')
            elif path == '/frame.jpg':
                live.watch(+1)
                try:
                    with live.cond:
                        seq = live.seq
                    jpeg, _ = live.next(seq)
                finally:
                    live.watch(-1)
                if jpeg is None:
                    self.send_error(503, 'no drawing yet')
                else:
                    self._send(jpeg, 'image/jpeg')
            elif path == '/stream.mjpg':
                self._stream()
            else:
                self.send_error(404)

        def _send(self, body, kind):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            live.watch(+1)
            try:
                with live.cond:
                    seq = live.seq                            # from the next drawing: an old one may be stale
                while not live._stop.is_set():
                    jpeg, seq = live.next(seq)
                    if jpeg is not None:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n'
                                         % len(jpeg) + jpeg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass                                          # the page went away
            finally:
                live.watch(-1)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _message(text):
    """A drawing that says ``text`` (what is missing)."""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (960, 120), (26, 26, 30))
    ImageDraw.Draw(image).text((16, 44), text[:120], fill=(255, 215, 60), font=get_font(20))
    return np.asarray(image)
