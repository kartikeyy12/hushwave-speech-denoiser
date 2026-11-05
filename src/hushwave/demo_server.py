"""Local server for the demo interface (``make demo-live``, ``make demo-replay``).

The browser is only the audio device: it captures the microphone at 16 kHz, posts blocks of raw
float32 samples to ``/api/process`` and plays what comes back. Every sample is processed by the
streaming engine in Python (``live.LiveSession``). The server listens on the loopback interface
by default and serves two directories only: ``demo/web`` and ``examples``.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from hushwave.live import MAX_BLOCK, ROOT, LiveSession, load_b5
from hushwave.systems import System

BLOCK = 256
MAX_SESSIONS = 4
SESSION_TTL_S = 300.0
MAX_BODY = 4 * MAX_BLOCK
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".wav": "audio/wav",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


class DemoApp:
    """State shared by all requests: the frozen system, the sessions and the file roots."""

    def __init__(
        self,
        system: System,
        web_root: Path | None = None,
        examples_root: Path | None = None,
        performance_path: Path | None = None,
    ) -> None:
        self.system = system
        self.web_root = (web_root or ROOT / "demo" / "web").resolve()
        self.examples_root = (examples_root or ROOT / "examples").resolve()
        self.performance_path = performance_path or ROOT / "results" / "engine_performance.json"
        self._sessions: dict[str, tuple[LiveSession, threading.Lock, float]] = {}
        self._lock = threading.Lock()

    def info(self) -> dict[str, Any]:
        probe = LiveSession(self.system)
        gain, est = dict(self.system.gain), dict(self.system.est)
        timing: dict[str, Any] = {}
        if self.performance_path.is_file():
            data = json.loads(self.performance_path.read_text(encoding="utf-8"))
            row = next((r for r in data.get("timing", []) if r.get("block") == BLOCK), None)
            if row:
                timing = {
                    "real_time_factor": row["real_time_factor"],
                    "worst_block_ms": row["worst_block_ms"],
                    "block_budget_ms": row["block_budget_ms"],
                }
        return {
            "fs": probe.fs,
            "block": BLOCK,
            "delay_samples": probe.delay_samples,
            "delay_ms": probe.delay_ms,
            "frame_ms": self.system.frame_ms,
            "rule": self.system.rule,
            "estimator": self.system.estimator,
            "prefilter": self.system.prefilter,
            "gain": gain,
            "est": est,
            "benchmark": timing,
        }

    def _expire(self, now: float) -> None:
        stale = [sid for sid, (_, _, seen) in self._sessions.items() if now - seen > SESSION_TTL_S]
        for sid in stale:
            del self._sessions[sid]

    def open_session(self) -> tuple[str, LiveSession]:
        with self._lock:
            self._expire(time.monotonic())
            if len(self._sessions) >= MAX_SESSIONS:
                raise OverflowError("too many open sessions")
            sid = secrets.token_hex(8)
            session = LiveSession(self.system)
            self._sessions[sid] = (session, threading.Lock(), time.monotonic())
            return sid, session

    def get_session(self, sid: str) -> tuple[LiveSession, threading.Lock] | None:
        with self._lock:
            entry = self._sessions.get(sid)
            if entry is None:
                return None
            self._sessions[sid] = (entry[0], entry[1], time.monotonic())
            return entry[0], entry[1]

    def close_session(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def resolve_file(self, url_path: str) -> Path | None:
        """Map a URL path to a file under one of the two served roots, or None."""
        parts = [p for p in url_path.split("/") if p]
        if not parts:
            return self._inside(self.web_root, ["index.html"])
        if parts[0] == "web":
            return self._inside(self.web_root, parts[1:])
        if parts[0] == "examples":
            return self._inside(self.examples_root, parts[1:])
        return None

    @staticmethod
    def _inside(root: Path, parts: list[str]) -> Path | None:
        if not parts:
            return None
        candidate = root.joinpath(*parts).resolve()
        if root not in candidate.parents or not candidate.is_file():
            return None
        return candidate


def make_handler(app: DemoApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "HushWave"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(
            self,
            status: int,
            body: bytes = b"",
            content_type: str = "application/octet-stream",
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/api/info":
                self._json(HTTPStatus.OK, app.info())
                return
            target = app.resolve_file(url.path)
            if target is None:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            kind = TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0]
            self._send(HTTPStatus.OK, target.read_bytes(), kind or "application/octet-stream")

        do_HEAD = do_GET  # noqa: N815

        def _body(self) -> bytes | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return None
            if length < 0 or length > MAX_BODY:
                self.close_connection = True
                return None
            return self.rfile.read(length)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/api/session":
                self._body()
                try:
                    sid, session = app.open_session()
                except OverflowError as error:
                    self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
                    return
                self._json(
                    HTTPStatus.OK,
                    {
                        "sid": sid,
                        "fs": session.fs,
                        "block": BLOCK,
                        "delay_samples": session.delay_samples,
                    },
                )
            elif url.path == "/api/process":
                self._process(query)
            elif url.path == "/api/close":
                self._body()
                found = app.close_session(query.get("sid", [""])[0])
                self._json(HTTPStatus.OK, {"closed": found})
            else:
                self._body()
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def _process(self, query: dict[str, list[str]]) -> None:
            body = self._body()
            if body is None or len(body) == 0 or len(body) % 4 != 0:
                self._error(HTTPStatus.BAD_REQUEST, "body must be float32 samples")
                return
            entry = app.get_session(query.get("sid", [""])[0])
            if entry is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown session")
                return
            session, lock = entry
            enhanced = query.get("enh", ["1"])[0] != "0"
            samples = np.frombuffer(body, dtype="<f4").astype(np.float64)
            try:
                with lock:
                    out, stats = session.process(samples, enhanced)
            except ValueError as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._send(
                HTTPStatus.OK,
                out.astype("<f4").tobytes(),
                "application/octet-stream",
                {
                    "X-Level-In": f"{stats.level_in_dbfs:.1f}",
                    "X-Level-Out": f"{stats.level_out_dbfs:.1f}",
                    "X-Compute-Ms": f"{stats.compute_ms:.3f}",
                },
            )

    return Handler


def make_server(app: DemoApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server


def default_app() -> DemoApp:
    return DemoApp(load_b5())
