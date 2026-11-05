"""The demo server: static files, the processing endpoint and session handling (M9)."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from hushwave import demo_server
from hushwave.demo_server import BLOCK, MAX_BODY, MAX_SESSIONS, DemoApp, make_server
from hushwave.live import LiveSession, load_b5


@pytest.fixture
def app(tmp_path: Path) -> DemoApp:
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "manifest.json").write_text('{"clips": []}', encoding="utf-8")
    (examples / "clip.wav").write_bytes(b"RIFF")
    return DemoApp(load_b5(), examples_root=examples)


@pytest.fixture
def conn(app: DemoApp) -> Iterator[http.client.HTTPConnection]:
    server = make_server(app, "127.0.0.1", 0)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    yield connection
    connection.close()
    server.shutdown()
    server.server_close()


def _get(conn: http.client.HTTPConnection, path: str) -> tuple[int, dict[str, str], bytes]:
    conn.request("GET", path)
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def _post(
    conn: http.client.HTTPConnection, path: str, body: bytes = b""
) -> tuple[int, dict[str, str], bytes]:
    conn.request("POST", path, body=body, headers={"Content-Type": "application/octet-stream"})
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def _open(conn: http.client.HTTPConnection) -> str:
    status, _, body = _post(conn, "/api/session")
    assert status == 200
    sid = json.loads(body)["sid"]
    assert isinstance(sid, str)
    return sid


def test_info_describes_the_frozen_system(conn: http.client.HTTPConnection) -> None:
    status, headers, body = _get(conn, "/api/info")
    info = json.loads(body)
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert info["fs"] == 16_000 and info["block"] == BLOCK
    assert info["delay_samples"] == 511
    assert info["rule"] == "wiener" and info["estimator"] == "mcra"
    assert info["prefilter"] == "spectral"
    assert info["gain"]["g_min"] == 0.03


def test_info_includes_the_benchmark_when_available(tmp_path: Path) -> None:
    perf = tmp_path / "perf.json"
    perf.write_text(
        json.dumps({"timing": [{"block": 256, "real_time_factor": 0.02, "worst_block_ms": 0.5,
                                "block_budget_ms": 16.0}]}),
        encoding="utf-8",
    )  # fmt: skip
    app = DemoApp(load_b5(), performance_path=perf)
    assert app.info()["benchmark"]["block_budget_ms"] == 16.0
    perf.write_text(json.dumps({"timing": [{"block": 128}]}), encoding="utf-8")
    assert app.info()["benchmark"] == {}
    assert DemoApp(load_b5(), performance_path=tmp_path / "missing.json").info()["benchmark"] == {}


def test_serves_the_interface_files(conn: http.client.HTTPConnection) -> None:
    status, headers, body = _get(conn, "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert b"HushWave" in body
    for path, kind in (("/web/app.js", "javascript"), ("/web/style.css", "css")):
        status, headers, _ = _get(conn, path)
        assert status == 200 and kind in headers["Content-Type"]


def test_serves_example_files_and_head(conn: http.client.HTTPConnection) -> None:
    status, headers, body = _get(conn, "/examples/clip.wav")
    assert (status, headers["Content-Type"], body) == (200, "audio/wav", b"RIFF")
    conn.request("HEAD", "/examples/manifest.json")
    response = conn.getresponse()
    assert response.status == 200 and response.read() == b""


@pytest.mark.parametrize(
    "path",
    [
        "/web/../../README.md",
        "/examples/../manifest.json/../../pyproject.toml",
        "/web/nope.js",
        "/examples",
        "/src/hushwave/live.py",
        "/api/unknown",
    ],
)
def test_unknown_and_escaping_paths_are_404(conn: http.client.HTTPConnection, path: str) -> None:
    conn.request("GET", path)
    response = conn.getresponse()
    response.read()
    assert response.status == 404


def test_processing_matches_the_offline_session(
    conn: http.client.HTTPConnection, rng: np.random.Generator
) -> None:
    sid = _open(conn)
    reference = LiveSession(load_b5())
    x = (0.05 * rng.standard_normal(BLOCK * 20)).astype("<f4")
    for i in range(20):
        block = x[i * BLOCK : (i + 1) * BLOCK]
        status, headers, body = _post(conn, f"/api/process?sid={sid}&enh=1", block.tobytes())
        assert status == 200
        out = np.frombuffer(body, dtype="<f4")
        expected, _ = reference.process(block.astype(np.float64), True)
        np.testing.assert_allclose(out, expected, atol=1e-6)
        assert float(headers["X-Compute-Ms"]) >= 0.0
        assert -90.0 <= float(headers["X-Level-In"]) <= 0.0
        assert -90.0 <= float(headers["X-Level-Out"]) <= 0.0


def test_bypass_flag_returns_the_input(conn: http.client.HTTPConnection) -> None:
    sid = _open(conn)
    block = (0.1 * np.sin(np.arange(BLOCK) / 5.0)).astype("<f4")
    _post(conn, f"/api/process?sid={sid}&enh=1", block.tobytes())
    _post(conn, f"/api/process?sid={sid}&enh=0", block.tobytes())
    _, _, body = _post(conn, f"/api/process?sid={sid}&enh=0", block.tobytes())
    np.testing.assert_allclose(np.frombuffer(body, dtype="<f4"), block, atol=1e-6)


@pytest.mark.parametrize("body", [b"", b"abc", b"abcde"])
def test_rejects_bodies_that_are_not_float32(conn: http.client.HTTPConnection, body: bytes) -> None:
    sid = _open(conn)
    status, _, payload = _post(conn, f"/api/process?sid={sid}", body)
    assert status == 400 and "error" in json.loads(payload)


def test_rejects_non_finite_samples(conn: http.client.HTTPConnection) -> None:
    sid = _open(conn)
    block = np.zeros(BLOCK, dtype="<f4")
    block[7] = np.nan
    status, _, payload = _post(conn, f"/api/process?sid={sid}", block.tobytes())
    assert status == 400 and "NaN" in json.loads(payload)["error"]


def test_oversized_body_is_refused_and_the_connection_closes(
    conn: http.client.HTTPConnection,
) -> None:
    sid = _open(conn)
    status, headers, _ = _post(conn, f"/api/process?sid={sid}", bytes(MAX_BODY + 4))
    assert status == 400
    assert headers.get("Connection", "").lower() == "close"


def test_unknown_session_is_404(conn: http.client.HTTPConnection) -> None:
    status, _, _ = _post(conn, "/api/process?sid=nope", bytes(4 * BLOCK))
    assert status == 404


def test_close_session(conn: http.client.HTTPConnection) -> None:
    sid = _open(conn)
    _, _, body = _post(conn, f"/api/close?sid={sid}")
    assert json.loads(body) == {"closed": True}
    _, _, body = _post(conn, f"/api/close?sid={sid}")
    assert json.loads(body) == {"closed": False}


def test_session_limit_and_expiry(
    conn: http.client.HTTPConnection, app: DemoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(MAX_SESSIONS):
        _open(conn)
    status, _, _ = _post(conn, "/api/session")
    assert status == 503
    monkeypatch.setattr(demo_server, "SESSION_TTL_S", -1.0)
    status, _, _ = _post(conn, "/api/session")
    assert status == 200


def test_unknown_post_path_is_404(conn: http.client.HTTPConnection) -> None:
    status, _, _ = _post(conn, "/api/other", b"xxxx")
    assert status == 404


def test_keep_alive_serves_many_requests_on_one_connection(
    conn: http.client.HTTPConnection,
) -> None:
    sid = _open(conn)
    block = np.zeros(BLOCK, dtype="<f4").tobytes()
    for _ in range(50):
        status, _, _ = _post(conn, f"/api/process?sid={sid}", block)
        assert status == 200


def test_invalid_content_length_closes_the_connection(conn: http.client.HTTPConnection) -> None:
    conn.putrequest("POST", "/api/process?sid=x")
    conn.putheader("Content-Length", "abc")
    conn.endheaders()
    response = conn.getresponse()
    response.read()
    assert response.status == 400


def test_default_app_uses_the_frozen_system() -> None:
    assert demo_server.default_app().system.name == "B5"
