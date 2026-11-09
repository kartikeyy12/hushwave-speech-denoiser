"""Talk to the HushWave board from the computer, without the browser.

Self-contained: needs only Python 3.10 or newer and pyserial (py -m pip install pyserial).
Run it with the Windows Python, because the board's COM port belongs to Windows, not to WSL.

    py mcu_probe.py --list                 show the serial ports
    py mcu_probe.py --watch 20             show ports appearing and disappearing for 20 s
    py mcu_probe.py --port COM10           open the port, say hello, stream test blocks

If this works and the page does not, the board and the firmware are fine and the problem is in
the browser or in what else holds the port. If this fails too, the problem is the board, the
cable, the port or the Arduino settings. The frame format is in docs/PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

SYNC = b"\xa5\x5a"
BLOCK = 256
MAX_PAYLOAD = 1024
TYPE_AUDIO_IN, TYPE_HELLO, TYPE_RESET = 0x01, 0x02, 0x03
TYPE_AUDIO_OUT, TYPE_HELLO_REPLY, TYPE_RESET_ACK = 0x81, 0x82, 0x83
BLOCK_SECONDS = BLOCK / 16000
LOST_AFTER_S = 0.5
ESPRESSIF_VID = 0x303A
BRIDGES = {
    (0x10C4, 0xEA60): "CP210x USB to UART bridge",
    (0x1A86, 0x7523): "CH340 USB to UART bridge",
    (0x1A86, 0x55D3): "CH343 USB to UART bridge",
    (0x0403, 0x6001): "FTDI USB to UART bridge",
}


# ------------------------------------------------------------------------ protocol
def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: polynomial 0x1021, initial value 0xFFFF, no reflection."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_frame(kind: int, seq: int, payload: bytes = b"") -> bytes:
    body = struct.pack("<BBH", kind, seq, len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


@dataclass(frozen=True)
class Frame:
    kind: int
    seq: int
    payload: bytes


class Parser:
    """Byte-stream parser with sync marker, length and CRC checks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.crc_errors = 0
        self.stray = bytearray()  # the first bytes that were not part of any frame

    def _skip(self, count: int) -> None:
        if len(self.stray) < 64:
            self.stray += self._buffer[:count]
        del self._buffer[:count]

    def push(self, data: bytes) -> list[Frame]:
        self._buffer += data
        frames: list[Frame] = []
        while True:
            start = self._buffer.find(SYNC)
            if start < 0:
                keep = 1 if self._buffer.endswith(SYNC[:1]) else 0
                self._skip(len(self._buffer) - keep)
                return frames
            if start:
                self._skip(start)
            if len(self._buffer) < 6:
                return frames
            kind, seq, length = struct.unpack_from("<BBH", self._buffer, 2)
            if length > MAX_PAYLOAD:
                self._skip(1)
                continue
            total = 6 + length + 2
            if len(self._buffer) < total:
                return frames
            body = bytes(self._buffer[2 : 6 + length])
            (received,) = struct.unpack_from("<H", self._buffer, 6 + length)
            if received != crc16(body):
                self.crc_errors += 1
                self._skip(1)
                continue
            frames.append(Frame(kind, seq, bytes(self._buffer[6 : 6 + length])))
            del self._buffer[:total]


@dataclass(frozen=True)
class Hello:
    version: int
    fs: int
    block: int
    delay: int
    name: str


def decode_hello(frame: Frame) -> Hello | None:
    if frame.kind != TYPE_HELLO_REPLY or len(frame.payload) < 9:
        return None
    version, fs, block, delay = struct.unpack_from("<BHHH", frame.payload, 0)
    return Hello(version, fs, block, delay, frame.payload[7:].decode("ascii", errors="replace"))


def test_block(index: int) -> bytes:
    """One block of a quiet tone with noise, as 16-bit PCM."""
    rng = random.Random(index)
    values = []
    for i in range(BLOCK):
        t = (index * BLOCK + i) / 16000
        sample = 0.2 * math.sin(2 * math.pi * 440 * t) + 0.02 * rng.gauss(0, 1)
        values.append(max(-32768, min(32767, round(sample * 32768))))
    return struct.pack(f"<{BLOCK}h", *values)


# ---------------------------------------------------------------------------- links
class Link(Protocol):
    def write(self, data: bytes) -> None: ...

    def read(self, timeout: float) -> bytes:
        """Return the bytes that arrive within `timeout` seconds (empty if none)."""
        ...

    def close(self) -> None: ...


def import_serial() -> Any:
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial is missing. Run: py -m pip install pyserial") from None
    return serial


class SerialLink:
    def __init__(self, port: str, baud: int, dtr: bool, rts: bool) -> None:
        serial = import_serial()
        self._serial = serial.Serial()
        self._serial.port = port
        self._serial.baudrate = baud
        self._serial.timeout = 0
        self._serial.write_timeout = 2.0
        self._serial.dtr = dtr  # set before open so that opening does not pulse the lines
        self._serial.rts = rts
        self._serial.open()

    def write(self, data: bytes) -> None:
        self._serial.write(data)
        self._serial.flush()

    def read(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        while True:
            waiting = int(self._serial.in_waiting)
            if waiting:
                return bytes(self._serial.read(waiting))
            if time.monotonic() >= deadline:
                return b""
            time.sleep(0.002)

    def close(self) -> None:
        self._serial.close()


# ---------------------------------------------------------------------------- ports
@dataclass(frozen=True)
class PortInfo:
    device: str
    description: str
    vid: int | None
    pid: int | None


def list_ports() -> list[PortInfo]:
    import_serial()
    from serial.tools import list_ports as tools

    found = [PortInfo(p.device, p.description, p.vid, p.pid) for p in tools.comports()]
    return sorted(found, key=lambda p: (len(p.device), p.device))


def classify(port: PortInfo) -> str:
    if port.vid == ESPRESSIF_VID:
        return "Espressif USB device (the board's native USB port)"
    if port.vid is not None and port.pid is not None:
        return BRIDGES.get((port.vid, port.pid), "")
    return ""


def format_port(port: PortInfo) -> str:
    ids = f"{port.vid:04X}:{port.pid:04X}" if port.vid is not None and port.pid is not None else "-"
    note = classify(port)
    return f"  {port.device:<8} {port.description}  [USB {ids}]" + (f"  <- {note}" if note else "")


def show_ports(ports: Sequence[PortInfo]) -> None:
    if not ports:
        print("No serial ports found. Check the cable (some carry power only) and the USB port.")
        return
    print("Serial ports:")
    for port in ports:
        print(format_port(port))


def watch_ports(
    seconds: float,
    lister: Callable[[], list[PortInfo]] = list_ports,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Print every change of the port list. Press RST on the board while this runs."""
    known = {p.device: p for p in lister()}
    print("Watching. Plug the board in, unplug it, or press RST now.")
    show_ports(list(known.values()))
    elapsed = 0.0
    step = 0.25
    while elapsed < seconds:
        sleep(step)
        elapsed += step
        now = {p.device: p for p in lister()}
        for device in sorted(set(now) - set(known)):
            print(f"  +{elapsed:5.1f} s  appeared: {format_port(now[device]).strip()}")
        for device in sorted(set(known) - set(now)):
            print(f"  +{elapsed:5.1f} s  gone:     {format_port(known[device]).strip()}")
        known = now


def open_failure_hint(message: str) -> str:
    text = message.lower()
    if "access is denied" in text or "permissionerror" in text or "busy" in text:
        return (
            "Another program holds the port. Close the Arduino IDE (and its Serial Monitor), "
            "other browser tabs and any terminal program, then try again."
        )
    if "cannot find" in text or "no such file" in text or "filenotfound" in text:
        return (
            "The port is gone. The board probably restarted under another COM number. Run --list."
        )
    return "Unplug and replug the board, check the COM number with --list and try again."


# ---------------------------------------------------------------------------- probe
@dataclass
class Report:
    hello: Hello | None = None
    sent: int = 0
    received: int = 0
    lost: int = 0
    compute_us: list[int] = field(default_factory=list)
    rtt_ms: list[float] = field(default_factory=list)
    bad_status: int = 0
    stray: bytes = b""

    @property
    def ok(self) -> bool:
        return self.hello is not None and self.sent > 0 and self.lost == 0 and self.bad_status == 0


def handshake(link: Link, parser: Parser, seconds: float, start_seq: int = 0) -> Hello | None:
    deadline = time.monotonic() + seconds
    seq = start_seq
    while time.monotonic() < deadline:
        seq = (seq + 1) & 0xFF
        link.write(encode_frame(TYPE_HELLO, seq))
        until = time.monotonic() + 0.25
        while time.monotonic() < until:
            for frame in parser.push(link.read(max(0.0, until - time.monotonic()))):
                hello = decode_hello(frame)
                if hello is not None:
                    return hello
    return None


def stream(
    link: Link, parser: Parser, report: Report, blocks: int, depth: int, realtime: bool
) -> None:
    sent_at: dict[int, float] = {}
    seq = 100
    started = time.monotonic()
    while report.received + report.lost < blocks:
        now = time.monotonic()
        due = (now - started) / BLOCK_SECONDS if realtime else float(blocks)
        while report.sent < blocks and report.sent <= due and len(sent_at) < depth:
            seq = (seq + 1) & 0xFF
            link.write(encode_frame(TYPE_AUDIO_IN, seq, test_block(report.sent)))
            sent_at[seq] = time.monotonic()
            report.sent += 1
        data = link.read(0.005)
        now = time.monotonic()
        for frame in parser.push(data):
            if frame.kind != TYPE_AUDIO_OUT or len(frame.payload) != 4 + 2 * BLOCK:
                continue
            origin = sent_at.pop(frame.seq, None)
            if origin is None:
                continue
            compute_us, status = struct.unpack_from("<HH", frame.payload, 0)
            report.received += 1
            report.compute_us.append(compute_us)
            report.rtt_ms.append(1000 * (now - origin))
            report.bad_status += 1 if status else 0
        for key, origin in list(sent_at.items()):
            if now - origin > LOST_AFTER_S:
                del sent_at[key]
                report.lost += 1


def probe(link: Link, blocks: int = 200, wait: float = 5.0, realtime: bool = True) -> Report:
    parser = Parser()
    report = Report()
    report.hello = handshake(link, parser, wait)
    report.stray = bytes(parser.stray)
    if report.hello is None:
        return report
    link.write(encode_frame(TYPE_RESET, 200))
    time.sleep(0.05)
    parser.push(link.read(0.0))
    stream(link, parser, report, blocks, depth=4, realtime=realtime)
    return report


def _stats(values: Sequence[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    return mean, ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], ordered[-1]


def explain_silence(stray: bytes) -> list[str]:
    if stray:
        text = stray.decode("ascii", errors="replace").replace("\r", " ").replace("\n", " ")
        return [
            "The port sends data that is not HushWave frames. First bytes: " + text[:60],
            "The sketch is probably not running, or another program is on the board.",
            "Press RST and run this again. Check that the sketch was uploaded.",
        ]
    return [
        "Nothing was received. The port opened, but no firmware answers on it.",
        "If the cable is in the native USB port, Tools > USB CDC On Boot must be Enabled and",
        "USB Mode must be Hardware CDC and JTAG. Upload again after changing them.",
        "Then press RST and run this again.",
    ]


def print_report(report: Report) -> None:
    if report.hello is None:
        print("The board did not answer the hello.")
        for line in explain_silence(report.stray):
            print(line)
        print("result: FAIL")
        return
    hello = report.hello
    print(
        f"board:       {hello.name} (version {hello.version}, {hello.fs} Hz, "
        f"block {hello.block}, delay {hello.delay} samples)"
    )
    print(f"blocks:      {report.sent} sent, {report.received} answered, {report.lost} lost")
    if report.received:
        mean, _, peak = _stats([c / 1000 for c in report.compute_us])
        print(f"compute:     mean {mean:.2f} ms, max {peak:.2f} ms (real time needs under 16 ms)")
        mean, p95, peak = _stats(report.rtt_ms)
        print(f"round trip:  mean {mean:.2f} ms, p95 {p95:.2f} ms, max {peak:.2f} ms")
    if report.bad_status:
        print(f"status:      {report.bad_status} blocks came back with a non-zero status")
    print("result: " + ("PASS" if report.ok else "FAIL"))


# ------------------------------------------------------------------------------ cli
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the HushWave board over USB serial.")
    parser.add_argument("--list", action="store_true", help="list the serial ports and stop")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="watch the port list")
    parser.add_argument("--port", help="port name, for example COM10")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--blocks", type=int, default=200, help="test blocks to stream")
    parser.add_argument("--wait", type=float, default=5.0, help="seconds to wait for the hello")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds to wait after opening")
    parser.add_argument("--dtr", choices=("on", "off"), default="off")
    parser.add_argument("--rts", choices=("on", "off"), default="off")
    parser.add_argument(
        "--fast", action="store_true", help="send blocks as fast as they are answered"
    )
    return parser.parse_args(argv)


def choose_port(ports: Sequence[PortInfo]) -> str | None:
    espressif = [p for p in ports if p.vid == ESPRESSIF_VID]
    if len(espressif) == 1:
        return espressif[0].device
    return None


def open_link(port: str, baud: int, dtr: bool, rts: bool) -> Link:
    return SerialLink(port, baud, dtr, rts)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.watch is not None:
        watch_ports(args.watch)
        return 0
    port = args.port
    if args.list or not port:
        ports = list_ports()
        show_ports(ports)
        port = None if args.list else choose_port(ports)
        if port is None:
            if not args.list:
                print("Choose one with --port, for example: py mcu_probe.py --port COM10")
            return 0
        print(f"Using {port}.")
    try:
        link = open_link(port, args.baud, args.dtr == "on", args.rts == "on")
    except Exception as error:  # pyserial raises SerialException, PermissionError, OSError
        print(f"Could not open {port}: {error}")
        print(open_failure_hint(f"{type(error).__name__} {error}"))
        print("result: FAIL")
        return 2
    try:
        print(f"Opened {port} at {args.baud} baud. Waiting {args.settle:.1f} s.")
        time.sleep(args.settle)
        report = probe(link, args.blocks, args.wait, realtime=not args.fast)
    finally:
        link.close()
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
