"""Start the HushWave demo interface: ``demo.py replay`` or ``demo.py live``."""

from __future__ import annotations

import argparse
import sys
import webbrowser

from hushwave.demo_server import default_app, make_server


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "live"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        app = default_app()
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.mode == "replay" and not (app.examples_root / "manifest.json").is_file():
        print("error: examples/manifest.json is missing; run: make examples", file=sys.stderr)
        return 1
    try:
        server = make_server(app, args.host, args.port)
    except OSError as error:
        print(f"error: cannot listen on {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1
    url = f"http://localhost:{args.port}/#{args.mode}"
    print(f"HushWave demo at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
