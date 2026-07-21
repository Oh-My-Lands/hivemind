#!/usr/bin/env python3
"""
Local HTTP endpoint wrapping the serverless handler.

The point is dev/prod parity: the frontend talks to the *same* handler function
with the *same* request and response shape it will hit on RunPod, so the two
cannot drift. Run it on a GPU pod during development, point the frontend at it,
and deploying changes only the URL.

    cd /workspace/hivemind/engine
    python3 deploy/runpod/dev_server.py --port 8080

    curl -s localhost:8080/run -H 'content-type: application/json' -d '{
      "input": {"fen": "<fenA>|<fenB>", "movetime": 3000,
                "multipv": 5, "analysisBoard": 1}
    }' | python3 -m json.tool

This is a development tool: single-threaded, no auth, no TLS. Do not put it on
a public port. Serving in production is RunPod's job.
"""

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import handler as handler_module  # noqa: E402

LOG = logging.getLogger("dev_server")

MAX_BODY_BYTES = 1 << 20  # a position request is tiny; anything larger is wrong


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The frontend runs on a different origin during development.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._respond(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/run", "/runsync"):
            self._respond(404, {"error": "not found; POST to /run"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._respond(400, {"error": "bad Content-Length"})
            return

        if length <= 0 or length > MAX_BODY_BYTES:
            self._respond(400, {"error": "missing or oversized body"})
            return

        try:
            job = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._respond(400, {"error": f"invalid JSON: {exc}"})
            return

        if "input" not in job:
            # Accept a bare input object too, since it is an easy mistake and
            # the intent is unambiguous.
            job = {"input": job}

        try:
            result = handler_module.handler_sync(job)
        except Exception as exc:  # handler catches most things itself
            LOG.exception("handler failed")
            self._respond(500, {"error": str(exc)})
            return

        # RunPod wraps handler output in {"output": ...}; mirror that so client
        # code written against the deployed endpoint works unchanged here.
        self._respond(200, {"output": result, "status": "COMPLETED"})

    def log_message(self, fmt: str, *args) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="default is loopback; bind 0.0.0.0 only behind a tunnel")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not os.path.isdir("networks"):
        LOG.warning(
            "no ./networks in %s -- the engine looks for it relative to its "
            "working directory and will fail without it", os.getcwd()
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOG.info("listening on http://%s:%d  (POST /run)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
