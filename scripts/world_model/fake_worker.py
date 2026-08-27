from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def serve(*, request_dir: Path, ready_file: Path, tool_name: str) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        ready_file,
        {
            "status": "ok",
            "tool": tool_name,
            "startup_elapsed_sec": 0.0,
        },
    )

    while True:
        requests = sorted(request_dir.glob("*.request.json"))
        if not requests:
            time.sleep(0.02)
            continue
        request_path = requests[0]
        response_path = request_path.with_name(request_path.name.replace(".request.json", ".response.json"))
        request: dict[str, Any] = {}
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if request.get("type") == "shutdown":
                _write_json(response_path, {"status": "ok", "request_id": request.get("request_id")})
                request_path.unlink(missing_ok=True)
                return
            if request.get("force_error"):
                raise RuntimeError("forced fake worker error")
            artifact_path = Path(request["output"])
            payload = {
                "tool": request.get("task_type") or tool_name,
                "status": "ok",
                "execution_mode": "fake_worker",
                "request_id": request.get("request_id"),
            }
            _write_json(artifact_path, payload)
            _write_json(
                response_path,
                {
                    "status": "ok",
                    "request_id": request.get("request_id"),
                    "artifact_path": str(artifact_path),
                    "elapsed_sec": 0.0,
                    "payload": payload,
                },
            )
        except Exception as exc:
            _write_json(
                response_path,
                {
                    "status": "error",
                    "request_id": request.get("request_id"),
                    "message": str(exc),
                },
            )
        finally:
            request_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-dir", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--tool-name", default="fake_worker")
    args = parser.parse_args()
    serve(request_dir=Path(args.request_dir), ready_file=Path(args.ready_file), tool_name=args.tool_name)


if __name__ == "__main__":
    main()
