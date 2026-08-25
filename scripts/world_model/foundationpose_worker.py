from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_foundationpose import load_foundationpose_context, run_foundationpose


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def serve(
    *,
    request_dir: Path,
    ready_file: Path,
    est_refine_iter: int,
    track_refine_iter: int,
    debug: int,
) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    startup_start = time.perf_counter()
    context = load_foundationpose_context()
    ready_file.write_text(
        json.dumps(
            {
                "status": "ok",
                "tool": "foundationpose_worker",
                "startup_elapsed_sec": time.perf_counter() - startup_start,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    while True:
        requests = sorted(request_dir.glob("*.request.json"))
        if not requests:
            time.sleep(0.05)
            continue
        request_path = requests[0]
        response_path = request_path.with_name(request_path.name.replace(".request.json", ".response.json"))
        request: dict[str, Any] = {}
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if request.get("type") == "shutdown":
                _write_response(response_path, {"status": "ok", "request_id": request.get("request_id")})
                request_path.unlink(missing_ok=True)
                return

            task_start = time.perf_counter()
            output_path = Path(request["output"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            run_foundationpose(
                video=Path(request["video"]),
                object_plan=Path(request["object_plan"]),
                output=output_path,
                est_refine_iter=int(request.get("est_refine_iter") or est_refine_iter),
                track_refine_iter=int(request.get("track_refine_iter") or track_refine_iter),
                debug=int(request.get("debug") or debug),
                debug_artifacts=int(request.get("debug_artifacts") or 0),
                context=context,
                execution_mode="persistent_worker",
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            _write_response(
                response_path,
                {
                    "status": "ok",
                    "request_id": request.get("request_id"),
                    "artifact_path": request["output"],
                    "elapsed_sec": time.perf_counter() - task_start,
                    "payload": payload,
                },
            )
        except Exception as exc:
            _write_response(
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
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    serve(
        request_dir=Path(args.request_dir),
        ready_file=Path(args.ready_file),
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
