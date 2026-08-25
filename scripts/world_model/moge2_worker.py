from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_moge2_intrinsics import (
    DEFAULT_MOGE2_MODEL,
    MOGE2_SAMPLE_COUNT,
    load_moge2_model,
    run_moge2_intrinsics_with_model,
)


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def serve(*, request_dir: Path, ready_file: Path, model_name: str, num_frames: int) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    startup_start = time.perf_counter()
    model, device = load_moge2_model(model_name=model_name)
    ready_file.write_text(
        json.dumps(
            {
                "status": "ok",
                "tool": "moge2_worker",
                "model_name": model_name,
                "device": device,
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
            requested_model = str(request.get("model_name") or model_name)
            if requested_model != model_name:
                raise ValueError(
                    f"Persistent MoGe-2 worker loaded {model_name!r}; received request for {requested_model!r}."
                )

            task_start = time.perf_counter()
            run_moge2_intrinsics_with_model(
                model=model,
                device=device,
                video=Path(request["video"]),
                output=Path(request["output"]),
                model_name=model_name,
                num_frames=int(request.get("num_frames") or num_frames),
                execution_mode="persistent_worker",
            )
            payload = json.loads(Path(request["output"]).read_text(encoding="utf-8"))
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
    parser.add_argument("--model-name", default=DEFAULT_MOGE2_MODEL)
    parser.add_argument("--num-frames", type=int, default=MOGE2_SAMPLE_COUNT)
    args = parser.parse_args()
    serve(
        request_dir=Path(args.request_dir),
        ready_file=Path(args.ready_file),
        model_name=args.model_name,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
