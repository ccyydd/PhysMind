from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from run_video_metric_depth import DEFAULT_VIDEO_METRIC_DEPTH_MODEL
from run_video_metric_depth import load_video_metric_depth_model, run_video_metric_depth_with_model


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def serve(*, request_dir: Path, ready_file: Path, model_dir: str, batch_size: int) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    startup_start = time.perf_counter()
    model, device = load_video_metric_depth_model(model_dir=model_dir)
    ready_file.write_text(
        json.dumps(
            {
                "status": "ok",
                "tool": "video_metric_depth_worker",
                "model_dir": model_dir,
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
            request_type = request.get("type")
            if request_type == "shutdown":
                _write_response(response_path, {"status": "ok", "request_id": request.get("request_id")})
                request_path.unlink(missing_ok=True)
                return

            task_start = time.perf_counter()
            previous_debug_artifacts = os.environ.get("PHYSMIND_DEBUG_ARTIFACTS")
            os.environ["PHYSMIND_DEBUG_ARTIFACTS"] = "1" if request.get("debug_artifacts") else "0"
            try:
                run_video_metric_depth_with_model(
                    model=model,
                    device=device,
                    video=Path(request["video"]),
                    object_plan=Path(request["object_plan"]),
                    output=Path(request["output"]),
                    model_dir=str(request.get("model_dir") or model_dir),
                    batch_size=int(request.get("batch_size") or batch_size),
                    execution_mode="persistent_worker",
                )
            finally:
                if previous_debug_artifacts is None:
                    os.environ.pop("PHYSMIND_DEBUG_ARTIFACTS", None)
                else:
                    os.environ["PHYSMIND_DEBUG_ARTIFACTS"] = previous_debug_artifacts
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
    parser.add_argument("--model-dir", default=DEFAULT_VIDEO_METRIC_DEPTH_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    serve(
        request_dir=Path(args.request_dir),
        ready_file=Path(args.ready_file),
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
