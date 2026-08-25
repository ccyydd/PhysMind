from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .run_sam3d_objects import SAM3D_ROOT, load_sam3d_inference, run_sam3d_objects


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def serve(
    *,
    request_dir: Path,
    ready_file: Path,
    config_path: Path,
    seed: int,
    compile_model: bool,
) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    startup_start = time.perf_counter()
    inference = load_sam3d_inference(config_path=config_path, compile_model=compile_model)
    ready_file.write_text(
        json.dumps(
            {
                "status": "ok",
                "tool": "sam3d_worker",
                "config_path": str(config_path),
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
            run_sam3d_objects(
                video=Path(request["video"]),
                object_plan=Path(request["object_plan"]),
                output=Path(request["output"]),
                config_path=Path(request.get("config_path") or config_path),
                seed=int(request.get("seed") or seed),
                compile_model=bool(request.get("compile_model") or compile_model),
                preloaded_inference=inference,
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
    parser.add_argument("--config-path", default=str(SAM3D_ROOT / "checkpoints" / "hf" / "pipeline.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    serve(
        request_dir=Path(args.request_dir),
        ready_file=Path(args.ready_file),
        config_path=Path(args.config_path),
        seed=args.seed,
        compile_model=args.compile,
    )


if __name__ == "__main__":
    main()
