from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_geocalib_gravity import DEFAULT_GEOCALIB_WEIGHTS
from run_geocalib_gravity import load_geocalib_model, run_geocalib_gravity_with_model


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def serve(*, request_dir: Path, ready_file: Path, weights: str, num_frames: int) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    startup_start = time.perf_counter()
    model, device = load_geocalib_model(weights=weights)
    ready_file.write_text(
        json.dumps(
            {
                "status": "ok",
                "tool": "geocalib_worker",
                "weights": weights,
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
            run_geocalib_gravity_with_model(
                model=model,
                device=device,
                video=Path(request["video"]),
                output=Path(request["output"]),
                weights=str(request.get("weights") or weights),
                num_frames=int(request.get("num_frames") or num_frames),
                execution_mode="persistent_worker",
                frame_start=request.get("frame_start"),
                frame_end=request.get("frame_end"),
                gravity_z_constraint=request.get("gravity_z_constraint"),
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
    parser.add_argument("--weights", default=DEFAULT_GEOCALIB_WEIGHTS)
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()
    serve(
        request_dir=Path(args.request_dir),
        ready_file=Path(args.ready_file),
        weights=args.weights,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
