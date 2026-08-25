from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from run_sam3_video_tracks import _load_model, run_sam3_video_tracks


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def serve(
    *,
    request_dir: Path,
    ready_file: Path,
    checkpoint: str | None,
    sam3_version: str,
    compile_model: bool,
    max_num_objects: int,
    async_loading_frames: bool,
) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    startup_start = time.perf_counter()
    model, use_fa3 = _load_model(
        checkpoint=checkpoint,
        sam3_version=sam3_version,
        compile_model=compile_model,
        max_num_objects=max_num_objects,
        async_loading_frames=async_loading_frames,
    )
    ready_file.write_text(
        json.dumps(
            {
                "status": "ok",
                "tool": "sam3_video_tracks_worker",
                "checkpoint": checkpoint,
                "sam3_version": sam3_version,
                "compile": bool(compile_model),
                "max_num_objects": int(max_num_objects),
                "async_loading_frames": bool(async_loading_frames),
                "use_fa3": bool(use_fa3),
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
                run_sam3_video_tracks(
                    video=Path(request["video"]),
                    object_plan=Path(request["object_plan"]),
                    output=Path(request["output"]),
                    checkpoint=str(request.get("checkpoint") or checkpoint) if request.get("checkpoint") or checkpoint else None,
                    sam3_version=str(request.get("sam3_version") or sam3_version),
                    mode=str(request.get("mode") or "auto"),
                    bench=request.get("bench"),
                    generic_prompt=str(
                        request.get("generic_prompt")
                        or "all movable objects on the floor, including balls, cylinders, and cubes"
                    ),
                    prompt_frame_index=int(request.get("prompt_frame_index") or 0),
                    compile_model=bool(request.get("compile_model", compile_model)),
                    max_num_objects=int(request.get("max_num_objects") or max_num_objects),
                    async_loading_frames=bool(request.get("async_loading_frames", async_loading_frames)),
                    propagation_direction=str(request.get("propagation_direction") or "forward"),
                    max_frame_num_to_track=(
                        int(request["max_frame_num_to_track"])
                        if request.get("max_frame_num_to_track") is not None
                        else None
                    ),
                    model=model,
                    use_fa3=use_fa3,
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sam3-version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-num-objects", type=int, default=8)
    parser.add_argument("--async-loading-frames", action="store_true")
    args = parser.parse_args()
    serve(
        request_dir=Path(args.request_dir),
        ready_file=Path(args.ready_file),
        checkpoint=args.checkpoint,
        sam3_version=args.sam3_version,
        compile_model=args.compile,
        max_num_objects=args.max_num_objects,
        async_loading_frames=args.async_loading_frames,
    )


if __name__ == "__main__":
    main()
