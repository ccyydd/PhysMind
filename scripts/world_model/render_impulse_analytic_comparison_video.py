from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SYSID_PATH = SCRIPT_DIR / "run_impulse_analytic_sysid.py"
SYSID_SPEC = importlib.util.spec_from_file_location(
    "physmind_run_impulse_analytic_sysid_for_video",
    SYSID_PATH,
)
if SYSID_SPEC is None or SYSID_SPEC.loader is None:
    raise RuntimeError(f"failed to import {SYSID_PATH}")
sysid = importlib.util.module_from_spec(SYSID_SPEC)
SYSID_SPEC.loader.exec_module(sysid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fit", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()
    sysid.render_comparison_video_from_result_json(
        input_fit=Path(args.input_fit),
        result_json=Path(args.result_json),
        output_mp4=Path(args.output_mp4),
        width=int(args.width),
        height=int(args.height),
        fps_override=args.fps,
    )


if __name__ == "__main__":
    main()
