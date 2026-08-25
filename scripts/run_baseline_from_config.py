from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"LLM config must be a JSON object: {path}")
    if not payload.get("provider") or not payload.get("model"):
        raise ValueError(f"LLM config must contain provider and model: {path}")
    return payload


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _build_physmind_args(command: dict[str, Any]) -> list[str]:
    args = ["python", "-u", "physmind.py"]
    for key, value in command.items():
        if value is None or value is False:
            continue
        flag = _flag_name(str(key))
        if value is True:
            args.append(flag)
            continue
        args.extend([flag, str(value)])
    return args


def _extra_env(config: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    openrouter = config.get("openrouter")
    if isinstance(openrouter, dict):
        env["PHYSMIND_OPENROUTER_EXTRA_BODY"] = json.dumps(openrouter, ensure_ascii=False)
    return env


def _shell_join_env(env: dict[str, str]) -> str:
    if not env:
        return ""
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())) + " "


def _build_foreground_command(command: dict[str, Any], conda_env: str) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        *_build_physmind_args(command),
    ]


def _build_tmux_shell_command(config: dict[str, Any], command: dict[str, Any], conda_env: str, log_path: Path) -> str:
    physmind_args = " ".join(shlex.quote(item) for item in _build_physmind_args(command))
    env_prefix = _shell_join_env(_extra_env(config))
    return (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"{env_prefix}conda run --no-capture-output -n {shlex.quote(conda_env)} "
        f"{physmind_args} 2>&1 | tee {shlex.quote(str(log_path))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhysMind with an LLM API config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--mode", default="direct-answer")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--answer-format", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--scene-ids", default=None)
    parser.add_argument("--question-types", default=None)
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--resume-rerun-statuses", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--conda-env", default="physmind")
    parser.add_argument("--tmux", action="store_true", help="Launch the baseline in a detached tmux session.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("--session", default=None, help="Override the tmux session name.")
    parser.add_argument("--log-path", type=Path, default=None, help="Override the tmux log path.")
    args = parser.parse_args()

    config = _load_config(args.config)
    command: dict[str, Any] = {
        "bench": args.bench,
        "mode": args.mode,
        "provider": config["provider"],
        "model": config["model"],
        "dataset_root": args.dataset_root or config.get("dataset_root"),
        "limit": args.limit,
        "num_workers": args.num_workers,
        "scenario": args.scenario,
        "scene_ids": args.scene_ids,
        "question_types": args.question_types,
        "resume_run_dir": args.resume_run_dir,
        "resume_rerun_statuses": args.resume_rerun_statuses,
        "run_dir": args.run_dir,
    }
    num_frames = args.num_frames if args.num_frames is not None else config.get("num_frames")
    if num_frames is not None:
        command["num_frames"] = int(num_frames)
    max_output_tokens = (
        args.max_output_tokens
        if args.max_output_tokens is not None
        else config.get("max_output_tokens")
    )
    if max_output_tokens is not None:
        command["max_output_tokens"] = int(max_output_tokens)
    answer_format = args.answer_format if args.answer_format is not None else config.get("answer_format")
    if answer_format is not None:
        command["answer_format"] = str(answer_format)
    env = os.environ.copy()
    env.update(_extra_env(config))

    if args.tmux:
        session = args.session or f"{args.bench}_{args.mode}_{str(config['model']).replace('/', '-')}"
        log_path = args.log_path or Path(f"runs/{session}.log")
        log_path = log_path if log_path.is_absolute() else REPO_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        shell_command = _build_tmux_shell_command(config, command, args.conda_env, log_path)
        tmux_command = ["tmux", "new-session", "-d", "-s", session, shell_command]
        print(" ".join(shlex.quote(item) for item in tmux_command))
        if not args.dry_run:
            subprocess.run(tmux_command, cwd=REPO_ROOT, check=True)
        return

    command_list = _build_foreground_command(command, args.conda_env)
    print(_shell_join_env(_extra_env(config)) + " ".join(shlex.quote(item) for item in command_list))
    if not args.dry_run:
        subprocess.run(command_list, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
