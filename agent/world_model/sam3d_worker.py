from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_SAM3D_WORKER_CMD = "python -m scripts.world_model.sam3d_worker"


def _external_env() -> Dict[str, str]:
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(project_root)
        if not existing_pythonpath
        else f"{project_root}{os.pathsep}{existing_pythonpath}"
    )
    return env


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


class SAM3DWorkerClient:
    def __init__(self, run_dir: Path, *, startup_timeout_sec: float = 1800.0, request_timeout_sec: float = 7200.0):
        self.run_dir = run_dir
        self.worker_dir = run_dir / "workers" / "sam3d"
        self.request_dir = self.worker_dir / "requests"
        self.stdout_path = self.worker_dir / "stdout.log"
        self.stderr_path = self.worker_dir / "stderr.log"
        self.ready_path = self.worker_dir / "ready.json"
        self.startup_timeout_sec = startup_timeout_sec
        self.request_timeout_sec = request_timeout_sec
        self.process: Optional[subprocess.Popen[str]] = None
        self._stdout_handle: Any = None
        self._stderr_handle: Any = None
        self.startup_info: Dict[str, Any] = {}

    def start(self) -> Dict[str, Any]:
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.request_dir.mkdir(parents=True, exist_ok=True)
        for path in self.request_dir.glob("*.json"):
            path.unlink()
        if self.ready_path.exists():
            self.ready_path.unlink()

        command = os.getenv("PHYSMIND_SAM3D_WORKER_CMD", DEFAULT_SAM3D_WORKER_CMD)
        args = shlex.split(command) + ["--request-dir", str(self.request_dir), "--ready-file", str(self.ready_path)]
        self._stdout_handle = self.stdout_path.open("w", encoding="utf-8")
        self._stderr_handle = self.stderr_path.open("w", encoding="utf-8")
        start = time.perf_counter()
        self.process = subprocess.Popen(
            args,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=_external_env(),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
        )
        while True:
            if self.ready_path.exists():
                self.startup_info = json.loads(self.ready_path.read_text(encoding="utf-8"))
                self.startup_info.setdefault("startup_elapsed_sec", time.perf_counter() - start)
                self.startup_info["command"] = command
                self.startup_info["stdout_log"] = str(self.stdout_path)
                self.startup_info["stderr_log"] = str(self.stderr_path)
                return self.startup_info
            if self.process.poll() is not None:
                raise RuntimeError(
                    "SAM3D worker exited before ready. "
                    f"stdout_tail={_tail(self.stdout_path)!r} stderr_tail={_tail(self.stderr_path)!r}"
                )
            if time.perf_counter() - start > self.startup_timeout_sec:
                self.close(kill=True)
                raise TimeoutError(f"SAM3D worker did not become ready within {self.startup_timeout_sec:.1f}s")
            time.sleep(0.25)

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("SAM3D worker is not running.")
        request_id = uuid.uuid4().hex
        request_path = self.request_dir / f"{request_id}.request.json"
        response_path = self.request_dir / f"{request_id}.response.json"
        tmp_path = self.request_dir / f"{request_id}.request.tmp"
        payload = dict(payload)
        payload["request_id"] = request_id
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.rename(request_path)

        start = time.perf_counter()
        while True:
            if response_path.exists():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                response_path.unlink()
                return response
            if self.process.poll() is not None:
                raise RuntimeError(
                    "SAM3D worker exited while processing request. "
                    f"stdout_tail={_tail(self.stdout_path)!r} stderr_tail={_tail(self.stderr_path)!r}"
                )
            if time.perf_counter() - start > self.request_timeout_sec:
                raise TimeoutError(f"SAM3D worker request timed out after {self.request_timeout_sec:.1f}s")
            time.sleep(0.1)

    def close(self, *, kill: bool = False) -> None:
        if self.process is not None and self.process.poll() is None:
            if kill:
                self.process.kill()
            else:
                try:
                    self.request({"type": "shutdown"})
                    self.process.wait(timeout=30)
                except Exception:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
        if self._stdout_handle is not None:
            self._stdout_handle.close()
            self._stdout_handle = None
        if self._stderr_handle is not None:
            self._stderr_handle.close()
            self._stderr_handle = None
