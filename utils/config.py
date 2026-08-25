from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass
class ModelConfig:
    provider: str = "openrouter"
    model: str = "google/gemini-3-flash-preview"
    api_key: Optional[str] = None
    num_frames: int = 8
    max_output_tokens: int = 4096
    max_retries: int = 2
    request_timeout: float = 180.0
    openrouter_extra_body: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_safe_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.pop("api_key", None)
        return payload


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_api_key(provider: str, api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key

    provider_name = (provider or "openai").lower()
    if provider_name == "openrouter":
        env_name = "OPEN_ROUTER_KEY"
    elif provider_name == "closeai":
        env_name = "CLOSEAI_API_KEY"
    elif provider_name == "google":
        env_name = "GEMINI_API_KEY"
    else:
        env_name = "OPENAI_API_KEY"

    value = os.getenv(env_name)
    if not value:
        raise ValueError(
            f"API key not provided for provider '{provider_name}'. "
            f"Set {env_name} or pass --api-key."
        )
    return value


def default_model_for_provider(provider: str) -> str:
    provider_name = (provider or "openai").lower()
    if provider_name == "openrouter":
        return "google/gemini-3-flash-preview"
    if provider_name == "google":
        return "gemini-2.5-flash"
    return "gpt-5"


def build_model_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    num_frames: int = 8,
    max_output_tokens: int = 4096,
    max_retries: int = 2,
    request_timeout: float = 180.0,
    require_api_key: bool = True,
) -> ModelConfig:
    load_project_env()
    resolved_provider = (provider or os.getenv("PHYSMIND_PROVIDER", "openrouter")).lower()
    resolved_model = model or os.getenv("PHYSMIND_MODEL") or default_model_for_provider(resolved_provider)
    resolved_key = resolve_api_key(resolved_provider, api_key=api_key) if require_api_key else api_key
    openrouter_extra_body = None
    raw_extra_body = os.getenv("PHYSMIND_OPENROUTER_EXTRA_BODY")
    if raw_extra_body:
        import json

        openrouter_extra_body = json.loads(raw_extra_body)
        if not isinstance(openrouter_extra_body, dict):
            raise ValueError("PHYSMIND_OPENROUTER_EXTRA_BODY must decode to a JSON object.")
    return ModelConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_key,
        num_frames=num_frames,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        request_timeout=request_timeout,
        openrouter_extra_body=openrouter_extra_body,
    )
