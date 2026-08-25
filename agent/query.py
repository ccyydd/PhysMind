from __future__ import annotations

import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from openai import OpenAI

from utils.config import ModelConfig
from utils.video import (
    encode_image_bytes_as_data_url,
    get_cached_video_data_url,
    sample_uniform_frames,
    sample_uniform_frames_with_indices,
    sample_video_frames_at_indices,
)


ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


@dataclass
class ModelResponse:
    text: str
    usage: Optional[dict[str, Optional[int]]] = None


def create_openai_compatible_client(config: ModelConfig) -> OpenAI:
    provider = config.provider.lower()
    if provider == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.api_key,
            timeout=config.request_timeout,
        )
    if provider == "closeai":
        return OpenAI(
            base_url="https://api.openai-proxy.org/v1",
            api_key=config.api_key,
            timeout=config.request_timeout,
        )
    if provider == "openai":
        return OpenAI(api_key=config.api_key, timeout=config.request_timeout)
    raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")


def extract_answer_tag(text: str) -> Optional[str]:
    match = ANSWER_TAG_PATTERN.search(text or "")
    if not match:
        return None
    return match.group(1).strip()


def infer_model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if "gemini" in name:
        return "gemini"
    return "gpt"


def _format_request_context(request_context: Optional[dict[str, Any]]) -> str:
    if not request_context:
        return ""
    fields = []
    for key in ("scene_index", "question_id", "question_type"):
        value = request_context.get(key)
        if value is not None:
            fields.append(f"{key}={value}")
    return (" " + " ".join(fields)) if fields else ""


def _truncate_text(text: str, limit: int = 800) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _should_log_model_output() -> bool:
    return (os.getenv("PHYSMIND_LOG_MODEL_OUTPUT") or "").strip().lower() in {"1", "true", "yes", "on"}


def _log_model_output(text: str, request_context: Optional[dict[str, Any]] = None) -> None:
    if _should_log_model_output():
        print(f"[output]{_format_request_context(request_context)} {_truncate_text(text)}")
        return
    print(f"[output]{_format_request_context(request_context)} chars={len((text or '').strip())}")


def _log_openai_response(response: Any, request_context: Optional[dict[str, Any]] = None) -> None:
    metadata = []
    for attr in ("id", "model"):
        value = getattr(response, attr, None)
        if value:
            metadata.append(f"{attr}={value}")

    choices = getattr(response, "choices", None) or []
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason:
            metadata.append(f"finish_reason={finish_reason}")

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is not None:
            metadata.append(f"prompt_tokens={prompt_tokens}")
        if completion_tokens is not None:
            metadata.append(f"completion_tokens={completion_tokens}")
        if total_tokens is not None:
            metadata.append(f"total_tokens={total_tokens}")

    print(f"[response]{_format_request_context(request_context)} {' '.join(metadata)}".rstrip())


def _extract_openai_usage(response: Any) -> Optional[dict[str, Optional[int]]]:
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _openai_extra_kwargs(config: ModelConfig) -> dict[str, Any]:
    if config.provider.lower() == "openrouter" and config.openrouter_extra_body:
        return {"extra_body": config.openrouter_extra_body}
    return {}


def answer_with_gpt_video_frames(
    *,
    config: ModelConfig,
    prompt: str,
    video_path: Union[str, Path],
    request_context: Optional[dict[str, Any]] = None,
    continuation_messages: Optional[list[dict[str, Any]]] = None,
    additional_reference_frame_indices: Optional[Sequence[int]] = None,
) -> str:
    print(
        f"[request]{_format_request_context(request_context)} provider={config.provider} "
        f"model={config.model} mode=gpt-frames video={Path(video_path).name}"
    )
    client = create_openai_compatible_client(config)
    if additional_reference_frame_indices:
        uniform_frames = sample_uniform_frames_with_indices(
            video_path=video_path,
            num_frames=config.num_frames,
        )
        reference_frames = sample_video_frames_at_indices(
            video_path=video_path,
            frame_indices=additional_reference_frame_indices,
        )
        print(
            f"[request]{_format_request_context(request_context)} "
            f"uniform_frame_indices={[item.frame_index for item in uniform_frames]} "
            f"additional_reference_frame_indices={[item.frame_index for item in reference_frames]} "
            f"timeout={config.request_timeout}s"
        )
        frame_instruction = PHYSION_PP_FRAME_INPUT_INSTRUCTION.format(
            num_uniform_frames=len(uniform_frames)
        )
        content = [
            {
                "type": "text",
                "text": f"{frame_instruction}\n\n{prompt}",
            }
        ]
        for index, frame in enumerate(uniform_frames, start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Uniform sample {index}/{len(uniform_frames)} "
                        f"(original frame index {frame.frame_index})"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_bytes_as_data_url(frame.image_bytes)},
                }
            )
        for frame in reference_frames:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Additional target cue reference — not chronological "
                        f"(original frame index {frame.frame_index})"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_bytes_as_data_url(frame.image_bytes)},
                }
            )
    else:
        frame_bytes = sample_uniform_frames(video_path=video_path, num_frames=config.num_frames)
        print(
            f"[request]{_format_request_context(request_context)} "
            f"sampled_frames={len(frame_bytes)} timeout={config.request_timeout}s"
        )
        content = [
            {
                "type": "text",
                "text": (
                    "These images are uniformly sampled frames from a video. "
                    "Use the temporal sequence across frames to answer the question.\n\n"
                    f"{prompt}"
                ),
            }
        ]
        for frame in frame_bytes:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_bytes_as_data_url(frame)},
                }
            )

    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if continuation_messages:
        messages.extend(continuation_messages)

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        max_completion_tokens=config.max_output_tokens,
        **_openai_extra_kwargs(config),
    )
    _log_openai_response(response, request_context=request_context)
    text = (response.choices[0].message.content or "").strip()
    _log_model_output(text, request_context=request_context)
    return ModelResponse(text=text, usage=_extract_openai_usage(response))


def answer_with_sampled_video_frames(
    *,
    config: ModelConfig,
    prompt: str,
    video_path: Union[str, Path],
    request_context: Optional[dict[str, Any]] = None,
    frame_intro: Optional[str] = None,
    continuation_messages: Optional[list[dict[str, Any]]] = None,
) -> ModelResponse:
    print(
        f"[request]{_format_request_context(request_context)} provider={config.provider} "
        f"model={config.model} mode=sampled-frames video={Path(video_path).name}"
    )
    sampled_frames = sample_uniform_frames_with_indices(video_path=video_path, num_frames=config.num_frames)
    frame_indices = [item.frame_index for item in sampled_frames]
    print(
        f"[request]{_format_request_context(request_context)} sampled_frame_indices={frame_indices} "
        f"timeout={config.request_timeout}s"
    )
    intro = frame_intro or (
        "These images are uniformly sampled candidate frames from a video. "
        "Each image is labeled by its original zero-based frame index."
    )
    text_prompt = f"{intro}\nCandidate frame indices: {frame_indices}\n\n{prompt}"
    if config.provider == "google":
        from PIL import Image
        import google.generativeai as genai

        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model)
        content: list[Any] = [text_prompt]
        for item in sampled_frames:
            content.append(f"Frame index: {item.frame_index}")
            content.append(Image.open(BytesIO(item.image_bytes)))
        if continuation_messages:
            chat = model.start_chat(history=[{"role": "user", "parts": content}])
            for message in continuation_messages[:-1]:
                role = "model" if message.get("role") == "assistant" else "user"
                text = str(message.get("content") or "")
                if text:
                    chat.history.append({"role": role, "parts": [text]})
            final_message = continuation_messages[-1]
            final_text = str(final_message.get("content") or "")
            response = chat.send_message(
                final_text,
                generation_config={"max_output_tokens": config.max_output_tokens},
                stream=False,
            )
        else:
            response = model.generate_content(
                content,
                generation_config={"max_output_tokens": config.max_output_tokens},
                stream=False,
            )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response.")
        output = text.strip()
        print(f"[response]{_format_request_context(request_context)} provider=google")
        _log_model_output(output, request_context=request_context)
        return ModelResponse(text=output, usage=None)

    client = create_openai_compatible_client(config)
    content = [
        {
            "type": "text",
            "text": text_prompt,
        }
    ]
    for item in sampled_frames:
        content.append(
            {
                "type": "text",
                "text": f"Frame index: {item.frame_index}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_bytes_as_data_url(item.image_bytes)},
            }
        )

    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if continuation_messages:
        messages.extend(continuation_messages)

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        max_completion_tokens=config.max_output_tokens,
        **_openai_extra_kwargs(config),
    )
    _log_openai_response(response, request_context=request_context)
    text = (response.choices[0].message.content or "").strip()
    _log_model_output(text, request_context=request_context)
    return ModelResponse(text=text, usage=_extract_openai_usage(response))


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/jpeg"


def answer_with_image_files(
    *,
    config: ModelConfig,
    prompt: str,
    image_paths: list[Union[str, Path]],
    image_labels: Optional[list[str]] = None,
    request_context: Optional[dict[str, Any]] = None,
) -> ModelResponse:
    paths = [Path(path) for path in image_paths]
    labels = image_labels or [path.name for path in paths]
    if len(labels) != len(paths):
        raise ValueError("image_labels must have the same length as image_paths.")
    print(
        f"[request]{_format_request_context(request_context)} provider={config.provider} "
        f"model={config.model} mode=image-files images={len(paths)} timeout={config.request_timeout}s"
    )

    if config.provider == "google":
        from PIL import Image
        import google.generativeai as genai

        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model)
        content: list[Any] = [prompt]
        for label, path in zip(labels, paths):
            content.append(f"Image label: {label}")
            content.append(Image.open(path))
        response = model.generate_content(
            content,
            generation_config={"max_output_tokens": config.max_output_tokens},
            stream=False,
        )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response.")
        output = text.strip()
        print(f"[response]{_format_request_context(request_context)} provider=google")
        _log_model_output(output, request_context=request_context)
        return ModelResponse(text=output, usage=None)

    client = create_openai_compatible_client(config)
    content = [{"type": "text", "text": prompt}]
    for label, path in zip(labels, paths):
        content.append({"type": "text", "text": f"Image label: {label}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_bytes_as_data_url(
                        path.read_bytes(),
                        mime_type=_image_mime_type(path),
                    )
                },
            }
        )

    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=config.max_output_tokens,
        **_openai_extra_kwargs(config),
    )
    _log_openai_response(response, request_context=request_context)
    text = (response.choices[0].message.content or "").strip()
    _log_model_output(text, request_context=request_context)
    return ModelResponse(text=text, usage=_extract_openai_usage(response))


def answer_with_gemini_google_video(
    *,
    config: ModelConfig,
    prompt: str,
    video_path: Union[str, Path],
    request_context: Optional[dict[str, Any]] = None,
    continuation_messages: Optional[list[dict[str, Any]]] = None,
) -> ModelResponse:
    import google.generativeai as genai

    print(
        f"[request]{_format_request_context(request_context)} provider={config.provider} "
        f"model={config.model} mode=gemini-google video={Path(video_path).name}"
    )
    genai.configure(api_key=config.api_key)
    model = genai.GenerativeModel(config.model)
    uploaded_file = genai.upload_file(path=str(video_path))
    print(f"[request]{_format_request_context(request_context)} uploaded local video to Gemini provider")
    if continuation_messages:
        chat = model.start_chat(history=[{"role": "user", "parts": [prompt, uploaded_file]}])
        for message in continuation_messages[:-1]:
            role = "model" if message.get("role") == "assistant" else "user"
            text = str(message.get("content") or "")
            if text:
                chat.history.append({"role": role, "parts": [text]})
        final_message = continuation_messages[-1]
        final_text = str(final_message.get("content") or "")
        response = chat.send_message(
            final_text,
            generation_config={"max_output_tokens": config.max_output_tokens},
            stream=False,
        )
    else:
        response = model.generate_content(
            [prompt, uploaded_file],
            generation_config={"max_output_tokens": config.max_output_tokens},
            stream=False,
        )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response.")
    output = text.strip()
    print(f"[response]{_format_request_context(request_context)} provider=google")
    _log_model_output(output, request_context=request_context)
    return ModelResponse(text=output, usage=None)


def answer_with_gemini_openrouter_video(
    *,
    config: ModelConfig,
    prompt: str,
    video_path: Union[str, Path],
    request_context: Optional[dict[str, Any]] = None,
    continuation_messages: Optional[list[dict[str, Any]]] = None,
) -> ModelResponse:
    video_path = Path(video_path)
    video_data_url, cache_hit = get_cached_video_data_url(video_path)
    client = create_openai_compatible_client(config)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(
            f"[request]{_format_request_context(request_context)} provider={config.provider} model={config.model} "
            f"mode=gemini-openrouter video={video_path.name} cache_hit={cache_hit} "
            f"data_url_bytes={len(video_data_url)} timeout={config.request_timeout}s attempt={attempt}/{max_attempts}"
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_url},
                    },
                ],
            }
        ]
        if continuation_messages:
            messages.extend(continuation_messages)

        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_completion_tokens=config.max_output_tokens,
            **_openai_extra_kwargs(config),
        )
        _log_openai_response(response, request_context=request_context)
        text = (response.choices[0].message.content or "").strip()
        if text:
            _log_model_output(text, request_context=request_context)
            return ModelResponse(text=text, usage=_extract_openai_usage(response))

        print(f"[empty-response]{_format_request_context(request_context)} video={video_path.name} attempt={attempt}/{max_attempts}")
        print(f"[empty-response]{_format_request_context(request_context)} prompt={_truncate_text(prompt, limit=500)}")
        if attempt < max_attempts:
            print(f"[retry]{_format_request_context(request_context)} retrying after empty response")

    raise ValueError(
        f"OpenRouter Gemini returned an empty response for this request after {max_attempts} attempts. "
        "This may indicate an intermittent upstream or routing issue."
    )


def answer_video_question(
    *,
    config: ModelConfig,
    prompt: str,
    video_path: Union[str, Path],
    request_context: Optional[dict[str, Any]] = None,
    continuation_messages: Optional[list[dict[str, Any]]] = None,
    additional_reference_frame_indices: Optional[Sequence[int]] = None,
) -> ModelResponse:
    family = infer_model_family(config.model)
    if family == "gpt":
        return answer_with_gpt_video_frames(
            config=config,
            prompt=prompt,
            video_path=video_path,
            request_context=request_context,
            continuation_messages=continuation_messages,
            additional_reference_frame_indices=additional_reference_frame_indices,
        )

    if config.provider == "google":
        return answer_with_gemini_google_video(
            config=config,
            prompt=prompt,
            video_path=video_path,
            request_context=request_context,
            continuation_messages=continuation_messages,
        )

    if config.provider == "openrouter":
        return answer_with_gemini_openrouter_video(
            config=config,
            prompt=prompt,
            video_path=video_path,
            request_context=request_context,
            continuation_messages=continuation_messages,
        )

    raise ValueError(
        f"Provider '{config.provider}' does not support Gemini-style video input in this project."
    )
