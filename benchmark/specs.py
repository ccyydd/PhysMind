from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkChoicePolicy:
    prompt_text: str


CLEVRER_CHOICE_POLICY = BenchmarkChoicePolicy(
    prompt_text=(
        "CLEVRER multiple-choice questions are multi-select: zero, one, or multiple choices can be correct. "
        "Judge every choice independently. Select all choices whose answer is yes; if no choice is correct, output NONE. "
        "Do not choose only the single most obvious or best choice."
    )
)





SAM3_VIDEO_TRACK_PROMPTS_BY_BENCHMARK: dict[str, list[dict[str, Any]]] = {
    "clevrer": [
        {"concept_id": "clevrer_dynamic_objects", "prompt": "all visible objects", "source": "bench:clevrer"},
    ],
    "physion_pp": [
        {
            "concept_id": "physion_dynamic_objects",
            "prompt": "all visible objects",
            "source": "bench:physion_pp",
        },
    ],
}
