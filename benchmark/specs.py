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


# Grounding DINO recognition settings for Physion++.
GDINO_CONFIDENCE_THRESHOLD = 0.25
GDINO_BOX_AREA_CAP_FRACTION = 0.6
GDINO_SAME_LABEL_NMS_IOU = 0.5
# Boxes touching this many frame edges are treated as frame-spanning background.
GDINO_EDGE_MARGIN_PX = 2
GDINO_EDGE_SIDES_TO_DROP = 3

# Per-scenario query vocabularies omit background nouns that would be grounded as objects.
GDINO_VOCAB_BY_PHYSION_PP_SCENARIO: dict[str, list[str]] = {
    "bouncy_platform_pp": [
        "small dark red block", "large dark wooden platform", "wooden frame",
        "yellow mat", "green mat", "teal mat", "mat",
    ],
    "bouncy_wall_pp": [
        "small dark block in the air", "small dark object", "large wooden panel",
        "mat", "colored mat",
    ],
    "friction_collision_pp": [
        "small green object", "small colored cube", "small colored block",
    ],
    "friction_platform_pp": [
        "small red block", "small red cylinder", "yellow mat",
        "large wedge ramp", "house-shaped structure",
    ],
    "mass_collision_pp": [
        "brown ball", "small colored cube", "small green object",
        "small cone", "small cylinder",
    ],
}


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
