from __future__ import annotations

from benchmark.clevrer import ClevrerQuestion
from benchmark.physion_pp import (
    PHYSION_PP_BASELINE_CURTAIN_SCENARIOS,
    PHYSION_PP_BASELINE_NO_CURTAIN_SCENARIOS,
    PhysionPPQuestion,
)
from benchmark.specs import CLEVRER_CHOICE_POLICY


ANSWER_FORMATS = {"answer-tag", "plain-answer"}


def _answer_marker(answer_format: str) -> str:
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"Unsupported answer format: {answer_format}")
    if answer_format == "plain-answer":
        return "Answer:"
    return "<answer>"


def _descriptive_instruction(answer_format: str) -> str:
    marker = _answer_marker(answer_format)
    if answer_format == "answer-tag":
        return "First reason about the video and the physical events. Then output your final answer in <answer> </answer>."
    return f"First reason about the video and the physical events. Then end with a separate line formatted exactly as: {marker} your_answer"


def _multiple_choice_instruction(answer_format: str) -> str:
    marker = _answer_marker(answer_format)
    if answer_format == "answer-tag":
        return (
            "First reason about the video and the physical events. Then output the letters of all correct choices in <answer> </answer>.\n"
            "If multiple choices are correct, separate the letters with commas.\n"
            "If no choices are correct, output NONE inside <answer> </answer>."
        )
    return (
        "First reason about the video and the physical events. Then end with a separate line formatted exactly as:\n"
        f"{marker} A,C\n"
        "If multiple choices are correct, separate the letters with commas.\n"
        f"If no choices are correct, write exactly: {marker} NONE"
    )


DESCRIPTIVE_PROMPT = """You are solving a CLEVRER video physical reasoning question.
Watch the video carefully and answer the question.
"""


MULTIPLE_CHOICE_PROMPT = """You are solving a CLEVRER video physical reasoning question.
Watch the video carefully and determine which choices are correct.
"""


PHYSION_PP_NO_CURTAIN_OCP_PROMPT = """You are solving a Physion++ video physical reasoning question.
Watch the video carefully and predict future object contact.

This is one continuous scene with no curtain or rearrangement. Infer hidden mass, friction, or bounciness from motion, not appearance.

Near the end, the scene freezes for two seconds while cues identify the targets: red is the AGENT and yellow is the PATIENT. The freeze and cues do not advance time, physically stop objects, or reset velocity. Continue from just before the freeze using the preceding motion.

First reason about the physical evidence and future motion. Then output exactly yes or no in <answer> </answer>.
"""


PHYSION_PP_CURTAIN_OCP_PROMPT = """You are solving a Physion++ video physical reasoning question.
Watch the video carefully and predict future object contact.

Before the curtain, infer hidden mass, friction, or bounciness from motion, not appearance. After the curtain, objects may be rearranged: use the new setup and transfer only those properties, not prior positions or trajectories.

Near the end, the scene freezes for two seconds while cues identify the targets: red is the AGENT and yellow is the PATIENT. The freeze and cues do not advance time, physically stop objects, or reset velocity. Continue from just before the freeze using post-curtain motion.

First reason about the physical evidence and future motion. Then output exactly yes or no in <answer> </answer>.
"""


PHYSION_PP_TRACKING_CUE_NO_CURTAIN_OCP_PROMPT = """You are solving a Physion++ video physical reasoning question.
Watch the video frames carefully and predict future object contact.

This is one continuous scene with no curtain or rearrangement. Infer hidden mass, friction, or bounciness from motion, not appearance.

Throughout the input, a red overlay continuously identifies the AGENT and a yellow overlay continuously identifies the PATIENT. These tracking overlays are annotations only and do not change either object's appearance, motion, or physical properties. The clip ends immediately before the original stationary blinking-cue interval; no freeze or separate cue frame is included. Continue from the final observed motion.

First reason about the physical evidence and future motion. Then output exactly yes or no in <answer> </answer>.
"""


PHYSION_PP_TRACKING_CUE_CURTAIN_OCP_PROMPT = """You are solving a Physion++ video physical reasoning question.
Watch the video frames carefully and predict future object contact.

Before the curtain, infer hidden mass, friction, or bounciness from motion, not appearance. After the curtain, objects may be rearranged: use the new setup and transfer only those properties, not prior positions or trajectories.

Throughout the input, a red overlay continuously identifies the AGENT and a yellow overlay continuously identifies the PATIENT. These tracking overlays are annotations only and do not change either object's appearance, motion, or physical properties. The clip ends immediately before the original stationary blinking-cue interval; no freeze or separate cue frame is included. Continue from the final observed post-curtain motion.

First reason about the physical evidence and future motion. Then output exactly yes or no in <answer> </answer>.
"""








def build_descriptive_prompt(question: ClevrerQuestion, *, answer_format: str = "answer-tag") -> str:
    return (
        f"{DESCRIPTIVE_PROMPT}\n"
        f"{_descriptive_instruction(answer_format)}\n"
        f"Question type: {question.question_type}\n"
        f"Question: {question.question}"
    )


def build_multiple_choice_prompt(question: ClevrerQuestion, *, answer_format: str = "answer-tag") -> str:
    choice_lines = []
    for index, choice in enumerate(question.choices):
        label = chr(ord("A") + index)
        choice_lines.append(f"{label}. {choice.choice}")
    return (
        f"{MULTIPLE_CHOICE_PROMPT}\n"
        f"{_multiple_choice_instruction(answer_format)}\n"
        f"{CLEVRER_CHOICE_POLICY.prompt_text}\n"
        f"Question type: {question.question_type}\n"
        f"Question: {question.question}\n"
        "Choices:\n"
        f"{chr(10).join(choice_lines)}"
    )


def build_physion_pp_ocp_prompt(
    question: PhysionPPQuestion,
    *,
    scenario: str,
    persistent_tracking_cues: bool = False,
) -> str:
    if scenario in PHYSION_PP_BASELINE_NO_CURTAIN_SCENARIOS:
        task_prompt = (
            PHYSION_PP_TRACKING_CUE_NO_CURTAIN_OCP_PROMPT
            if persistent_tracking_cues
            else PHYSION_PP_NO_CURTAIN_OCP_PROMPT
        )
    elif scenario in PHYSION_PP_BASELINE_CURTAIN_SCENARIOS:
        task_prompt = (
            PHYSION_PP_TRACKING_CUE_CURTAIN_OCP_PROMPT
            if persistent_tracking_cues
            else PHYSION_PP_CURTAIN_OCP_PROMPT
        )
    else:
        raise ValueError(f"Unsupported Physion++ baseline scenario: {scenario}")
    return (
        f"{task_prompt}\n"
        f"Question type: {question.question_type}\n"
        f"Question: {question.question}"
    )
