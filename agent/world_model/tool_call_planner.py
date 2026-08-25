from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from agent.query import create_openai_compatible_client
from agent.world_model.tool_calling import (
    ToolCall,
    ToolCallRequest,
    ToolCallValidationError,
    ToolRegistry,
    build_prompt_only_tool_call_instruction,
    build_tool_call_repair_instruction,
    default_qcpr_tool_registry,
    parse_prompt_only_tool_call_response,
)
from benchmark.clevrer import ClevrerQuestion
from benchmark.specs import CLEVRER_CHOICE_POLICY
from utils.config import ModelConfig
from utils.terminal import terminal_print


def _log_tool(message: str) -> None:
    terminal_print(f"[query-conditioned-physical-rollout] tool=query_conditioned_physical_rollout {message}", flush=True)


CAUSAL_EVENT_TOOL_POLICY = (
    "For causality/responsibility choices phrased as collision/contact events, use event-time evidence before edit evidence: "
    "call get_event_time for the candidate event pair and for the target event pair. "
    "If the candidate event frame is greater than or equal to the target event frame, it is not responsible for the target; "
    "do not call simulate_edit for that event choice. "
    "If the candidate event is earlier, timing alone is still insufficient: if the events share an object, this is plausible "
    "causal-chain evidence; otherwise check for bridge contact between any candidate-event object and any target-event object "
    "after the candidate event and before the target event. "
    "Use simulate_edit only for object presence, entering, removal, or other non-event choices."
)
CLEVRER_TOOL_PLAN_REPAIR_DECISION_ID = "ROL-003.clevrer_tool_plan_repair"
CLEVRER_TOOL_PLAN_REPAIR_ROUTE = (
    "planning.json_repair_fallback"
)




class ToolCallPlanner:
    def plan(
        self,
        *,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory_summary: dict[str, Any],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        source_fit_error: Any,
        object_reference_map: dict[str, Any],
        available_rollout_ids: list[str] | None = None,
        previous_tool_results: list[dict[str, Any]] | None = None,
        fallback: ToolCallRequest,
        request_context: dict[str, Any],
        tool_plan_repair_route: dict[str, Any] | None,
    ) -> tuple[ToolCallRequest, dict[str, Any]]:
        raise NotImplementedError


class PromptOnlyToolCallPlanner(ToolCallPlanner):
    def __init__(self, *, config: ModelConfig, registry: ToolRegistry | None = None):
        self.config = config
        self.registry = registry or default_qcpr_tool_registry()

    def plan(
        self,
        *,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory_summary: dict[str, Any],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        source_fit_error: Any,
        object_reference_map: dict[str, Any],
        available_rollout_ids: list[str] | None = None,
        previous_tool_results: list[dict[str, Any]] | None = None,
        fallback: ToolCallRequest,
        request_context: dict[str, Any],
        tool_plan_repair_route: dict[str, Any] | None,
    ) -> tuple[ToolCallRequest, dict[str, Any]]:
        active_repair_route = _require_clevrer_tool_plan_repair_route(
            tool_plan_repair_route,
            request_context=request_context,
        )
        prompt = self._tool_planning_prompt(
            question=question,
            objects=objects,
            trajectory_summary=trajectory_summary,
            collisions=collisions,
            in_out_events=in_out_events,
            source_fit_error=source_fit_error,
            object_reference_map=object_reference_map,
            available_rollout_ids=available_rollout_ids,
            previous_tool_results=previous_tool_results,
        )
        try:
            raw_text = self._request_tool_plan(prompt=prompt, request_context=request_context)
            parsed = parse_prompt_only_tool_call_response(raw_text, self.registry)
            return parsed, {
                "status": "ok",
                "planner": "vlm_prompt_json",
                "raw_response": raw_text,
            }
        except ToolCallValidationError as exc:
            try:
                repair_prompt = build_tool_call_repair_instruction(
                    original_text=raw_text if "raw_text" in locals() else "",
                    validation_error=exc,
                    registry=self.registry,
                )
                repaired_text = self._request_tool_plan(prompt=repair_prompt, request_context=request_context)
                parsed = parse_prompt_only_tool_call_response(repaired_text, self.registry)
                return parsed, {
                    "status": "ok_after_repair",
                    "planner": "vlm_prompt_json",
                    "validation_errors": exc.issues,
                    "raw_response": raw_text if "raw_text" in locals() else "",
                    "repair_response": repaired_text,
                    **(
                        {
                            "clevrer_tool_plan_repair_route": deepcopy(
                                active_repair_route
                            )
                        }
                        if active_repair_route is not None
                        else {}
                    ),
                }
            except Exception as repair_exc:
                return fallback, {
                    "status": "fallback_after_validation_error",
                    "planner": "deterministic",
                    "validation_errors": exc.issues,
                    "repair_error": str(repair_exc),
                    "raw_response": raw_text if "raw_text" in locals() else "",
                    **(
                        {
                            "clevrer_tool_plan_repair_route": deepcopy(
                                active_repair_route
                            )
                        }
                        if active_repair_route is not None
                        else {}
                    ),
                }
        except Exception as exc:
            return fallback, {
                "status": "fallback_after_request_error",
                "planner": "deterministic",
                "error_message": str(exc),
                **(
                    {
                        "clevrer_tool_plan_repair_route": deepcopy(
                            active_repair_route
                        )
                    }
                    if active_repair_route is not None
                    else {}
                ),
            }

    def _request_tool_plan(self, *, prompt: str, request_context: dict[str, Any]) -> str:
        _log_tool(
            "tool_plan request "
            f"scene={request_context.get('scene_index')} question={request_context.get('question_id')} "
            f"provider={self.config.provider} model={self.config.model}"
        )
        client = create_openai_compatible_client(self.config)
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You plan physics rollout tool calls. Return valid JSON inside <answer> tags only. "
                        'Use "base" or exact rollout_id values returned by previous tool results; never invent rollout ids.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=self.config.max_output_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        _log_tool(
            "tool_plan response "
            f"scene={request_context.get('scene_index')} question={request_context.get('question_id')} "
            f"chars={len(text)}"
        )
        return text

    def _tool_planning_prompt(
        self,
        *,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory_summary: dict[str, Any],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        source_fit_error: Any,
        object_reference_map: dict[str, Any],
        available_rollout_ids: list[str] | None = None,
        previous_tool_results: list[dict[str, Any]] | None = None,
    ) -> str:
        choices = [
            {
                "choice_id": choice.choice_id,
                "choice_letter": chr(ord("A") + index),
                "choice": choice.choice,
            }
            for index, choice in enumerate(question.choices)
        ]
        summary = {
            "objects": objects,
            "trajectory_summary": trajectory_summary,
            "collisions": collisions[:20],
            "in_out_events": in_out_events[:20],
            "source_fit_error": source_fit_error,
            "object_reference_map": object_reference_map,
            "available_rollout_ids": available_rollout_ids or ["base"],
            "previous_tool_results": previous_tool_results or [],
        }
        return (
            "Plan the minimum tool calls needed to answer this physical reasoning question.\n"
            f"{CLEVRER_CHOICE_POLICY.prompt_text}\n"
            "Use inspect tools for ordinary descriptive or predictive questions when useful. "
            "For predictive questions about what happens next, use check_contact for candidate future interactions mentioned by the choices; do not rely only on inspecting the final state. "
            "Predictive contact/event-time checks use a boundary-aware start_frame near source_last_frame, with a five-frame buffer before the observed cutoff. "
            f"{CAUSAL_EVENT_TOOL_POLICY}\n"
            "Use simulate_edit for counterfactual edits such as removing an object. "
            "For counterfactual questions, decide the scope from the wording: will/will-not-happen under an edit means inspect the edited rollout directly over its full frame range; cause/responsibility/prevention/change means compare base and edited. Do not use predictive start-frame filtering. "
            "After an edited rollout, use its collision summary first, and use check_contact without start_frame for contact or collision claims that matter to the answer unless a time range is explicit. "
            "Use compare_rollouts when the question depends on how an edit changes collisions or trajectories. "
            "If previous_tool_results is empty, first create any needed edited rollout before inspecting it. "
            'For rollout_id fields, use only ids from World summary.available_rollout_ids. '
            "Never invent names such as unnamed_edit_1 or edited_rollout_1. "
            "Do not invent object ids; use only listed object_id values. "
            "When the question refers to one concrete object, map that reference to exactly one best object_id "
            "from object_reference_map before issuing simulate_edit. Do not issue multiple mutually exclusive "
            "edits for alternative interpretations; if the reference is ambiguous, choose the "
            "single best match and explain the uncertainty in reasoning_summary.\n\n"
            f"Question type: {question.question_type}\n"
            f"Question: {question.question}\n"
            f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
            f"World summary: {json.dumps(summary, ensure_ascii=False)}\n\n"
            f"{build_prompt_only_tool_call_instruction(self.registry)}"
        )


class NativeOpenAICompatibleToolCallPlanner(PromptOnlyToolCallPlanner):
    def plan(
        self,
        *,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory_summary: dict[str, Any],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        source_fit_error: Any,
        object_reference_map: dict[str, Any],
        available_rollout_ids: list[str] | None = None,
        previous_tool_results: list[dict[str, Any]] | None = None,
        fallback: ToolCallRequest,
        request_context: dict[str, Any],
        tool_plan_repair_route: dict[str, Any] | None,
    ) -> tuple[ToolCallRequest, dict[str, Any]]:
        prompt = self._native_tool_planning_prompt(
            question=question,
            objects=objects,
            trajectory_summary=trajectory_summary,
            collisions=collisions,
            in_out_events=in_out_events,
            source_fit_error=source_fit_error,
            object_reference_map=object_reference_map,
            available_rollout_ids=available_rollout_ids,
            previous_tool_results=previous_tool_results,
        )
        try:
            raw_tool_calls, raw_content = self._request_native_tool_plan(prompt=prompt, request_context=request_context)
            parsed = self._parse_native_tool_calls(raw_tool_calls)
            return parsed, {
                "status": "ok",
                "planner": "native_openai_tools",
                "raw_content": raw_content,
                "raw_tool_call_count": len(raw_tool_calls),
            }
        except ToolCallValidationError as exc:
            return fallback, {
                "status": "fallback_after_validation_error",
                "planner": "deterministic",
                "validation_errors": exc.issues,
            }
        except Exception as exc:
            return fallback, {
                "status": "fallback_after_request_error",
                "planner": "deterministic",
                "error_message": str(exc),
            }

    def _request_native_tool_plan(
        self,
        *,
        prompt: str,
        request_context: dict[str, Any],
    ) -> tuple[list[Any], str]:
        _log_tool(
            "native_tool_plan request "
            f"scene={request_context.get('scene_index')} question={request_context.get('question_id')} "
            f"provider={self.config.provider} model={self.config.model}"
        )
        client = create_openai_compatible_client(self.config)
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You plan physics rollout tool calls. Use the provided tools; do not answer the question directly. "
                        'Use "base" or exact rollout_id values returned by previous tool results; never invent rollout ids.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tools=self._openai_tools_payload(),
            tool_choice="auto",
            max_completion_tokens=self.config.max_output_tokens,
        )
        message = response.choices[0].message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        content = str(getattr(message, "content", "") or "")
        _log_tool(
            "native_tool_plan response "
            f"scene={request_context.get('scene_index')} question={request_context.get('question_id')} "
            f"tool_calls={len(tool_calls)} chars={len(content)}"
        )
        return tool_calls, content

    def _openai_tools_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self.registry.to_prompt_schema()
        ]

    def _parse_native_tool_calls(self, raw_tool_calls: list[Any]) -> ToolCallRequest:
        if not raw_tool_calls:
            raise ToolCallValidationError(["native tool response did not contain tool_calls"])
        tool_calls = []
        for index, raw_call in enumerate(raw_tool_calls):
            function = getattr(raw_call, "function", None)
            name = getattr(function, "name", None)
            arguments_text = getattr(function, "arguments", "{}")
            call_id = getattr(raw_call, "id", None)
            if not isinstance(name, str) or not name:
                raise ToolCallValidationError([f"tool_calls[{index}].function.name is missing"])
            try:
                arguments = json.loads(arguments_text or "{}")
            except json.JSONDecodeError as exc:
                raise ToolCallValidationError([f"tool_calls[{index}].function.arguments JSON parse failed: {exc.msg}"]) from None
            if not isinstance(arguments, dict):
                raise ToolCallValidationError([f"tool_calls[{index}].function.arguments must decode to an object"])
            tool_calls.append(
                ToolCall(
                    tool=name,
                    arguments=arguments,
                    call_id=call_id if isinstance(call_id, str) else None,
                )
            )
        return validate_native_tool_call_request(
            ToolCallRequest(
                tool_calls=tool_calls,
                reasoning_summary="native OpenAI-compatible tool call response",
                expected_evidence=["tool call results"],
            ),
            self.registry,
        )

    def _native_tool_planning_prompt(
        self,
        *,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory_summary: dict[str, Any],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        source_fit_error: Any,
        object_reference_map: dict[str, Any],
        available_rollout_ids: list[str] | None = None,
        previous_tool_results: list[dict[str, Any]] | None = None,
    ) -> str:
        choices = [
            {
                "choice_id": choice.choice_id,
                "choice_letter": chr(ord("A") + index),
                "choice": choice.choice,
            }
            for index, choice in enumerate(question.choices)
        ]
        summary = {
            "objects": objects,
            "trajectory_summary": trajectory_summary,
            "collisions": collisions[:20],
            "in_out_events": in_out_events[:20],
            "source_fit_error": source_fit_error,
            "object_reference_map": object_reference_map,
            "available_rollout_ids": available_rollout_ids or ["base"],
            "previous_tool_results": previous_tool_results or [],
        }
        return (
            "Plan the minimum available tool calls needed to answer this physical reasoning question.\n"
            f"{CLEVRER_CHOICE_POLICY.prompt_text}\n"
            "Use inspect tools for ordinary descriptive or predictive questions when useful. "
            "For predictive questions about what happens next, use check_contact for candidate future interactions mentioned by the choices; do not rely only on inspecting the final state. "
            "Predictive contact/event-time checks use a boundary-aware start_frame near source_last_frame, with a five-frame buffer before the observed cutoff. "
            f"{CAUSAL_EVENT_TOOL_POLICY}\n"
            "Use simulate_edit for counterfactual edits such as removing an object. "
            "For counterfactual questions, decide the scope from the wording: will/will-not-happen under an edit means inspect the edited rollout directly over its full frame range; cause/responsibility/prevention/change means compare base and edited. Do not use predictive start-frame filtering. "
            "After an edited rollout, use its collision summary first, and use check_contact without start_frame for contact or collision claims that matter to the answer unless a time range is explicit. "
            "Use compare_rollouts when the question depends on how an edit changes collisions or trajectories. "
            "If previous_tool_results is empty, first create any needed edited rollout before inspecting it. "
            'For rollout_id fields, use only ids from World summary.available_rollout_ids. '
            "Never invent names such as unnamed_edit_1 or edited_rollout_1. "
            "Do not invent object ids; use only listed object_id values. "
            "When the question refers to one concrete object, map that reference to exactly one best object_id "
            "from object_reference_map before issuing simulate_edit. Do not issue multiple mutually exclusive "
            "edits for alternative interpretations; if the reference is ambiguous, choose the "
            "single best match and explain the uncertainty in reasoning_summary.\n\n"
            f"Question type: {question.question_type}\n"
            f"Question: {question.question}\n"
            f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
            f"World summary: {json.dumps(summary, ensure_ascii=False)}"
        )


class NativeGeminiToolCallPlanner(PromptOnlyToolCallPlanner):
    def plan(
        self,
        *,
        question: ClevrerQuestion,
        objects: list[dict[str, Any]],
        trajectory_summary: dict[str, Any],
        collisions: list[dict[str, Any]],
        in_out_events: list[dict[str, Any]],
        source_fit_error: Any,
        object_reference_map: dict[str, Any],
        available_rollout_ids: list[str] | None = None,
        previous_tool_results: list[dict[str, Any]] | None = None,
        fallback: ToolCallRequest,
        request_context: dict[str, Any],
        tool_plan_repair_route: dict[str, Any] | None,
    ) -> tuple[ToolCallRequest, dict[str, Any]]:
        prompt = self._native_tool_planning_prompt(
            question=question,
            objects=objects,
            trajectory_summary=trajectory_summary,
            collisions=collisions,
            in_out_events=in_out_events,
            source_fit_error=source_fit_error,
            object_reference_map=object_reference_map,
            available_rollout_ids=available_rollout_ids,
            previous_tool_results=previous_tool_results,
        )
        try:
            raw_function_calls, raw_text = self._request_gemini_tool_plan(prompt=prompt, request_context=request_context)
            parsed = self._parse_gemini_function_calls(raw_function_calls)
            return parsed, {
                "status": "ok",
                "planner": "native_gemini_tools",
                "raw_content": raw_text,
                "raw_tool_call_count": len(raw_function_calls),
            }
        except ToolCallValidationError as exc:
            return fallback, {
                "status": "fallback_after_validation_error",
                "planner": "deterministic",
                "validation_errors": exc.issues,
            }
        except Exception as exc:
            return fallback, {
                "status": "fallback_after_request_error",
                "planner": "deterministic",
                "error_message": str(exc),
            }

    def _request_gemini_tool_plan(
        self,
        *,
        prompt: str,
        request_context: dict[str, Any],
    ) -> tuple[list[Any], str]:
        import google.generativeai as genai

        _log_tool(
            "gemini_tool_plan request "
            f"scene={request_context.get('scene_index')} question={request_context.get('question_id')} "
            f"provider={self.config.provider} model={self.config.model}"
        )
        genai.configure(api_key=self.config.api_key)
        model = genai.GenerativeModel(
            self.config.model,
            tools=self._gemini_tools_payload(),
            tool_config=self._gemini_tool_config_payload(),
            system_instruction=(
                "You plan physics rollout tool calls. Use the provided functions; do not answer the question directly. "
                'Use "base" or exact rollout_id values returned by previous tool results; never invent rollout ids.'
            ),
        )
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": self.config.max_output_tokens},
            stream=False,
        )
        function_calls = self._extract_gemini_function_calls(response)
        text = self._safe_gemini_text(response)
        _log_tool(
            "gemini_tool_plan response "
            f"scene={request_context.get('scene_index')} question={request_context.get('question_id')} "
            f"tool_calls={len(function_calls)} chars={len(text)}"
        )
        return function_calls, text

    def _gemini_tools_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "function_declarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": self._gemini_schema(tool["parameters"]),
                    }
                    for tool in self.registry.to_prompt_schema()
                ]
            }
        ]

    def _gemini_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"type", "description", "properties", "required", "items", "enum", "nullable", "format"}
        output = {}
        for key, value in schema.items():
            if key not in allowed_keys:
                continue
            if key == "properties" and isinstance(value, dict):
                output[key] = {
                    str(name): self._gemini_schema(item if isinstance(item, dict) else {})
                    for name, item in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                output[key] = self._gemini_schema(value)
            else:
                output[key] = value
        return output

    def _gemini_tool_config_payload(self) -> dict[str, Any]:
        return {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": self.registry.names(),
            }
        }

    def _extract_gemini_function_calls(self, response: Any) -> list[Any]:
        function_calls = []
        for candidate in list(getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in list(getattr(content, "parts", None) or []):
                function_call = getattr(part, "function_call", None)
                if function_call is not None and getattr(function_call, "name", None):
                    function_calls.append(function_call)
        return function_calls

    def _safe_gemini_text(self, response: Any) -> str:
        try:
            return str(getattr(response, "text", "") or "")
        except Exception:
            return ""

    def _parse_gemini_function_calls(self, function_calls: list[Any]) -> ToolCallRequest:
        if not function_calls:
            raise ToolCallValidationError(["Gemini response did not contain function calls"])
        tool_calls = []
        for function_call in function_calls:
            name = getattr(function_call, "name", None)
            args = getattr(function_call, "args", None)
            if not isinstance(name, str) or not name:
                raise ToolCallValidationError(["Gemini function call name is missing"])
            if args is None:
                arguments = {}
            elif isinstance(args, dict):
                arguments = dict(args)
            else:
                arguments = dict(args)
            tool_calls.append(ToolCall(tool=name, arguments=arguments, call_id=None))
        return validate_native_tool_call_request(
            ToolCallRequest(
                tool_calls=tool_calls,
                reasoning_summary="native Gemini function call response",
                expected_evidence=["tool call results"],
            ),
            self.registry,
        )


def validate_native_tool_call_request(request: ToolCallRequest, registry: ToolRegistry) -> ToolCallRequest:
    payload = {
        "tool_calls": [item.to_dict() for item in request.tool_calls],
        "reasoning_summary": request.reasoning_summary,
        "expected_evidence": request.expected_evidence,
    }
    parsed = parse_prompt_only_tool_call_response(json.dumps(payload), registry)
    return ToolCallRequest(
        tool_calls=parsed.tool_calls,
        reasoning_summary=request.reasoning_summary,
        expected_evidence=request.expected_evidence,
    )


def build_tool_call_planner(config: ModelConfig | None) -> ToolCallPlanner | None:
    if config is None:
        return None
    provider = str(config.provider).lower()
    mode = os.getenv("PHYSMIND_QCPR_TOOL_PLANNER", "prompt_json").strip().lower()
    if mode in {"none", "deterministic"}:
        return None
    if provider == "google":
        return NativeGeminiToolCallPlanner(config=config)
    if mode in {"native_openai", "openai_tools", "native_tools"}:
        return NativeOpenAICompatibleToolCallPlanner(config=config)
    return PromptOnlyToolCallPlanner(config=config)
