from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_ROUTE_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "pipeline_route_policy.json"
)


class RoutePolicyValidationError(ValueError):
    """Raised when the route policy is incomplete, ambiguous, or malformed."""


class ScopeLevel(str, Enum):
    BENCHMARK = "benchmark"
    SCENARIO = "scenario"
    SCENE = "scene"
    SEGMENT = "segment"
    ROLE = "role"
    OBJECT = "object"
    QUESTION = "question"
    FAILURE = "failure"
    ENTRYPOINT = "entrypoint"
    RUN = "run"


class SelectionKind(str, Enum):
    SHARED_DEFAULT = "shared_default"
    EXPLICIT_PRIOR = "explicit_prior"
    DETERMINISTIC_MEASUREMENT = "deterministic_measurement"
    VLM_DECISION = "vlm_decision"
    RUNTIME_FALLBACK = "runtime_fallback"


class PipelineStage(str, Enum):
    INPUT = "input"
    SCENE_ASSESSMENT = "scene_assessment"
    OBJECT_PLAN = "object_plan"
    TRACKING = "tracking"
    LABELING = "labeling"
    INTRINSICS = "intrinsics"
    METRIC_DEPTH = "metric_depth"
    GEOMETRY = "geometry"
    MESH_CONDITIONING = "mesh_conditioning"
    POSE = "pose"
    POSE_CORRECTION = "pose_correction"
    SWR = "swr"
    ROLLOUT = "rollout"
    FINAL_ANSWER = "final_answer"
    DIRECT_ANSWER = "direct_answer"


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class RouteScope:
    level: ScopeLevel
    benchmarks: tuple[str, ...]
    scenarios: tuple[str, ...]
    segments: tuple[str, ...]
    roles: tuple[str, ...]
    question_types: tuple[str, ...]
    entrypoints: tuple[str, ...]


@dataclass(frozen=True)
class RouteResolution:
    scope: RouteScope
    selection_kind: SelectionKind
    route: str


@dataclass(frozen=True)
class RouteDecision:
    decision_id: str
    pipeline_stage: PipelineStage
    decision_granularity: ScopeLevel
    resolutions: tuple[RouteResolution, ...]


@dataclass(frozen=True)
class RouteOverride:
    decision_id: str
    route: str


@dataclass(frozen=True)
class OptionalRoute:
    option_id: str
    scope: RouteScope
    default_enabled: bool
    route_overrides: tuple[RouteOverride, ...]


@dataclass(frozen=True)
class RouteContext:
    benchmark: str
    scenario: str | None = None
    segment: str | None = None
    role: str | None = None
    question_type: str | None = None
    entrypoint: str | None = None


@dataclass(frozen=True)
class RoutePolicy:
    schema_version: int
    policy_id: str
    benchmarks: tuple[BenchmarkSpec, ...]
    decisions: tuple[RouteDecision, ...]
    approved_optional_routes: tuple[OptionalRoute, ...]

    def decision(self, decision_id: str) -> RouteDecision:
        matches = [item for item in self.decisions if item.decision_id == decision_id]
        if len(matches) != 1:
            raise RoutePolicyValidationError(
                f"expected exactly one decision {decision_id!r}, found {len(matches)}"
            )
        return matches[0]

    def resolve(self, decision_id: str, context: RouteContext) -> RouteResolution:
        decision = self.decision(decision_id)
        matches = [
            resolution
            for resolution in decision.resolutions
            if _scope_matches(resolution.scope, context)
        ]
        if len(matches) != 1:
            raise RoutePolicyValidationError(
                f"decision {decision_id!r} resolved to {len(matches)} routes for {context!r}"
            )
        return matches[0]

    def optional_route(self, option_id: str) -> OptionalRoute:
        matches = [
            item for item in self.approved_optional_routes if item.option_id == option_id
        ]
        if len(matches) != 1:
            raise RoutePolicyValidationError(
                f"expected exactly one optional route {option_id!r}, found {len(matches)}"
            )
        return matches[0]

    def enabled_optional_routes(
        self,
        option_ids: Sequence[str],
    ) -> tuple[OptionalRoute, ...]:
        if isinstance(option_ids, (str, bytes)):
            raise RoutePolicyValidationError(
                "enabled optional routes must be a sequence of option IDs, not a string"
            )
        normalized_ids: list[str] = []
        for index, option_id in enumerate(option_ids):
            if not isinstance(option_id, str) or not option_id or option_id.strip() != option_id:
                raise RoutePolicyValidationError(
                    f"enabled optional route at index {index} must be a non-empty, "
                    "whitespace-free option ID"
                )
            normalized_ids.append(option_id)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise RoutePolicyValidationError(
                f"enabled optional route IDs must be unique, got {normalized_ids!r}"
            )
        return tuple(self.optional_route(option_id) for option_id in normalized_ids)

    def resolve_record(
        self,
        decision_id: str,
        context: RouteContext,
        *,
        enabled_option_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        decision = self.decision(decision_id)
        resolution = self.resolve(decision_id, context)
        record_context = {"benchmark": context.benchmark}
        for name in (
            "scenario",
            "segment",
            "role",
            "question_type",
            "entrypoint",
        ):
            value = getattr(context, name)
            if value is not None:
                record_context[name] = value
        record = {
            "decision_id": decision.decision_id,
            "pipeline_stage": decision.pipeline_stage.value,
            "decision_granularity": decision.decision_granularity.value,
            "context": record_context,
            "selection_kind": resolution.selection_kind.value,
            "route": resolution.route,
        }
        applicable_overrides: list[tuple[OptionalRoute, RouteOverride]] = []
        for option in self.enabled_optional_routes(enabled_option_ids):
            if not _scope_matches(option.scope, context):
                continue
            applicable_overrides.extend(
                (option, override)
                for override in option.route_overrides
                if override.decision_id == decision_id
            )
        if not applicable_overrides:
            return record
        if len(applicable_overrides) != 1:
            option_ids = [option.option_id for option, _ in applicable_overrides]
            raise RoutePolicyValidationError(
                f"decision {decision_id!r} has conflicting enabled optional routes "
                f"for {context!r}: {option_ids!r}"
            )
        option, override = applicable_overrides[0]
        record.update(
            {
                "selection_kind": SelectionKind.EXPLICIT_PRIOR.value,
                "route": override.route,
                "option_id": option.option_id,
            }
        )
        return record


def load_route_policy(path: Path | str = DEFAULT_ROUTE_POLICY_PATH) -> RoutePolicy:
    policy_path = Path(path)
    try:
        with policy_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutePolicyValidationError(
            f"failed to read route policy {policy_path}: {exc}"
        ) from exc
    return parse_route_policy(payload, source=str(policy_path))


def parse_route_policy(payload: Any, *, source: str = "<memory>") -> RoutePolicy:
    root = _mapping(payload, source)
    _exact_keys(
        root,
        {
            "schema_version",
            "policy_id",
            "benchmarks",
            "decisions",
            "optional_profiles",
        },
        source,
    )
    schema_version = _integer(root["schema_version"], f"{source}.schema_version")
    if schema_version != 2:
        raise RoutePolicyValidationError(
            f"{source}.schema_version must be 2, got {schema_version}"
        )
    benchmarks = _mapping(root["benchmarks"], f"{source}.benchmarks")
    decisions = _mapping(root["decisions"], f"{source}.decisions")
    optional_profiles = _mapping(
        root["optional_profiles"], f"{source}.optional_profiles"
    )
    policy = RoutePolicy(
        schema_version=schema_version,
        policy_id=_string(root["policy_id"], f"{source}.policy_id"),
        benchmarks=tuple(
            _parse_benchmark(name, scenarios, f"{source}.benchmarks.{name}")
            for name, scenarios in benchmarks.items()
        ),
        decisions=tuple(
            _parse_decision(decision_id, item, f"{source}.decisions.{decision_id}")
            for decision_id, item in decisions.items()
        ),
        approved_optional_routes=tuple(
            _parse_optional_route(
                option_id, item, f"{source}.optional_profiles.{option_id}"
            )
            for option_id, item in optional_profiles.items()
        ),
    )
    _validate_policy(policy, source)
    return policy


def _parse_benchmark(name: Any, scenarios: Any, path: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=_string(name, f"{path}.name"),
        scenarios=_string_tuple(scenarios, f"{path}.scenarios"),
    )


def _parse_scope(payload: Any, path: str, *, level: ScopeLevel) -> RouteScope:
    item = _mapping(payload, path)
    _only_keys(
        item,
        {
            "benchmarks",
            "scenarios",
            "segments",
            "roles",
            "question_types",
            "entrypoints",
        },
        path,
    )
    return RouteScope(
        level=level,
        benchmarks=_string_tuple(item.get("benchmarks", []), f"{path}.benchmarks"),
        scenarios=_string_tuple(item.get("scenarios", []), f"{path}.scenarios"),
        segments=_string_tuple(item.get("segments", []), f"{path}.segments"),
        roles=_string_tuple(item.get("roles", []), f"{path}.roles"),
        question_types=_string_tuple(
            item.get("question_types", []), f"{path}.question_types"
        ),
        entrypoints=_string_tuple(
            item.get("entrypoints", []), f"{path}.entrypoints"
        ),
    )


def _parse_resolution(
    payload: Any, path: str, *, level: ScopeLevel
) -> RouteResolution:
    item = _mapping(payload, path)
    _exact_keys(item, {"when", "selection", "use"}, path)
    return RouteResolution(
        scope=_parse_scope(item["when"], f"{path}.when", level=level),
        selection_kind=_enum(
            SelectionKind, item["selection"], f"{path}.selection"
        ),
        route=_string(item["use"], f"{path}.use"),
    )


def _parse_decision(decision_id: Any, payload: Any, path: str) -> RouteDecision:
    item = _mapping(payload, path)
    _exact_keys(item, {"stage", "level", "routes"}, path)
    level = _enum(ScopeLevel, item["level"], f"{path}.level")
    return RouteDecision(
        decision_id=_string(decision_id, f"{path}.decision_id"),
        pipeline_stage=_enum(
            PipelineStage, item["stage"], f"{path}.stage"
        ),
        decision_granularity=level,
        resolutions=tuple(
            _parse_resolution(value, f"{path}.routes[{index}]", level=level)
            for index, value in enumerate(
                _sequence(item["routes"], f"{path}.routes")
            )
        ),
    )


def _parse_override(decision_id: Any, route: Any, path: str) -> RouteOverride:
    return RouteOverride(
        decision_id=_string(decision_id, f"{path}.decision_id"),
        route=_string(route, f"{path}.route"),
    )


def _parse_optional_route(option_id: Any, payload: Any, path: str) -> OptionalRoute:
    item = _mapping(payload, path)
    _exact_keys(
        item,
        {
            "level",
            "when",
            "default_enabled",
            "overrides",
        },
        path,
    )
    level = _enum(ScopeLevel, item["level"], f"{path}.level")
    overrides = _mapping(item["overrides"], f"{path}.overrides")
    return OptionalRoute(
        option_id=_string(option_id, f"{path}.option_id"),
        scope=_parse_scope(item["when"], f"{path}.when", level=level),
        default_enabled=_boolean(item["default_enabled"], f"{path}.default_enabled"),
        route_overrides=tuple(
            _parse_override(
                decision_id, route, f"{path}.overrides.{decision_id}"
            )
            for decision_id, route in overrides.items()
        ),
    )


def _validate_policy(policy: RoutePolicy, source: str) -> None:
    if not policy.benchmarks:
        raise RoutePolicyValidationError(f"{source}.benchmarks must not be empty")
    benchmark_names = [item.benchmark for item in policy.benchmarks]
    _require_unique(benchmark_names, f"{source}.benchmarks")
    for benchmark in policy.benchmarks:
        _require_unique(
            benchmark.scenarios, f"{source}.benchmarks[{benchmark.benchmark}].scenarios"
        )

    if not policy.decisions:
        raise RoutePolicyValidationError(f"{source}.decisions must not be empty")
    decision_ids = [item.decision_id for item in policy.decisions]
    _require_unique(decision_ids, f"{source}.decisions.decision_id")
    benchmark_map = {item.benchmark: set(item.scenarios) for item in policy.benchmarks}
    decision_map = {item.decision_id: item for item in policy.decisions}

    for decision in policy.decisions:
        if not decision.resolutions:
            raise RoutePolicyValidationError(
                f"decision {decision.decision_id!r} has no resolutions"
            )
        for resolution in decision.resolutions:
            if resolution.scope.level is not decision.decision_granularity:
                raise RoutePolicyValidationError(
                    f"decision {decision.decision_id!r} has granularity "
                    f"{decision.decision_granularity.value!r} but resolution scope level "
                    f"{resolution.scope.level.value!r}"
                )
            _validate_scope(
                resolution.scope,
                benchmark_map,
                f"decision {decision.decision_id!r}",
            )
        for index, left in enumerate(decision.resolutions):
            for right in decision.resolutions[index + 1 :]:
                if _scopes_overlap(left.scope, right.scope, benchmark_map):
                    raise RoutePolicyValidationError(
                        f"decision {decision.decision_id!r} has ambiguous overlapping scopes: "
                        f"{left.scope!r} and {right.scope!r}"
                    )

    option_ids = [item.option_id for item in policy.approved_optional_routes]
    _require_unique(option_ids, f"{source}.optional_profiles.option_id")
    for option in policy.approved_optional_routes:
        _validate_scope(option.scope, benchmark_map, f"optional route {option.option_id!r}")
        if option.default_enabled:
            raise RoutePolicyValidationError(
                f"optional route {option.option_id!r} must be disabled by default"
            )
        if not option.route_overrides:
            raise RoutePolicyValidationError(
                f"optional route {option.option_id!r} must name overrides"
            )
        override_ids = [item.decision_id for item in option.route_overrides]
        _require_unique(override_ids, f"optional route {option.option_id!r}.route_overrides")
        for override in option.route_overrides:
            decision = decision_map.get(override.decision_id)
            if decision is None:
                raise RoutePolicyValidationError(
                    f"optional route {option.option_id!r} overrides unknown decision "
                    f"{override.decision_id!r}"
                )
            matching_defaults = {
                resolution.route
                for resolution in decision.resolutions
                if _scopes_overlap(resolution.scope, option.scope, benchmark_map)
            }
            if len(matching_defaults) != 1:
                raise RoutePolicyValidationError(
                    f"optional route {option.option_id!r} overlaps multiple base routes "
                    f"for {override.decision_id!r}: "
                    f"{sorted(matching_defaults)!r}"
                )
            if override.route in matching_defaults:
                raise RoutePolicyValidationError(
                    f"optional route {option.option_id!r} override "
                    f"{override.decision_id!r} does not change the route"
                )


def _validate_scope(
    scope: RouteScope,
    benchmark_map: Mapping[str, set[str]],
    path: str,
) -> None:
    if not scope.benchmarks:
        raise RoutePolicyValidationError(f"{path} scope must name at least one benchmark")
    _require_unique(scope.benchmarks, f"{path}.scope.benchmarks")
    _require_unique(scope.scenarios, f"{path}.scope.scenarios")
    _require_unique(scope.segments, f"{path}.scope.segments")
    _require_unique(scope.roles, f"{path}.scope.roles")
    _require_unique(scope.question_types, f"{path}.scope.question_types")
    _require_unique(scope.entrypoints, f"{path}.scope.entrypoints")
    unknown_benchmarks = set(scope.benchmarks) - set(benchmark_map)
    if unknown_benchmarks:
        raise RoutePolicyValidationError(
            f"{path} scope has unknown benchmarks {sorted(unknown_benchmarks)!r}"
        )
    if scope.scenarios:
        if any(not benchmark_map[benchmark] for benchmark in scope.benchmarks):
            raise RoutePolicyValidationError(
                f"{path} scope cannot apply scenario filters to a benchmark without scenarios"
            )
        allowed = set.intersection(*(benchmark_map[name] for name in scope.benchmarks))
        unknown_scenarios = set(scope.scenarios) - allowed
        if unknown_scenarios:
            raise RoutePolicyValidationError(
                f"{path} scope has unknown scenarios {sorted(unknown_scenarios)!r}"
            )


def _scope_matches(scope: RouteScope, context: RouteContext) -> bool:
    if context.benchmark not in scope.benchmarks:
        return False
    if scope.scenarios and context.scenario not in scope.scenarios:
        return False
    if scope.segments and context.segment not in scope.segments:
        return False
    if scope.roles and context.role not in scope.roles:
        return False
    if scope.question_types and context.question_type not in scope.question_types:
        return False
    if scope.entrypoints and context.entrypoint not in scope.entrypoints:
        return False
    return True


def _scopes_overlap(
    left: RouteScope,
    right: RouteScope,
    benchmark_map: Mapping[str, set[str]],
) -> bool:
    if not _concrete_contexts(left, benchmark_map).intersection(
        _concrete_contexts(right, benchmark_map)
    ):
        return False
    return all(
        _selector_overlap(left_values, right_values)
        for left_values, right_values in (
            (left.segments, right.segments),
            (left.roles, right.roles),
            (left.question_types, right.question_types),
            (left.entrypoints, right.entrypoints),
        )
    )


def _concrete_contexts(
    scope: RouteScope, benchmark_map: Mapping[str, set[str]]
) -> set[tuple[str, str | None]]:
    contexts: set[tuple[str, str | None]] = set()
    for benchmark in scope.benchmarks:
        available = benchmark_map[benchmark]
        if not available:
            contexts.add((benchmark, None))
            continue
        selected = set(scope.scenarios) if scope.scenarios else available
        contexts.update((benchmark, scenario) for scenario in selected)
    return contexts


def _selector_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    return not left or not right or bool(set(left).intersection(right))


def _mapping(payload: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RoutePolicyValidationError(f"{path} must be an object")
    return payload


def _sequence(payload: Any, path: str) -> Sequence[Any]:
    if not isinstance(payload, list):
        raise RoutePolicyValidationError(f"{path} must be an array")
    return payload


def _exact_keys(payload: Mapping[str, Any], expected: set[str], path: str) -> None:
    keys = set(payload)
    missing = expected - keys
    unknown = keys - expected
    if missing or unknown:
        raise RoutePolicyValidationError(
            f"{path} fields mismatch: missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}"
        )


def _only_keys(payload: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise RoutePolicyValidationError(
            f"{path} has unknown fields {sorted(unknown)!r}"
        )


def _string(payload: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(payload, str) or (not allow_empty and not payload.strip()):
        raise RoutePolicyValidationError(f"{path} must be a non-empty string")
    return payload


def _string_tuple(payload: Any, path: str) -> tuple[str, ...]:
    values = _sequence(payload, path)
    return tuple(_string(value, f"{path}[{index}]") for index, value in enumerate(values))


def _integer(payload: Any, path: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise RoutePolicyValidationError(f"{path} must be an integer")
    return payload


def _boolean(payload: Any, path: str) -> bool:
    if not isinstance(payload, bool):
        raise RoutePolicyValidationError(f"{path} must be a boolean")
    return payload


def _enum(enum_type: type[Enum], payload: Any, path: str) -> Any:
    value = _string(payload, path)
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = sorted(item.value for item in enum_type)
        raise RoutePolicyValidationError(
            f"{path} must be one of {allowed!r}, got {value!r}"
        ) from exc


def _require_unique(values: Sequence[str], path: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise RoutePolicyValidationError(
            f"{path} contains duplicates {sorted(duplicates)!r}"
        )
