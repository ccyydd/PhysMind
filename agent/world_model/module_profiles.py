from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DEFAULT_MODULE_PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "pipeline_module_profiles.json"
)


class ModuleProfileValidationError(ValueError):
    """Raised when executable module profiles are malformed or ambiguous."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ModuleProfileValidationError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ModuleProfileValidationError(f"{path} must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ModuleProfileValidationError(
            f"{path} must be a non-empty string without surrounding whitespace"
        )
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModuleProfileValidationError(f"{path} must be an integer")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ModuleProfileValidationError(
            f"{path} keys must be {sorted(expected)!r}, got {sorted(actual)!r}"
        )


def _freeze_json(value: Any, path: str) -> Any:
    if isinstance(value, dict):
        frozen = {
            _string(key, f"{path}.<key>"): _freeze_json(item, f"{path}.{key}")
            for key, item in value.items()
        }
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ModuleProfileValidationError(
        f"{path} must contain only JSON-compatible values"
    )


@dataclass(frozen=True)
class ModuleSpec:
    implementation: str
    parameters: Mapping[str, Any]

    def parameter(self, name: str) -> Any:
        if name not in self.parameters:
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} has no parameter {name!r}"
            )
        return self.parameters[name]

    def require_string(self, name: str) -> str:
        value = self.parameter(name)
        if not isinstance(value, str):
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} parameter {name!r} must be a string"
            )
        return value

    def require_number(self, name: str) -> float:
        value = self.parameter(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} parameter {name!r} must be numeric"
            )
        return float(value)

    def require_boolean(self, name: str) -> bool:
        value = self.parameter(name)
        if not isinstance(value, bool):
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} parameter {name!r} must be a boolean"
            )
        return value

    def require_integer(self, name: str) -> int:
        value = self.parameter(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} parameter {name!r} must be an integer"
            )
        return value

    def require_strings(self, name: str) -> tuple[str, ...]:
        value = self.parameter(name)
        if not isinstance(value, tuple) or not all(
            isinstance(item, str) for item in value
        ):
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} parameter {name!r} must be a string array"
            )
        return value

    def require_integers(self, name: str) -> tuple[int, ...]:
        value = self.parameter(name)
        if not isinstance(value, tuple) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} parameter {name!r} must be an integer array"
            )
        return value

    def require_records(self, name: str) -> tuple[Mapping[str, Any], ...]:
        value = self.parameter(name)
        if not isinstance(value, tuple) or not all(
            isinstance(item, Mapping) for item in value
        ):
            raise ModuleProfileValidationError(
                f"module {self.implementation!r} parameter {name!r} must be an object array"
            )
        return value


@dataclass(frozen=True)
class RouteModuleProfile:
    decision_id: str
    route: str
    benchmark: str
    scenarios: tuple[str, ...]
    modules: Mapping[str, ModuleSpec]

    def module(self, name: str) -> ModuleSpec:
        if name not in self.modules:
            raise ModuleProfileValidationError(
                f"route profile {self.route!r} has no module {name!r}"
            )
        return self.modules[name]


@dataclass(frozen=True)
class TrackingModuleProfile(RouteModuleProfile):
    pass


@dataclass(frozen=True)
class ModuleProfilePolicy:
    schema_version: int
    profile_set_id: str
    shared_modules: Mapping[str, ModuleSpec]
    route_profiles: Mapping[str, RouteModuleProfile]

    def shared_module(self, name: str) -> ModuleSpec:
        if name not in self.shared_modules:
            raise ModuleProfileValidationError(
                f"module profile policy has no shared module {name!r}"
            )
        return self.shared_modules[name]

    def resolve_route(
        self,
        decision_id: str,
        route: str,
        *,
        benchmark: str,
        scenario: str,
    ) -> RouteModuleProfile:
        profile = self.route_profiles.get(route)
        if profile is None:
            raise ModuleProfileValidationError(
                f"no module profile for resolved route {route!r}"
            )
        if profile.decision_id != decision_id:
            raise ModuleProfileValidationError(
                f"route profile {route!r} expects decision_id "
                f"{profile.decision_id!r}, got {decision_id!r}"
            )
        if profile.benchmark != benchmark:
            raise ModuleProfileValidationError(
                f"route profile {route!r} expects benchmark "
                f"{profile.benchmark!r}, got {benchmark!r}"
            )
        if profile.scenarios:
            if scenario not in profile.scenarios:
                raise ModuleProfileValidationError(
                    f"route profile {route!r} does not allow scenario {scenario!r}; "
                    f"expected one of {list(profile.scenarios)!r}"
                )
        elif scenario:
            raise ModuleProfileValidationError(
                f"route profile {route!r} does not allow a scenario, got {scenario!r}"
            )
        return profile

    def resolve_tracking(
        self,
        tracking_route: str,
        *,
        scenario: str,
    ) -> TrackingModuleProfile:
        profile = self.resolve_route(
            "TRK-001.tracking_family",
            tracking_route,
            benchmark="physion_pp",
            scenario=scenario,
        )
        if not isinstance(profile, TrackingModuleProfile):
            raise ModuleProfileValidationError(
                f"tracking route {tracking_route!r} did not resolve to a tracking profile"
            )
        return profile


def _parse_module(value: Any, path: str) -> ModuleSpec:
    item = _mapping(value, path)
    _exact_keys(item, {"implementation", "parameters"}, path)
    parameters = _mapping(item["parameters"], f"{path}.parameters")
    return ModuleSpec(
        implementation=_string(item["implementation"], f"{path}.implementation"),
        parameters=_freeze_json(parameters, f"{path}.parameters"),
    )


def _parse_modules(value: Any, path: str) -> Mapping[str, ModuleSpec]:
    items = _mapping(value, path)
    if not items:
        raise ModuleProfileValidationError(f"{path} must not be empty")
    return MappingProxyType(
        {
            _string(name, f"{path}.<key>"): _parse_module(module, f"{path}.{name}")
            for name, module in items.items()
        }
    )


def _parse_strings(value: Any, path: str) -> tuple[str, ...]:
    items = tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )
    if len(items) != len(set(items)):
        raise ModuleProfileValidationError(f"{path} must not contain duplicates")
    return items


def parse_module_profile_policy(
    payload: Any,
    *,
    source: str = "<memory>",
) -> ModuleProfilePolicy:
    root = _mapping(payload, source)
    _exact_keys(
        root,
        {"schema_version", "profile_set_id", "shared_modules", "route_profiles"},
        source,
    )
    schema_version = _integer(root["schema_version"], f"{source}.schema_version")
    if schema_version != 2:
        raise ModuleProfileValidationError(
            f"{source}.schema_version must be 2, got {schema_version}"
        )
    raw_profiles = _mapping(root["route_profiles"], f"{source}.route_profiles")
    if not raw_profiles:
        raise ModuleProfileValidationError(
            f"{source}.route_profiles must not be empty"
        )
    profiles: dict[str, RouteModuleProfile] = {}
    for route, raw_profile in raw_profiles.items():
        route = _string(route, f"{source}.route_profiles.<key>")
        path = f"{source}.route_profiles.{route}"
        item = _mapping(raw_profile, path)
        _exact_keys(
            item,
            {"decision_id", "route", "benchmark", "scenarios", "modules"},
            path,
        )
        declared_route = _string(item["route"], f"{path}.route")
        if declared_route != route:
            raise ModuleProfileValidationError(
                f"{path}.route must equal its mapping key {route!r}"
            )
        decision_id = _string(item["decision_id"], f"{path}.decision_id")
        profile_type = (
            TrackingModuleProfile
            if decision_id == "TRK-001.tracking_family"
            else RouteModuleProfile
        )
        profiles[route] = profile_type(
            decision_id=decision_id,
            route=declared_route,
            benchmark=_string(item["benchmark"], f"{path}.benchmark"),
            scenarios=_parse_strings(item["scenarios"], f"{path}.scenarios"),
            modules=_parse_modules(item["modules"], f"{path}.modules"),
        )
    return ModuleProfilePolicy(
        schema_version=schema_version,
        profile_set_id=_string(root["profile_set_id"], f"{source}.profile_set_id"),
        shared_modules=_parse_modules(
            root["shared_modules"], f"{source}.shared_modules"
        ),
        route_profiles=MappingProxyType(profiles),
    )


def load_module_profile_policy(
    path: Path | str = DEFAULT_MODULE_PROFILE_PATH,
) -> ModuleProfilePolicy:
    profile_path = Path(path)
    try:
        with profile_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModuleProfileValidationError(
            f"failed to read module profile policy {profile_path}: {exc}"
        ) from exc
    return parse_module_profile_policy(payload, source=str(profile_path))


@lru_cache(maxsize=1)
def default_module_profile_policy() -> ModuleProfilePolicy:
    return load_module_profile_policy()
