from __future__ import annotations

from typing import Any, Iterable


PHYSION_PP_AGENT_SPHERE_SCENARIOS = {
    "friction_platform_pp",
    "bouncy_wall_pp",
}
PHYSION_PP_MAIN_SCENARIOS = {
    "friction_platform_pp",
    "bouncy_wall_pp",
    "bouncy_platform_pp",
    "friction_collision_pp",
    "mass_collision_pp",
}
FOUNDATIONPOSE_AGENT_GEOMETRY_DECISION_ID = (
    "GEO-003.foundationpose_agent_geometry"
)
SPHERE_AGENT_SKIP_FOUNDATIONPOSE_ROUTE = "agent_geometry.sphere_no_foundationpose"
NATIVE_MESH_FOUNDATIONPOSE_ROUTE = "agent_geometry.native_foundationpose"
FOUNDATIONPOSE_AGENT_GEOMETRY_ROUTES = {
    SPHERE_AGENT_SKIP_FOUNDATIONPOSE_ROUTE,
    NATIVE_MESH_FOUNDATIONPOSE_ROUTE,
}


def object_plan_scenario(payload: Any) -> str:
    if isinstance(payload, dict):
        special_scene = payload.get("special_scene")
    else:
        special_scene = getattr(payload, "special_scene", None)
    special_scene = special_scene if isinstance(special_scene, dict) else {}
    scene_metadata = special_scene.get("scene_metadata")
    scene_metadata = scene_metadata if isinstance(scene_metadata, dict) else {}
    for value in (scene_metadata.get("scenario"), special_scene.get("scenario")):
        if value:
            return str(value).strip().lower()
    return ""


def foundationpose_agent_geometry_route(
    object_plan_payload: Any,
    *,
    require_for_main_scenario: bool = True,
) -> dict[str, Any] | None:
    scenario = object_plan_scenario(object_plan_payload)
    if isinstance(object_plan_payload, dict):
        special_scene = object_plan_payload.get("special_scene")
    else:
        special_scene = getattr(object_plan_payload, "special_scene", None)
    special_scene = special_scene if isinstance(special_scene, dict) else {}
    route_record = special_scene.get("foundationpose_agent_geometry_route")
    if not isinstance(route_record, dict):
        if require_for_main_scenario and scenario in PHYSION_PP_MAIN_SCENARIOS:
            raise ValueError(
                "Physion++ main scenario is missing its FoundationPose "
                f"agent-geometry route record: scenario={scenario!r}"
            )
        return None
    if scenario not in PHYSION_PP_MAIN_SCENARIOS:
        raise ValueError(
            "FoundationPose agent-geometry route is not applicable to scenario "
            f"{scenario!r}"
        )
    if route_record.get("decision_id") != FOUNDATIONPOSE_AGENT_GEOMETRY_DECISION_ID:
        raise ValueError(
            "FoundationPose agent-geometry route has an unexpected decision_id: "
            f"{route_record.get('decision_id')!r}"
        )
    route = route_record.get("route")
    if route not in FOUNDATIONPOSE_AGENT_GEOMETRY_ROUTES:
        raise ValueError(f"unsupported FoundationPose agent-geometry route: {route!r}")
    context = route_record.get("context")
    if not isinstance(context, dict):
        raise ValueError("FoundationPose agent-geometry route is missing its context")
    expected_context = {
        "benchmark": "physion_pp",
        "scenario": scenario,
        "role": "agent",
    }
    for key, expected in expected_context.items():
        if context.get(key) != expected:
            raise ValueError(
                "FoundationPose agent-geometry route context mismatch: "
                f"{key}={context.get(key)!r}, expected {expected!r}"
            )
    if (
        route == SPHERE_AGENT_SKIP_FOUNDATIONPOSE_ROUTE
        and scenario not in PHYSION_PP_AGENT_SPHERE_SCENARIOS
    ):
        raise ValueError(
            f"sphere-agent FoundationPose skip is not applicable to {scenario!r}"
        )
    return route_record


def route_agent_geometry_policy_payload(
    *,
    route_record: dict[str, Any] | None,
    scenario: str,
    agent_object_ids: Iterable[str],
) -> dict[str, Any]:
    scenario = str(scenario or "").strip().lower()
    route = route_record.get("route") if isinstance(route_record, dict) else None
    if route == SPHERE_AGENT_SKIP_FOUNDATIONPOSE_ROUTE:
        mode = "sphere"
        gated = True
    elif route == NATIVE_MESH_FOUNDATIONPOSE_ROUTE:
        # Native-mesh routes retain their scenario-specific policy payload shape.
        mode = "native" if scenario in PHYSION_PP_AGENT_SPHERE_SCENARIOS else "sphere"
        gated = False
    elif route is None and scenario not in PHYSION_PP_MAIN_SCENARIOS:
        mode = "sphere"
        gated = False
    else:
        raise ValueError(
            f"cannot build agent-geometry policy for route={route!r}, scenario={scenario!r}"
        )
    effective_agent_ids = (
        sorted(str(value) for value in agent_object_ids) if gated else []
    )
    return {
        "version": 1,
        "mode": mode,
        "scenario": scenario,
        "scenario_gated": gated,
        "sphere_scenarios": sorted(PHYSION_PP_AGENT_SPHERE_SCENARIOS),
        "agent_object_ids": effective_agent_ids,
    }


def sphere_agent_track_ids(*, scenario: str, tracks_payload: Any) -> set[str]:
    scenario = str(scenario or "").strip().lower()
    if scenario not in PHYSION_PP_AGENT_SPHERE_SCENARIOS or not isinstance(tracks_payload, dict):
        return set()
    tracking = tracks_payload.get("physion_tracking")
    tracking = tracking if isinstance(tracking, dict) else {}
    role_binding = tracking.get("role_binding")
    role_binding = role_binding if isinstance(role_binding, dict) else {}
    if scenario == "friction_platform_pp":
        track_id = str(role_binding.get("agent_track") or "").strip()
        return {track_id} if track_id else set()
    assignments = role_binding.get("assignments")
    assignments = assignments if isinstance(assignments, dict) else {}
    track_ids = set()
    for segment in ("seg1", "seg2"):
        roles = assignments.get(segment)
        roles = roles if isinstance(roles, dict) else {}
        track_id = str(roles.get("agent") or "").strip()
        if track_id:
            track_ids.add(track_id)
    return track_ids


def sphere_agent_object_ids(
    *,
    scenario: str,
    tracks_payload: Any,
    target_objects: Iterable[Any],
) -> set[str]:
    track_ids = sphere_agent_track_ids(scenario=scenario, tracks_payload=tracks_payload)
    if not track_ids:
        return set()
    object_ids = set()
    for target in target_objects:
        if isinstance(target, dict):
            object_id = target.get("object_id")
            source_track_id = target.get("source_track_id")
        else:
            object_id = getattr(target, "object_id", None)
            source_track_id = getattr(target, "source_track_id", None)
        if object_id is not None and str(source_track_id or "") in track_ids:
            object_ids.add(str(object_id))
    return object_ids
