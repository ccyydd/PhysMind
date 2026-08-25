from __future__ import annotations

SCENE_ASSESSMENT_PROMPT = """You are assessing a video scene for a physical world-modeling pipeline.
Do not use benchmark defaults. Judge only from the video.

Return compact JSON inside <answer> </answer> with this shape:
{
  "horizontal_plane_motion": {
    "applies": true,
    "confidence": 0.8,
    "reason": "brief visual evidence"
  },
  "roll_stabilization": {
    "applies": true,
    "confidence": 0.8,
    "reason": "brief visual evidence"
  },
  "support_surfaces": [
    {
      "object_id": "support_plane",
      "description": "visible floor/table/support surface",
      "role": "static collision support",
      "geometry_type": "plane",
      "confidence": 0.8
    }
  ],
  "reasoning": "brief reason"
}

For horizontal_plane_motion, judge whether the main moving rigid objects appear to move on one shared support plane.
For roll_stabilization, judge whether the image roll is stable, with no visible in-plane camera rotation.
If uncertain, set applies to false and lower confidence rather than assuming a benchmark prior.
support_surfaces should include only visible fixed surfaces needed as collision support, such as floor, table, wall, or ramp.
"""


def build_scene_assessment_prompt() -> str:
    return SCENE_ASSESSMENT_PROMPT
