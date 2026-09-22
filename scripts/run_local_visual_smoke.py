#!/usr/bin/env python3
"""CPU-only, no-network Phase 2 visual tool chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel  # noqa: E402
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.runtime import AgentRuntime  # noqa: E402


def main() -> None:
    image = Image.new("RGB", (64, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 12, 35, 32), fill="navy")
    model = ScriptedAgentModel([
        'crop({"image":"img_1","x":10,"y":8,"width":32,"height":28})',
        'sharpen({"image":"img_2","amount":2})',
        'super_resolution({"image":"img_3","scale":2})',
        "The local visual tool chain completed.",
    ])
    trajectory = AgentRuntime(
        model=model, tool_registry=create_phase2_tool_registry(), max_agent_turns=4,
    ).run(question="Inspect the marked region.", images=[image], sample_id="phase2-local-visual-smoke")
    expected = ["img_1", "img_2", "img_3", "img_4"]
    passed = trajectory.status == "success" and trajectory.image_ids == expected and all(
        turn.status == "success" and len(turn.derived_images) == 1 for turn in trajectory.turns
    )
    report = {"passed": passed, "trajectory": trajectory.to_dict(),
              "final_registry_state": trajectory.images}
    report_path = PROJECT_ROOT / "reports" / "phase2_local_visual_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": passed, "tool_statuses": [t.status for t in trajectory.turns],
                      "images": trajectory.images, "report": str(report_path)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
