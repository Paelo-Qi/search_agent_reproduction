#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.agent.mock_tools import (  # noqa: E402
    ScriptedAgentModel,
    create_mock_tool_registry,
)
from opensearch_vl_repro.agent.runtime import AgentRuntime  # noqa: E402


def main() -> None:
    model = ScriptedAgentModel(
        [
            'image_search({"url":"img_1"})',
            '<tool_call>{"name":"text_search","arguments":'
            '{"q":"example entity","hl":"en","top_k":5}}</tool_call>',
            "The example entity is supported by the mock search observations.",
        ]
    )
    runtime = AgentRuntime(
        model=model,
        tool_registry=create_mock_tool_registry(),
        max_agent_turns=4,
    )
    trajectory = runtime.run(
        question="Identify the example entity and verify it with search.",
        images=[Image.new("RGB", (16, 16), color=(40, 80, 120))],
        sample_id="agent-mock-smoke",
        benchmark="synthetic",
    )
    print(json.dumps(trajectory.to_dict(), ensure_ascii=False, indent=2))
    if trajectory.status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

