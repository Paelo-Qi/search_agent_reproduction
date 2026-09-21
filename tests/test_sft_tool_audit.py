from __future__ import annotations

import json
from pathlib import Path

from opensearch_vl_repro.sft_tool_audit import (
    audit_records,
    extract_tool_calls,
    iter_json_array,
    render_markdown,
    run_audit,
)


def tool_turn(name: str, arguments: object) -> dict[str, str]:
    payload = json.dumps({"name": name, "arguments": arguments})
    return {"from": "gpt", "value": f"<think>audit</think><tool_call>{payload}</tool_call>"}


def fixture_records() -> list[object]:
    return [
        {
            "conversations": [
                {"from": "human", "value": "<image>question"},
                tool_turn("crop", {"image": "img_1", "bbox": [0, 0, 10, 10]}),
                {"from": "observation", "value": "Image saved as img_2"},
                tool_turn("crop", {"image": "img_2"}),
                {
                    "from": "observation",
                    "value": json.dumps({"status": "error", "error": "invalid bbox"}),
                },
                tool_turn("image_search", '{"image":"img_2"}'),
                {
                    "from": "observation",
                    "value": {"status": "success", "results": [{"title": "Result"}]},
                },
                {"from": "gpt", "value": "final"},
            ],
            "tools": "[]",
        },
        {
            "conversations": [
                {"from": "human", "value": "No tool question"},
                {"from": "gpt", "value": "Direct answer"},
            ]
        },
        {"conversations": "malformed"},
        {
            "conversations": [
                {"from": "human", "value": "question"},
                tool_turn("visit", {"url": "https://example.test"}),
            ]
        },
    ]


def test_extracts_tool_call_name_and_arguments() -> None:
    calls, errors = extract_tool_calls(tool_turn("text_search", {"query": "bridge"}))
    assert errors == []
    assert calls == [
        {"name": "text_search", "arguments": {"query": "bridge"}, "parse_error": None}
    ]


def test_associates_immediate_observations_and_json_or_text_formats() -> None:
    report = audit_records(fixture_records())
    crop = report["tools"]["crop"]
    assert crop["total_call_count"] == 2
    assert crop["trajectory_count"] == 1
    assert crop["max_calls_in_single_trajectory"] == 2
    assert crop["observation_formats"] == {"text": 1, "json": 1}
    assert crop["success_like_count"] == 1
    assert crop["failure_like_count"] == 1
    assert crop["examples"]["success_like"][0]["observation_preview"] == "Image saved as img_2"
    assert crop["common_top_level_keys"] == {"error": 1, "status": 1}


def test_argument_schemas_include_multiple_shapes_and_encoded_json() -> None:
    report = audit_records(fixture_records())
    crop = report["tools"]["crop"]
    image_search = report["tools"]["image_search"]
    assert crop["has_multiple_argument_schemas"] is True
    assert crop["argument_keys"]["image"]["count"] == 2
    assert crop["argument_keys"]["bbox"]["count"] == 1
    assert image_search["argument_types"] == {"string": 1}
    assert image_search["argument_schemas"][0]["schema"]["encoding"] == "json_string"
    assert image_search["observation_python_types"] == {"object": 1}


def test_trajectory_counts_combinations_and_transitions() -> None:
    report = audit_records(fixture_records())
    assert report["total_trajectories"] == 4
    assert report["trajectories_with_any_tool"] == 2
    assert report["trajectories_without_tool"] == 2
    assert report["total_tool_calls"] == 4
    assert report["max_tool_calls_per_trajectory"] == 3
    transitions = {(row["from"], row["to"]): row["count"] for row in report["tool_transitions"]}
    assert transitions[("crop", "crop")] == 1
    assert transitions[("crop", "image_search")] == 1
    combinations = {tuple(row["tools"]): row["trajectory_count"] for row in report["tool_combinations"]}
    assert combinations[("crop", "image_search")] == 1
    basis = report["tools"]["crop"]["compatibility_implication"]["basis"]
    assert basis["outgoing_transition_count"] == 2
    assert basis["incoming_transition_count"] == 1
    assert basis["pairwise_cooccurrence_trajectory_sum"] == 1


def test_malformed_trajectory_and_missing_observation_are_recorded() -> None:
    report = audit_records(fixture_records())
    assert report["malformed_trajectory_count"] == 2
    assert report["malformed_reasons"]["conversations is not a list"] == 1
    assert report["malformed_reasons"]["tool call is the final turn"] == 1
    assert report["tools"]["visit"]["missing_observation_count"] == 1
    assert report["tools"]["visit"]["unknown_count"] == 1


def test_streaming_reader_and_repeated_audit_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture_records()), encoding="utf-8")
    first_records = list(iter_json_array(path, chunk_size=17))
    second_records = list(iter_json_array(path, chunk_size=31))
    assert first_records == second_records == fixture_records()
    assert audit_records(first_records) == audit_records(second_records)


def test_local_file_audit_and_markdown_do_not_download(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture_records()), encoding="utf-8")
    report = run_audit(
        project_root=tmp_path,
        input_value=path,
        raw_dir=tmp_path / "raw",
        download_missing=False,
    )
    markdown = render_markdown(report)
    assert report["source_errors"] == []
    assert report["discovered_tools"] == ["crop", "image_search", "visit"]
    assert "## 8. Compatibility implications" in markdown
    assert "### Focused contract conclusions" in markdown
    assert "`image_search`" in markdown
    assert report["focus_tool_contracts"]["image_search"][
        "multiple_argument_schemas_observed"
    ] is False
