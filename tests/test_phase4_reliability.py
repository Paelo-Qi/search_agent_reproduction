from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
from PIL import Image

from opensearch_vl_repro.agent.image_registry import ImageRegistry
from opensearch_vl_repro.agent.layout_parsing import (
    LayoutApiConfig, LayoutBackendError, PaddleOCRAiStudioBackend,
)
from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
from opensearch_vl_repro.agent.reliability import (
    FileSystemToolCache, RetryPolicy, cache_identity, cached_tool_backend,
)
from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn
from opensearch_vl_repro.agent.search_providers import (
    SearchBackendError, SerperSearchBackend, load_search_config,
)
from opensearch_vl_repro.agent.tool_registry import ToolContext, ToolResult
from opensearch_vl_repro.evaluation import BatchRunner, BatchSample


ROOT = Path(__file__).resolve().parents[1]


def _context(image=None, *, sample_id="sample"):
    images = ImageRegistry()
    images.register_initial_image(image or Image.new("RGB", (8, 6), "navy"))
    return ToolContext(images, sample_id=sample_id, metadata={"previous_observation": "ignored"})


def test_text_cache_miss_then_hit_skips_provider_and_ignores_history(tmp_path):
    calls = []

    def backend(arguments, context):
        calls.append((arguments, context.sample_id))
        return ToolResult(status="success", observation="<observation>same</observation>",
                          metadata={"provider": "fake", "attempt_count": 1})

    cached = cached_tool_backend(
        tool="text_search", backend=backend, cache=FileSystemToolCache(tmp_path),
        behavior_version="search-v1", argument_defaults={"top_k": 5},
    )
    first = cached({"q": "  query  ", "hl": "EN"}, _context(sample_id="A"))
    second = cached({"hl": "en", "q": "query", "top_k": 5},
                    _context(sample_id="B"))
    assert first.metadata["cache_hit"] is False and first.metadata["attempt_count"] == 1
    assert second.metadata["cache_hit"] is True and second.metadata["attempt_count"] == 0
    assert first.metadata["cache_key"] == second.metadata["cache_key"]
    assert first.observation == second.observation
    assert calls == [({"q": "  query  ", "hl": "EN"}, "A")]
    entry = next(tmp_path.rglob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    assert "previous_observation" not in json.dumps(payload)
    assert "cache_hit" not in payload["metadata"]


def test_phase3_registry_caches_only_external_tools(tmp_path):
    class FakeSearchTools:
        def __init__(self):
            self.text_calls = 0

        def text_search(self, arguments, context):
            self.text_calls += 1
            return ToolResult(status="success", observation="text", metadata={"attempt_count": 1})

        def web_search(self, arguments, context):
            return ToolResult(status="success", observation="web")

        def image_search(self, arguments, context):
            return ToolResult(status="success", observation="image")

    tools = FakeSearchTools()
    registry = create_phase3_tool_registry(search_tools=tools, cache_dir=tmp_path)
    assert len(registry.list_tools()) == 8
    context = _context()
    crop = registry.execute("crop", {"image": "img_1", "x": 0, "y": 0,
                                     "width": 2, "height": 2}, context)
    assert crop.status == "success" and not list(tmp_path.rglob("*.json"))
    assert registry.execute("text_search", {"q": "x"}, context).metadata["cache_hit"] is False
    assert registry.execute("text_search", {"q": "x"}, context).metadata["cache_hit"] is True
    assert tools.text_calls == 1


def test_failed_result_is_not_cached_and_corruption_is_a_miss(tmp_path):
    calls = 0

    def failure(arguments, context):
        nonlocal calls
        calls += 1
        return ToolResult(status="error", error_type="timeout", observation="failed")

    cache = FileSystemToolCache(tmp_path)
    cached = cached_tool_backend(tool="web_search", backend=failure, cache=cache,
                                 behavior_version="search-v1")
    assert cached({"q": "x"}, _context()).metadata["cache_hit"] is False
    assert cached({"q": "x"}, _context()).metadata["cache_hit"] is False
    assert calls == 2 and not list(tmp_path.rglob("*.json"))

    key, _ = cache_identity("web_search", {"q": "broken"}, _context(),
                            behavior_version="search-v1")
    path = cache.path_for("web_search", key)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    recovered = cached({"q": "broken"}, _context())
    assert recovered.status == "error"
    assert recovered.metadata["cache_warning"].startswith("corrupt_cache_entry:")
    assert path.read_text(encoding="utf-8") == "{broken"  # failures do not overwrite it


def test_cache_artifact_redacts_secret_even_if_provider_echoes_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "super-secret-value")

    def backend(arguments, context):
        return ToolResult(
            status="success", observation="provider said super-secret-value",
            metadata={"echo": "super-secret-value", "super-secret-value": "key"},
        )

    cached = cached_tool_backend(tool="web_search", backend=backend,
                                 cache=FileSystemToolCache(tmp_path),
                                 behavior_version="search-v1")
    cached({"q": "super-secret-value"}, _context())
    artifact = next(tmp_path.rglob("*.json")).read_text(encoding="utf-8")
    assert "super-secret-value" not in artifact
    assert "[REDACTED]" in artifact


def test_image_cache_key_hashes_pixels_not_local_id_or_path(tmp_path):
    pixels = Image.new("RGB", (9, 7), "gold")
    png, bmp = tmp_path / "same.png", tmp_path / "same.bmp"
    pixels.save(png)
    pixels.save(bmp)
    a = _context(png, sample_id="A")
    b_images = ImageRegistry()
    b_images.register_initial_image(Image.new("RGB", (1, 1)))
    b_images.register_derived_image(Image.new("RGB", (1, 1)), parent_id="img_1")
    b_images.register_derived_image(bmp, parent_id="img_2")
    b = ToolContext(b_images, sample_id="B")
    key_a, safe_a = cache_identity("image_search", {"url": "img_1"}, a,
                                   behavior_version="image-v1")
    key_b, safe_b = cache_identity("image_search", {"url": "img_3"}, b,
                                   behavior_version="image-v1")
    key_c, _ = cache_identity("image_search", {"url": "img_1"},
                              _context(Image.new("RGB", (9, 7), "blue")),
                              behavior_version="image-v1")
    assert key_a == key_b and safe_a == safe_b
    assert key_a != key_c
    assert "img_" not in json.dumps(safe_a)


def test_layout_cache_flags_and_success_only(tmp_path):
    calls = 0

    def backend(arguments, context):
        nonlocal calls
        calls += 1
        if arguments.get("use_chart_recognition") is False:
            return ToolResult(status="error", error_type="timeout", observation="failed")
        return ToolResult(status="success", observation="layout", metadata={"page_count": 1})

    cached = cached_tool_backend(tool="layout_parsing", backend=backend,
                                 cache=FileSystemToolCache(tmp_path),
                                 behavior_version="layout-v1")
    context = _context()
    arguments = {"image": "img_1", "use_chart_recognition": True}
    assert cached(arguments, context).metadata["cache_hit"] is False
    assert cached(arguments, context).metadata["cache_hit"] is True
    assert cached({"image": "img_1", "use_chart_recognition": False}, context).status == "error"
    assert cached({"image": "img_1", "use_chart_recognition": False}, context).status == "error"
    assert calls == 3


class Response:
    def __init__(self, data=None, status=200):
        self.data, self.status_code = data, status

    def json(self):
        return self.data


class SearchSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.parametrize("outcomes,expected", [
    ([requests.ConnectionError(), Response({"organic": []})], 2),
    ([requests.Timeout(), requests.Timeout(), Response({"organic": []})], 3),
])
def test_search_transient_retry_without_test_sleep(monkeypatch, outcomes, expected):
    monkeypatch.setenv("SERPER_API_KEY", "secret")
    session = SearchSession(outcomes)
    backend = SerperSearchBackend(
        load_search_config(ROOT / "configs" / "search_backends.example.yaml").serper,
        session=session,
        retry=RetryPolicy(max_attempts=3, backoff_seconds=(0, 0), sleeper=lambda _: None),
    )
    assert backend.search("q", hl=None, limit=1) == ()
    assert backend.last_attempt_count == expected == session.calls


@pytest.mark.parametrize("error_type", ["authentication_error", "invalid_argument"])
def test_non_transient_errors_are_not_retried(error_type):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise SearchBackendError(error_type, "stop")

    with pytest.raises(SearchBackendError) as caught:
        RetryPolicy(max_attempts=3, sleeper=lambda _: None).run(operation)
    assert calls == 1 and caught.value.attempt_count == 1


class LayoutResponse:
    def __init__(self, payload=None, *, text="", status=200):
        self.payload, self.text, self.status_code = payload, text, status

    def json(self):
        return self.payload


class LayoutSession:
    def __init__(self, polls):
        self.polls = list(polls)
        self.post_count = 0
        self.poll_urls = []

    def post(self, *args, **kwargs):
        self.post_count += 1
        return LayoutResponse({"data": {"jobId": "same-job"}})

    def get(self, url, **kwargs):
        if "same-job" in url:
            self.poll_urls.append(url)
            outcome = self.polls.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return LayoutResponse(text=json.dumps({"result": {"layoutParsingResults": [
            {"markdown": {"text": "done"}}
        ]}}))


def _layout_config(max_poll=600):
    return LayoutApiConfig("paddleocr_aistudio", "https://example.test/jobs", "model",
                           "PADDLEOCR_ACCESS_TOKEN", 10, 1, max_poll)


def test_paddle_poll_retry_reuses_job_and_submit_once(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "token")
    session = LayoutSession([
        requests.ConnectionError(),
        LayoutResponse({"data": {"state": "done", "resultUrl": {
            "jsonUrl": "https://result.test/out.jsonl"}}}),
    ])
    backend = PaddleOCRAiStudioBackend(
        _layout_config(), session=session, sleep=lambda _: None,
        poll_retry=RetryPolicy(max_attempts=3, backoff_seconds=(0, 0), sleeper=lambda _: None),
    )
    document = backend.parse(Image.new("RGB", (4, 4)), use_chart_recognition=None,
                             use_doc_orientation_classify=None)
    assert document.blocks[0].content == "done"
    assert session.post_count == 1
    assert len(session.poll_urls) == 2 and len(set(session.poll_urls)) == 1
    assert backend.last_attempt_count == 2


class AdvancingClock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_paddle_deadline_never_resubmits(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "token")
    clock = AdvancingClock()
    session = LayoutSession([requests.Timeout(), requests.Timeout(), requests.Timeout()])
    backend = PaddleOCRAiStudioBackend(
        _layout_config(max_poll=1.5), session=session, clock=clock.clock, sleep=clock.sleep,
        poll_retry=RetryPolicy(max_attempts=3, backoff_seconds=(1, 1), sleeper=clock.sleep),
    )
    with pytest.raises(LayoutBackendError) as caught:
        backend.parse(Image.new("RGB", (4, 4)), use_chart_recognition=None,
                      use_doc_orientation_classify=None)
    assert getattr(caught.value, "error_type") == "timeout"
    assert session.post_count == 1


class FakeRuntime:
    def __init__(self):
        self.calls = {}

    def run(self, *, question, images, sample_id, benchmark):
        self.calls[sample_id] = self.calls.get(sample_id, 0) + 1
        failed = sample_id == "B" and self.calls[sample_id] == 1
        turn = AgentTurn(
            assistant_output='text_search({"q":"evidence"})',
            tool_call={"name": "text_search", "arguments": {"q": "evidence"}},
            observation="<observation>evidence</observation>",
            status="error" if failed else "success", error="timeout" if failed else None,
            metadata={"cache_hit": False, "attempt_count": 1},
            tool_latency_seconds=0.01,
        )
        return AgentTrajectory(
            sample_id, benchmark, [turn], None if failed else "answer",
            "tool_error" if failed else "success", ["img_1"],
            error="failed" if failed else None,
            images=[{"image_id": "img_1", "parent_id": None, "kind": "initial",
                     "size": [4, 4], "sha256": "a" * 64, "metadata": {}}],
        )


def _samples():
    return [BatchSample(name, "synthetic", f"question {name}",
                        [Image.new("RGB", (4, 4))]) for name in "ABCD"]


def test_batch_resume_retry_failed_and_trajectory_persistence(tmp_path):
    runtime = FakeRuntime()
    runner = BatchRunner(runtime, tmp_path)
    first = runner.run(_samples(), max_samples=2)
    assert first["success"] == 1 and first["failed"] == 1 and first["pending"] == 2
    resumed = runner.run(_samples())
    assert resumed["success"] == 3 and resumed["failed"] == 1
    assert runtime.calls == {"A": 1, "B": 1, "C": 1, "D": 1}
    retried = runner.run(_samples(), retry_failed=True)
    assert retried["success"] == 4 and runtime.calls["B"] == 2
    lines = (tmp_path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    records = {record["sample_id"]: record for record in map(json.loads, lines)}
    record = records["A"]
    assert record["question"] == "question A" and record["final_answer"] == "answer"
    turn = record["trajectory"]["turns"][0]
    assert turn["assistant_output"] and turn["tool_call"]["arguments"]
    assert turn["observation"] and turn["metadata"] and turn["tool_latency_seconds"]
    serialized = json.dumps(records)
    assert "base64" not in serialized and "raw image" not in serialized


def test_stale_running_becomes_failed_and_requires_explicit_retry(tmp_path):
    runtime = FakeRuntime()
    runner = BatchRunner(runtime, tmp_path)
    state = {"version": 1, "samples": {
        "A": {"benchmark": "synthetic", "status": "running", "attempts": 1,
              "error_type": None, "error": None}
    }}
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "status.json").write_text(json.dumps(state), encoding="utf-8")
    sample = _samples()[:1]
    summary = runner.run(sample)
    assert summary["failed"] == 1 and runtime.calls == {}
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["samples"]["A"]["error_type"] == "interrupted"
    assert runner.run(sample, retry_failed=True)["success"] == 1
    assert runtime.calls == {"A": 1}
