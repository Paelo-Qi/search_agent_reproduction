"""Gate-only reward orchestration around the EXISTING DeepSeek client transport.

Two independent bounded-retry requests and success-only versioned caches.
References exist here, never in workflow tasks or policy batches.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from opensearch_vl_repro.agent.reliability import redact_secrets
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.evaluation.judge import (
    JUDGE_PROMPT_VERSION, DeepSeekJudge, JudgeProviderError, JudgeSample,
    build_judge_messages, parse_judge_response,
)
from opensearch_vl_repro.rl.actor_gate import atomic_json
from opensearch_vl_repro.rl.live_workflow import ProviderInterruption
from opensearch_vl_repro.rl.query_judge import build_query_messages, parse_query_response
from opensearch_vl_repro.rl.reward import compose_reward, format_reward, unit_reward
from opensearch_vl_repro.rl.workflow_types import RLInfrastructureError

QUERY_PROMPT_VERSION = "rl-query-utility-v1"
ABNORMAL_ACCURACY_VERSION = "rl-accuracy-no-final-v1"


def bind_trajectory_reward(trajectory, total):
    """Gate C only: bind REAL outcome after both judges succeed, not tool-transition zeros."""
    value = unit_reward(total, "rLLM trajectory reward")
    if not trajectory.steps:
        raise ValueError("real rLLM trajectory steps required")
    trajectory.reward = value
    for index, step in enumerate(trajectory.steps):
        step.reward = value if index == len(trajectory.steps) - 1 else 0.0
        step.mc_return = value
        step.info = {**step.info, "reward_computed": True, "trajectory_reward": value}


def reward_cache_identity(*, kind, config, question, reference, trajectory, messages):
    if kind not in {"accuracy", "query"}:
        raise ValueError("separate judge cache kind required")
    return {"kind": kind, "prompt_version": JUDGE_PROMPT_VERSION if kind == "accuracy" else QUERY_PROMPT_VERSION,
            "judge_config": asdict(config), "question": question, "reference_answer": reference,
            "trajectory": trajectory, "messages": messages}


def cached_request(client: DeepSeekJudge, directory: Path, identity, messages, parser):
    key = canonical_json_sha256(identity)
    path = directory / identity["kind"] / (key + ".json")
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity or value.get("status") != "success":
            raise RLInfrastructureError("reward cache identity/status corrupted")
        parsed = parser(value["raw_response"])
        return parsed, {**value, "cache_hit": True}
    api_key = os.environ.get(client.config.api_key_env)
    if not api_key:
        raise ProviderInterruption("configuration_error", judge=True, detail="judge API key missing")
    def operation():
        raw = client.request_messages(messages, api_key)
        try:
            return raw, parser(raw)
        except (ValueError, TypeError) as exc:
            raise JudgeProviderError("invalid_response", str(exc), retryable=True, raw_response=raw) from exc
    try:
        (raw, parsed), attempts = client.retry.run(operation)
    except JudgeProviderError as exc:
        # Preserve existing reason enums, never fabricate model reward on failure.
        error_type = exc.error_type if exc.error_type != "invalid_request" else "configuration_error"
        raise ProviderInterruption(error_type, judge=True, detail=str(redact_secrets(str(exc)))) from exc
    value = {"identity": identity, "status": "success", "raw_response": redact_secrets(raw),
             "attempt_count": attempts, "cache_hit": False, "real_provider_request": True}
    atomic_json(path, value)
    return parsed, value


def live_rewards(*, client, directory, row, trajectory, fatal):
    # Full relevant trace, without latency/cache bookkeeping (stable across retries).
    trace = [{"tool_call": t.tool_call, "observation": t.observation, "status": t.status,
              "error": t.error, "assistant_output": t.assistant_output} for t in trajectory.turns]
    evidence = {"tool_trace": trace, "final_answer": trajectory.final_answer,
                "termination": trajectory.status, "fatal": fatal, "images": trajectory.images}
    sample = JudgeSample(row["source_sample_id"], "rl", row["question"], row["reference_answer"], trajectory.final_answer)
    accuracy_messages = build_judge_messages(sample)
    if not trajectory.final_answer:
        # No invented answer. Provider still judges real null answer and termination.
        accuracy_messages[0]["content"] += (
            " No final answer was produced. A null model_answer is an abnormal termination, "
            "not an answer; assess correctness conservatively and return the same verdict JSON.")
        payload = json.loads(accuracy_messages[1]["content"])
        payload.update(termination=trajectory.status, fatal=fatal, no_final_answer=True,
                       abnormal_prompt_version=ABNORMAL_ACCURACY_VERSION)
        accuracy_messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
    acc_identity = reward_cache_identity(kind="accuracy", config=client.config, question=row["question"],
        reference=row["reference_answer"], trajectory=evidence, messages=accuracy_messages)
    (verdict, reason), acc_audit = cached_request(client, directory, acc_identity, accuracy_messages, parse_judge_response)
    query_messages = build_query_messages(question=row["question"], reference_answer=row["reference_answer"],
                                          tool_trace=trace, final_answer=trajectory.final_answer)
    query_identity = reward_cache_identity(kind="query", config=client.config, question=row["question"],
        reference=row["reference_answer"], trajectory=evidence, messages=query_messages)
    query, query_audit = cached_request(client, directory, query_identity, query_messages, parse_query_response)
    total = compose_reward(format_reward(trajectory), float(verdict == "correct"), query.score)
    return {**asdict(total), "accuracy_judge": acc_audit, "query_judge": query_audit,
            "accuracy_reason": reason, "query_reason": query.reason}
