"""Gate C live fatal counter, using shared tool error attribution/interaction."""
from __future__ import annotations

import time

from opensearch_vl_repro.rl.fatal import classify_turn
from opensearch_vl_repro.rl.run_state import interrupt_reason_for, PROVIDER_ERROR_INTERRUPTS
from opensearch_vl_repro.rl.workflow_adapter import RLWorkflowAdapter
from opensearch_vl_repro.rl.workflow_types import RLInfrastructureError


class ProviderInterruption(RLInfrastructureError):
    def __init__(self, error_type, *, judge=False, detail=""):
        self.error_type = error_type
        self.reason = interrupt_reason_for(error_type, judge=judge)
        super().__init__(f"{self.reason}: {detail}")


class LiveRLWorkflowAdapter(RLWorkflowAdapter):
    def initialize_episode(self, **kwargs):
        self.consecutive_errors = 0
        self.fatal_turn = self.fatal_step = None
        self.infrastructure_failure = None
        return super().initialize_episode(**kwargs)

    def observe_turn(self, turn):
        error = turn.metadata.get("error_type") or turn.error
        if turn.status != "success" and error in PROVIDER_ERROR_INTERRUPTS:
            self.infrastructure_failure = ProviderInterruption(error, detail=turn.observation or "")
            raise self.infrastructure_failure
        kind = classify_turn(turn)  # unknown failures abort, never reward=0
        if kind in {"success", "neutral"}:
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
        if self.consecutive_errors == 3:
            self.fatal_turn = len(self.state().turns) - 1
            self.fatal_step = len(self.state().assistant_outputs) - 1
            self.state().status = "fatal"

    def handle_model_output(self, output):
        parsed = super().handle_model_output(output)
        if parsed.kind == "malformed_tool_call":
            self.observe_turn(self.state().turns[-1])
        return parsed

    def apply_tool_calls(self, parsed):
        for call in parsed.tool_calls:
            if self.state().status == "fatal":
                break  # never execute a fourth call after the third failure
            started = time.perf_counter()
            result = self.execute_tool_call(call)
            turn = self.commit_tool_result(call=call, result=result, assistant_output=parsed.raw_text,
                                          tool_latency_seconds=time.perf_counter() - started)
            self.observe_turn(turn)

    def fatal_metadata(self, *, termination, step_count):
        fatal = self.fatal_step is not None or termination in {"max_turns_exceeded", "max_response_length_exceeded"}
        cutoff = self.fatal_step if self.fatal_step is not None else step_count - 1
        return {"fatal": fatal, "fatal_turn": self.fatal_turn,
                "fatal_step": cutoff if fatal else None,
                "cutoff_policy": "include_entire_third-error_generated_response; mask_later_responses"}
