"""Main-only no-truncation rollout budget contract; no framework imports."""

CONTEXT_BUDGET_POLICY = "remaining-context-no-truncation-v1"


class ResponseContextBudgetExhausted(Exception):
    """No usable generation space; not a provider or model crash."""

    def __init__(self, diagnostics):
        self.diagnostics = dict(diagnostics)
        super().__init__(f"Main response context budget exhausted: {self.diagnostics}")
