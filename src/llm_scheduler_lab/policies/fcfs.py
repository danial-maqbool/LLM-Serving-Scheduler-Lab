from __future__ import annotations

from ..models import RequestState, SimulatorConfig
from .base import BatchPlan, active


class FCFSPolicy:
    """Serve one request at a time, oldest arrival first."""

    name = "fcfs"

    def plan(self, now_ms: float, states: list[RequestState], config: SimulatorConfig) -> BatchPlan:
        candidates = sorted(
            active(states),
            key=lambda s: (s.request.arrival_ms, s.request.request_id),
        )
        if not candidates:
            return BatchPlan()
        state = candidates[0]
        if state.prefill_remaining > 0:
            return BatchPlan(prefill_tokens={state.request.request_id: state.prefill_remaining})
        if state.decode_remaining > 0:
            return BatchPlan(decode_request_ids=[state.request.request_id])
        return BatchPlan()
