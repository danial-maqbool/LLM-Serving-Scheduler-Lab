from __future__ import annotations

from ..models import RequestState, SimulatorConfig
from .base import BatchPlan, active


class ContinuousBatchingPolicy:
    """Decode ready requests together; process one monolithic prefill when room exists."""

    name = "continuous_batching"

    def plan(self, now_ms: float, states: list[RequestState], config: SimulatorConfig) -> BatchPlan:
        candidates = active(states)
        ready = sorted(
            (s for s in candidates if s.ready_to_decode),
            key=lambda s: (s.request.arrival_ms, s.request.request_id),
        )[: config.max_batch_size]
        decode_ids = [s.request.request_id for s in ready]

        # Monolithic prefill deliberately allows an iteration to exceed the nominal token budget.
        # This models the long-iteration behavior that chunked prefill is intended to mitigate.
        waiting = sorted(
            (s for s in candidates if s.prefill_remaining > 0),
            key=lambda s: (s.request.arrival_ms, s.request.request_id),
        )
        prefill = {}
        if waiting and len(decode_ids) < config.max_batch_size:
            state = waiting[0]
            prefill[state.request.request_id] = state.prefill_remaining

        return BatchPlan(prefill_tokens=prefill, decode_request_ids=decode_ids)
