from __future__ import annotations

from dataclasses import dataclass

from ..models import RequestState, SimulatorConfig
from .base import BatchPlan, active


@dataclass(slots=True)
class ChunkedPrefillPolicy:
    chunk_size: int = 512
    name: str = "chunked_prefill"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def plan(self, now_ms: float, states: list[RequestState], config: SimulatorConfig) -> BatchPlan:
        candidates = active(states)
        ready = sorted(
            (s for s in candidates if s.ready_to_decode),
            key=lambda s: (s.request.arrival_ms, s.request.request_id),
        )[: config.max_batch_size]
        decode_ids = [s.request.request_id for s in ready]

        token_budget = max(0, config.max_tokens_per_iteration - len(decode_ids))
        slot_budget = max(0, config.max_batch_size - len(decode_ids))
        waiting = sorted(
            (s for s in candidates if s.prefill_remaining > 0),
            key=lambda s: (s.request.arrival_ms, s.request.request_id),
        )
        prefill: dict[int, int] = {}
        for state in waiting:
            if token_budget <= 0 or slot_budget <= 0:
                break
            take = min(self.chunk_size, state.prefill_remaining, token_budget)
            if take <= 0:
                continue
            prefill[state.request.request_id] = take
            token_budget -= take
            slot_budget -= 1
        return BatchPlan(prefill_tokens=prefill, decode_request_ids=decode_ids)
