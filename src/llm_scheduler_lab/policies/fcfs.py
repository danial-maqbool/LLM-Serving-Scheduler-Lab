from __future__ import annotations
from .base import BatchPlan, ScheduleView
from ..models import SimulatorConfig


class FCFSPolicy:
    """Non-preemptive request-at-a-time FIFO baseline."""
    name = 'fcfs'
    monolithic = True

    def plan(self, now_ms: float, view: ScheduleView, config: SimulatorConfig) -> BatchPlan:
        candidates = view.residents or view.waiting
        if not candidates:
            return BatchPlan()
        s = candidates[0]
        if s.prefill_remaining:
            return BatchPlan({s.request.request_id: s.prefill_remaining})
        return BatchPlan(decode_request_ids=[s.request.request_id])
