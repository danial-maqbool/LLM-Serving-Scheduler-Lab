from __future__ import annotations
from dataclasses import dataclass
from .base import BatchPlan, ScheduleView, mixed_plan
from ..models import SimulatorConfig


@dataclass(frozen=True, slots=True)
class ContinuousBatchingPolicy:
    """Iteration scheduling with whole prompts; not an implementation of vLLM."""
    priority: str = 'decode_first'
    prefill_order: str = 'fcfs'
    name = 'continuous_batching'
    monolithic = True

    def plan(self, now_ms: float, view: ScheduleView, config: SimulatorConfig) -> BatchPlan:
        return mixed_plan(view, config, None, self.priority, self.prefill_order)
