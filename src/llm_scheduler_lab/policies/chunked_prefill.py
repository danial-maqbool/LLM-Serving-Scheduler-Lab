from __future__ import annotations
from dataclasses import dataclass
from .base import BatchPlan, ScheduleView, mixed_plan
from ..models import SimulatorConfig, integer


@dataclass(frozen=True, slots=True)
class ChunkedPrefillPolicy:
    chunk_size: int = 512
    priority: str = 'decode_first'
    prefill_order: str = 'fcfs'
    name = 'chunked_prefill'
    monolithic = False

    def __post_init__(self) -> None:
        integer('chunk_size', self.chunk_size)

    def plan(self, now_ms: float, view: ScheduleView, config: SimulatorConfig) -> BatchPlan:
        return mixed_plan(view, config, self.chunk_size, self.priority, self.prefill_order)
