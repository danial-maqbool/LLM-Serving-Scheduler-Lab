from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from ..models import RequestState, SimulatorConfig


@dataclass(slots=True)
class BatchPlan:
    prefill_tokens: dict[int, int] = field(default_factory=dict)
    decode_request_ids: list[int] = field(default_factory=list)

    @property
    def total_prefill_tokens(self) -> int:
        return sum(self.prefill_tokens.values())


class SchedulingPolicy(Protocol):
    name: str

    def plan(
        self,
        now_ms: float,
        states: Sequence[RequestState],
        config: SimulatorConfig,
    ) -> BatchPlan:
        ...


def active(states: Sequence[RequestState]) -> list[RequestState]:
    return [s for s in states if s.admitted_ms is not None and not s.completed]


def by_id(states: Sequence[RequestState]) -> Mapping[int, RequestState]:
    return {s.request.request_id: s for s in states}
