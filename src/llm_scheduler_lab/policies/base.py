"""Scheduling plans are allocations, not time estimates."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Protocol, Sequence

from ..models import RequestState, SimulatorConfig


@dataclass(slots=True)
class BatchPlan:
    prefill_tokens: dict[int, int] = field(default_factory=dict)
    decode_request_ids: list[int] = field(default_factory=list)

    @property
    def total_prefill_tokens(self) -> int:
        return sum(self.prefill_tokens.values())


@dataclass(frozen=True, slots=True)
class ScheduleView:
    waiting: Sequence[RequestState]
    residents: Sequence[RequestState]


class SchedulingPolicy(Protocol):
    name: str
    monolithic: bool
    def plan(self, now_ms: float, view: ScheduleView, config: SimulatorConfig) -> BatchPlan: ...


def reservation(state: RequestState, config: SimulatorConfig) -> int:
    return state.request.prompt_tokens + config.output_token_reservation


def mixed_plan(view: ScheduleView, config: SimulatorConfig, chunk_size: int | None,
               priority: str, prefill_order: str) -> BatchPlan:
    if priority not in ('decode_first', 'prefill_first'):
        raise ValueError('unknown scheduling priority')
    if prefill_order not in ('fcfs', 'aging'):
        raise ValueError('unknown prefill order')
    plan = BatchPlan()
    tokens, slots = config.max_tokens_per_iteration, config.max_batch_size
    residents = list(view.residents)
    room = config.max_resident_requests - len(residents)
    kv_left = (config.kv_capacity_tokens - sum(reservation(s, config) for s in residents)
               if config.kv_capacity_tokens is not None else None)
    # Round-robin decoding when fewer token/sequence slots than ready requests exist.
    ready = sorted((s for s in residents if s.ready_to_decode),
                   key=lambda s: (s.last_service_end_ms, s.request.request_id))
    partial = [s for s in residents if s.prefill_remaining > 0]
    waiting = list(islice(view.waiting, config.max_batch_size))
    candidates = partial + waiting
    if prefill_order == 'aging':
        candidates.sort(key=lambda s: (s.last_service_end_ms if s.last_service_end_ms is not None
                                       else s.request.arrival_ms, s.request.request_id))
    else:
        candidates.sort(key=lambda s: (s.request.arrival_ms, s.request.request_id))

    def decodes() -> None:
        nonlocal tokens, slots
        for s in ready[:min(tokens, slots)]:
            plan.decode_request_ids.append(s.request.request_id)
            tokens -= 1; slots -= 1

    def prefills() -> None:
        nonlocal tokens, slots, room, kv_left
        blocked_new = False
        for s in candidates:
            if slots == 0 or tokens == 0 or len(plan.prefill_tokens) >= config.max_prefill_requests_per_iteration:
                break
            new = s.admitted_ms is None
            needed = reservation(s, config)
            if new and (blocked_new or room == 0 or (kv_left is not None and needed > kv_left)):
                blocked_new = True
                continue  # Existing partial prefills must still be able to progress.
            take = min(chunk_size or s.prefill_remaining, s.prefill_remaining, tokens)
            if chunk_size is None and take < s.prefill_remaining:
                # Never violate a hard budget to fit a monolithic prompt.
                break
            plan.prefill_tokens[s.request.request_id] = take
            tokens -= take; slots -= 1
            if new:
                room -= 1
                if kv_left is not None:
                    kv_left -= needed

    if priority == 'decode_first':
        decodes(); prefills()
    else:
        prefills(); decodes()
    return plan
