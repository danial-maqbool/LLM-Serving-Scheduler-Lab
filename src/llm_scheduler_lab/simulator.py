"""Discrete-iteration execution with hard resource checks before every batch."""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable

from .cost import AnalyticalCostModel, BatchWork, CostModel
from .models import Request, RequestState, SimulatorConfig, finite, integer
from .policies.base import BatchPlan, ScheduleView, SchedulingPolicy, reservation


@dataclass(frozen=True, slots=True)
class IterationRecord:
    index: int
    start_ms: float
    end_ms: float
    duration_ms: float
    prefill_tokens: int
    decode_sequences: int
    active_requests: int
    waiting_requests: int
    resident_requests: int
    prefill_queue_count: int
    decode_ready_count: int
    reserved_kv_tokens: int
    prefill_allocations: dict[int, int]
    decode_request_ids: tuple[int, ...]
    admitted_ids: tuple[int, ...]
    completed_ids: tuple[int, ...]
    output_tokens: int


@dataclass(slots=True)
class SimulationResult:
    policy_name: str
    states: list[RequestState]
    iterations: list[IterationRecord]
    config: SimulatorConfig
    cost_metadata: dict
    workload_mode: str = 'open_loop'

    @property
    def metrics(self) -> dict:
        from .metrics import summarize
        return summarize(self.states, self.iterations, self.config)


class Simulator:
    def __init__(self, config: SimulatorConfig | None = None, cost_model: CostModel | None = None) -> None:
        self.config = config or SimulatorConfig()
        self.cost_model = cost_model or AnalyticalCostModel.from_config(self.config)

    def run(self, requests: Iterable[Request], policy: SchedulingPolicy, *,
            concurrency: int | None = None, think_time_ms: float = 0.0) -> SimulationResult:
        cfg = self.config
        ordered = sorted(requests, key=lambda r: (r.arrival_ms, r.request_id))
        if len({r.request_id for r in ordered}) != len(ordered):
            raise ValueError('duplicate request IDs')
        if getattr(policy, 'monolithic', False) and any(r.prompt_tokens > cfg.max_tokens_per_iteration for r in ordered):
            raise ValueError('monolithic prompt exceeds hard token budget; increase the common budget or use chunking')
        if cfg.kv_capacity_tokens is not None:
            for r in ordered:
                if r.output_tokens > cfg.output_token_reservation:
                    raise ValueError('output exceeds configured reservation limit')
                if r.prompt_tokens + cfg.output_token_reservation > cfg.kv_capacity_tokens:
                    raise ValueError('request cannot fit the conservative KV reservation')
        finite('think_time_ms', think_time_ms)
        if concurrency is not None:
            integer('concurrency', concurrency)
            if any(r.arrival_ms != 0 for r in ordered):
                raise ValueError('closed-loop input must contain zero-time request templates')
        elif think_time_ms:
            raise ValueError('think time requires closed-loop concurrency')

        states: list[RequestState] = []
        future: list[tuple[float, int, RequestState]] = []
        templates = deque(ordered)

        def release(request: Request) -> None:
            state = RequestState(request)
            states.append(state)
            heapq.heappush(future, (request.arrival_ms, request.request_id, state))

        if concurrency is None:
            while templates:
                release(templates.popleft())
        else:
            for _ in range(min(concurrency, len(templates))):
                release(templates.popleft())
        waiting: deque[RequestState] = deque()
        residents: dict[int, RequestState] = {}
        iterations: list[IterationRecord] = []
        now = future[0][0] if future else 0.0
        while future or waiting or residents:
            while future and future[0][0] <= now:
                waiting.append(heapq.heappop(future)[2])
            if not waiting and not residents:
                now = future[0][0]
                continue
            if len(iterations) >= cfg.max_iterations:
                raise RuntimeError('max_iterations exceeded with unfinished work')
            view = ScheduleView(waiting, tuple(residents.values()))
            plan = policy.plan(now, view, cfg)
            lookup = {s.request.request_id: s for s in view.residents}
            # Only selected waiting IDs need lookup; validation still rejects future IDs.
            selected = set(plan.prefill_tokens) | set(plan.decode_request_ids)
            if plan.prefill_tokens:
                lookup.update((s.request.request_id, s) for s in waiting if s.request.request_id in selected)
            self._validate_plan(plan, lookup, now, view)
            work = self._batch_work(plan, lookup)
            duration = self.cost_model.duration_ms(work)
            finite('iteration duration', duration, positive=True)
            if not math.isfinite(duration) or duration <= 0 or now + duration <= now:
                raise ValueError('cost model must advance finite simulated time')
            end = now + duration
            if not math.isfinite(end):
                raise ValueError('simulated time overflow')
            waiting_count = len(waiting)
            prefill_count = waiting_count + sum(s.prefill_remaining > 0 for s in residents.values())
            decode_count = sum(s.ready_to_decode for s in residents.values())
            active_count = waiting_count + len(residents)
            admitted, completed = [], []
            outputs = 0
            for rid in plan.prefill_tokens:
                s = lookup[rid]
                if s.admitted_ms is None:
                    s.admitted_ms = now
                    residents[rid] = s
                    admitted.append(rid)
            if admitted:
                accepted = set(admitted)
                waiting = deque(s for s in waiting if s.request.request_id not in accepted)
            resident_count = len(residents)
            kv = sum(reservation(s, cfg) for s in residents.values()) if cfg.kv_capacity_tokens else 0
            for rid in selected:
                s = lookup[rid]
                previous_end = s.last_service_end_ms if s.last_service_end_ms is not None else s.request.arrival_ms
                s.max_service_gap_ms = max(s.max_service_gap_ms, now - previous_end)
                s.last_service_end_ms = end
            for rid, tokens in plan.prefill_tokens.items():
                s = lookup[rid]
                if s.prefill_started_ms is None:
                    s.prefill_started_ms = now
                s.prefill_remaining -= tokens
                s.prefill_service_ms += duration
                if s.prefill_remaining == 0:
                    s.prefill_finished_ms = end
                    if cfg.first_token_source == 'prefill':
                        s.emit(end, decode=False)
                        outputs += 1
            for rid in plan.decode_request_ids:
                s = lookup[rid]
                s.decode_service_ms += duration
                s.emit(end, decode=True)
                outputs += 1
            for rid in sorted(selected):
                if lookup[rid].completed:
                    completed.append(rid)
                    del residents[rid]
                    if concurrency is not None and templates:
                        release(replace(templates.popleft(), arrival_ms=end + think_time_ms))
            iterations.append(IterationRecord(
                len(iterations), now, end, duration, work.prefill_tokens, work.decode_sequences,
                active_count, waiting_count, resident_count, prefill_count, decode_count, kv,
                dict(plan.prefill_tokens), tuple(plan.decode_request_ids), tuple(admitted), tuple(completed), outputs))
            now = end
        return SimulationResult(policy.name, sorted(states, key=lambda s: s.request.request_id),
                                iterations, cfg, self.cost_model.metadata(),
                                'closed_loop' if concurrency is not None else 'open_loop')

    def _validate_plan(self, plan: BatchPlan, lookup: dict[int, RequestState],
                       now: float, view: ScheduleView) -> None:
        cfg = self.config
        p, d = set(plan.prefill_tokens), set(plan.decode_request_ids)
        for rid in p | d:
            integer('planned request ID', rid, 0)
        if not p and not d:
            raise RuntimeError('empty plan while eligible work remains')
        if len(d) != len(plan.decode_request_ids) or p & d:
            raise ValueError('requests must occur only once per batch')
        if len(p) + len(d) > cfg.max_batch_size:
            raise ValueError('combined batch exceeds sequence budget')
        if len(p) > cfg.max_prefill_requests_per_iteration:
            raise ValueError('prefill request budget exceeded')
        for rid, tokens in plan.prefill_tokens.items():
            integer('prefill allocation', tokens)
            if rid not in lookup or lookup[rid].completed or lookup[rid].request.arrival_ms > now:
                raise ValueError('prefill references unavailable request')
            if tokens > lookup[rid].prefill_remaining:
                raise ValueError('prefill allocation exceeds remaining prompt')
        for rid in d:
            if rid not in lookup or lookup[rid].admitted_ms is None or not lookup[rid].ready_to_decode:
                raise ValueError('decode references unavailable or unprefilled request')
        if plan.total_prefill_tokens + len(d) > cfg.max_tokens_per_iteration:
            raise ValueError('combined batch exceeds token budget')
        new = [lookup[rid] for rid in p if lookup[rid].admitted_ms is None]
        if len(view.residents) + len(new) > cfg.max_resident_requests:
            raise ValueError('resident request capacity exceeded')
        if cfg.kv_capacity_tokens is not None:
            total = sum(reservation(s, cfg) for s in (*view.residents, *new))
            if total > cfg.kv_capacity_tokens:
                raise ValueError('KV reservation budget exceeded')

    @staticmethod
    def _batch_work(plan: BatchPlan, lookup: dict[int, RequestState]) -> BatchWork:
        pairs = sum(tokens * (lookup[rid].request.prompt_tokens - lookup[rid].prefill_remaining)
                    + tokens * (tokens + 1) // 2 for rid, tokens in plan.prefill_tokens.items())
        context = sum(lookup[rid].request.prompt_tokens + len(lookup[rid].output_timestamps_ms)
                      for rid in plan.decode_request_ids)
        return BatchWork(plan.total_prefill_tokens, len(plan.decode_request_ids),
                         len(plan.prefill_tokens), pairs, context)
