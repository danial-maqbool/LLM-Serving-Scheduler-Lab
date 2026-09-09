from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from .metrics import summarize
from .models import Request, RequestState, SimulatorConfig
from .policies.base import BatchPlan, SchedulingPolicy, by_id


@dataclass(frozen=True, slots=True)
class IterationRecord:
    index: int
    start_ms: float
    end_ms: float
    duration_ms: float
    prefill_tokens: int
    decode_sequences: int
    active_requests: int


@dataclass(slots=True)
class SimulationResult:
    policy_name: str
    states: list[RequestState]
    iterations: list[IterationRecord]

    @property
    def metrics(self) -> dict[str, float]:
        return summarize(self.states, self.iterations)


class Simulator:
    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()

    def _iteration_duration(self, plan: BatchPlan) -> float:
        cfg = self.config
        decode_n = len(plan.decode_request_ids)
        return (
            cfg.fixed_iteration_overhead_ms
            + plan.total_prefill_tokens * cfg.prefill_ms_per_token
            + decode_n * cfg.decode_ms_per_sequence
            + (decode_n**2) * cfg.decode_batch_quadratic_ms
        )

    def run(self, requests: Iterable[Request], policy: SchedulingPolicy) -> SimulationResult:
        ordered = sorted(requests, key=lambda r: (r.arrival_ms, r.request_id))
        if not ordered:
            return SimulationResult(policy.name, [], [])

        states = [RequestState(r) for r in ordered]
        lookup = by_id(states)
        now_ms = ordered[0].arrival_ms
        cursor = 0
        iterations: list[IterationRecord] = []

        for iteration_index in range(self.config.max_iterations):
            while cursor < len(states) and states[cursor].request.arrival_ms <= now_ms + 1e-12:
                states[cursor].admitted_ms = now_ms
                cursor += 1

            if all(s.completed for s in states):
                return SimulationResult(policy.name, states, iterations)

            active_states = [s for s in states if s.admitted_ms is not None and not s.completed]
            if not active_states:
                if cursor >= len(states):
                    raise RuntimeError("simulation stalled with no active or future requests")
                now_ms = states[cursor].request.arrival_ms
                continue

            plan = policy.plan(now_ms, states, self.config)
            self._validate_plan(plan, lookup)
            if plan.total_prefill_tokens == 0 and not plan.decode_request_ids:
                raise RuntimeError(f"policy {policy.name!r} produced an empty plan while work remained")

            start_ms = now_ms
            duration_ms = self._iteration_duration(plan)
            end_ms = start_ms + duration_ms

            for request_id, tokens in plan.prefill_tokens.items():
                state = lookup[request_id]
                if state.prefill_started_ms is None:
                    state.prefill_started_ms = start_ms
                state.prefill_remaining -= tokens
                if state.prefill_remaining == 0:
                    state.prefill_finished_ms = end_ms

            for request_id in plan.decode_request_ids:
                state = lookup[request_id]
                state.decode_remaining -= 1
                state.decode_tokens_emitted += 1
                state.decode_service_ms += duration_ms
                if state.first_token_ms is None:
                    state.first_token_ms = end_ms
                if state.decode_remaining == 0:
                    state.completed_ms = end_ms

            iterations.append(
                IterationRecord(
                    index=iteration_index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration_ms,
                    prefill_tokens=plan.total_prefill_tokens,
                    decode_sequences=len(plan.decode_request_ids),
                    active_requests=len(active_states),
                )
            )
            now_ms = end_ms

        raise RuntimeError("max_iterations exceeded; likely scheduler starvation or a simulator bug")

    def _validate_plan(self, plan: BatchPlan, lookup: dict[int, RequestState]) -> None:
        if len(plan.decode_request_ids) != len(set(plan.decode_request_ids)):
            raise ValueError("decode request IDs must be unique within an iteration")
        if len(plan.decode_request_ids) > self.config.max_batch_size:
            raise ValueError("decode batch exceeds max_batch_size")
        for request_id, tokens in plan.prefill_tokens.items():
            if request_id not in lookup:
                raise ValueError(f"unknown prefill request id {request_id}")
            state = lookup[request_id]
            if state.admitted_ms is None or state.completed:
                raise ValueError("cannot prefill an unadmitted/completed request")
            if tokens <= 0 or tokens > state.prefill_remaining:
                raise ValueError("invalid prefill allocation")
        for request_id in plan.decode_request_ids:
            if request_id not in lookup:
                raise ValueError(f"unknown decode request id {request_id}")
            state = lookup[request_id]
            if not state.ready_to_decode:
                raise ValueError("cannot decode before prefill completion")
