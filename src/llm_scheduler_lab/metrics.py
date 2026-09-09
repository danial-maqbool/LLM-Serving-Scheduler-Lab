"""Metric definitions distinguish request means, token gaps, and finite windows."""
from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Sequence
from .models import RequestState, SimulatorConfig


def percentile(values: Sequence[float], q: float) -> float | None:
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('q must be in [0, 1]')
    if not values:
        return None
    if any(not math.isfinite(v) for v in values):
        raise ValueError('percentile samples must be finite')
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def distribution(values: Sequence[float], prefix: str) -> dict:
    return {f'{prefix}_count': len(values), f'{prefix}_mean_ms': fmean(values) if values else None,
            **{f'{prefix}_p{p}_ms': percentile(values, p / 100) for p in (50, 95, 99)}}


def request_metrics(state: RequestState) -> dict:
    r = state.request
    if not state.completed or state.first_token_ms is None or state.prefill_started_ms is None:
        raise ValueError('metrics require a completed request')
    times = state.output_timestamps_ms
    return {'request_id': r.request_id, 'arrival_ms': r.arrival_ms,
            'prompt_tokens': r.prompt_tokens, 'output_tokens': r.output_tokens,
            'ttft_ms': state.first_token_ms - r.arrival_ms,
            'tpot_ms': ((state.completed_ms - state.first_token_ms) / (r.output_tokens - 1)
                        if r.output_tokens > 1 else None),
            'e2e_ms': state.completed_ms - r.arrival_ms,
            'prefill_wait_ms': state.prefill_started_ms - r.arrival_ms,
            'decode_service_ms': state.decode_service_ms,
            'decode_elapsed_ms': state.completed_ms - state.prefill_finished_ms,
            'decode_descheduled_ms': max(0.0, state.completed_ms - state.prefill_finished_ms - state.decode_service_ms),
            'max_service_gap_ms': state.max_service_gap_ms,
            'itl_max_ms': max((b-a for a, b in zip(times, times[1:])), default=None)}


def summarize(states: Sequence[RequestState], iterations: Sequence[object],
              config: SimulatorConfig | None = None) -> dict:
    cfg = config or SimulatorConfig()
    if not states:
        return {'requests': 0, 'iterations': 0, 'wall_time_ms': 0.0,
                'request_throughput_rps': None, **distribution([], 'ttft'), **distribution([], 'tpot')}
    rows = [request_metrics(s) for s in states]
    result = {'requests': len(states), 'iterations': len(iterations), 'unfinished_requests': 0}
    for metric in ('ttft', 'tpot', 'e2e', 'prefill_wait', 'decode_service', 'decode_elapsed', 'decode_descheduled'):
        values = [row[f'{metric}_ms'] for row in rows if row[f'{metric}_ms'] is not None]
        result.update(distribution(values, metric))
    gaps = [b-a for s in states for a, b in zip(s.output_timestamps_ms, s.output_timestamps_ms[1:])]
    result.update(distribution(gaps, 'itl'))
    start = min(s.request.arrival_ms for s in states)
    end = max(s.completed_ms for s in states)
    wall_s = (end - start) / 1000
    durations = [it.duration_ms for it in iterations]
    busy_ms = sum(durations)
    inputs = sum(s.request.prompt_tokens for s in states)
    outputs = sum(s.request.output_tokens for s in states)
    processed = inputs + sum(s.decode_tokens_emitted for s in states)
    result.update({'wall_time_ms': end - start, 'busy_time_ms': busy_ms,
                   'request_throughput_rps': len(states) / wall_s,
                   'input_token_throughput_tps': inputs / wall_s,
                   'output_token_throughput_tps': outputs / wall_s,
                   'total_token_throughput_tps': (inputs + outputs) / wall_s,
                   'processed_token_throughput_tps': processed / wall_s,
                   'input_tokens': inputs, 'output_tokens': outputs, 'processed_tokens': processed,
                   'iteration_mean_ms': fmean(durations), 'iteration_std_ms': pstdev(durations),
                   'iteration_cv': pstdev(durations) / fmean(durations),
                   'server_busy_fraction': min(1.0, busy_ms / (end - start)),
                   'batch_slot_occupancy': sum((len(it.prefill_allocations) + it.decode_sequences)
                                               * it.duration_ms for it in iterations) / (busy_ms * cfg.max_batch_size),
                   'max_wait_ms': max(s.max_service_gap_ms for s in states),
                   'starvation_proxy_count': sum(s.max_service_gap_ms > cfg.long_wait_threshold_ms for s in states),
                   'max_resident_requests': max(it.resident_requests for it in iterations),
                   'max_reserved_kv_tokens': max(it.reserved_kv_tokens for it in iterations),
                   'max_queue_depth': max(it.waiting_requests for it in iterations)})
    # Jain fairness of waiting satisfaction. This is not fairness of GPU allocation.
    satisfaction = [1 / (1 + row['prefill_wait_ms'] / cfg.long_wait_threshold_ms) for row in rows]
    result['waiting_satisfaction_jain'] = sum(satisfaction) ** 2 / (len(satisfaction) * sum(v*v for v in satisfaction))
    # Finite arrival-window rate excludes drain. Cohort latencies above include drain.
    arrival_end = max(s.request.arrival_ms for s in states)
    window_ms = arrival_end - start
    in_window = [s for s in states if s.completed_ms <= arrival_end]
    window_outputs = sum(t <= arrival_end for s in states for t in s.output_timestamps_ms)
    result.update({'arrival_window_ms': window_ms,
                   'observed_request_throughput_rps': len(in_window) * 1000 / window_ms if window_ms > 0 else None,
                   'observed_output_throughput_tps': window_outputs * 1000 / window_ms if window_ms > 0 else None,
                   'backlog_at_arrival_end': len(states) - len(in_window),
                   'drain_time_ms': end - arrival_end,
                   'tpot_undefined_requests': sum(s.request.output_tokens == 1 for s in states)})
    return result
