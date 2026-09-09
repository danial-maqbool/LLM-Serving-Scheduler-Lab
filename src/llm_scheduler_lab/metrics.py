from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Iterable, Sequence

from .models import RequestState


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def request_metrics(state: RequestState) -> dict[str, float]:
    r = state.request
    if state.first_token_ms is None or state.completed_ms is None:
        raise ValueError(f"request {r.request_id} did not complete")
    ttft = state.first_token_ms - r.arrival_ms
    e2e = state.completed_ms - r.arrival_ms
    if r.output_tokens <= 1:
        tpot = 0.0
    else:
        tpot = (state.completed_ms - state.first_token_ms) / (r.output_tokens - 1)
    prefill_wait = (state.prefill_started_ms or r.arrival_ms) - r.arrival_ms
    return {
        "request_id": float(r.request_id),
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": e2e,
        "prefill_wait_ms": prefill_wait,
    }


def summarize(states: Sequence[RequestState], iterations: Sequence[object]) -> dict[str, float]:
    if not states:
        return {}
    rows = [request_metrics(s) for s in states]
    ttft = [r["ttft_ms"] for r in rows]
    tpot = [r["tpot_ms"] for r in rows]
    e2e = [r["e2e_ms"] for r in rows]
    prefill_wait = [r["prefill_wait_ms"] for r in rows]
    durations = [float(getattr(it, "duration_ms")) for it in iterations]
    start = min(s.request.arrival_ms for s in states)
    end = max(float(s.completed_ms or start) for s in states)
    wall_s = max((end - start) / 1000.0, 1e-12)
    output_tokens = sum(s.request.output_tokens for s in states)
    total_tokens = sum(s.request.prompt_tokens + s.request.output_tokens for s in states)
    duration_mean = fmean(durations) if durations else 0.0
    duration_std = pstdev(durations) if len(durations) > 1 else 0.0
    return {
        "requests": float(len(states)),
        "wall_time_ms": end - start,
        "request_throughput_rps": len(states) / wall_s,
        "output_token_throughput_tps": output_tokens / wall_s,
        "total_token_throughput_tps": total_tokens / wall_s,
        "ttft_mean_ms": fmean(ttft),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_mean_ms": fmean(tpot),
        "tpot_p95_ms": percentile(tpot, 0.95),
        "e2e_mean_ms": fmean(e2e),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "prefill_wait_mean_ms": fmean(prefill_wait),
        "iteration_mean_ms": duration_mean,
        "iteration_std_ms": duration_std,
        "iteration_cv": duration_std / duration_mean if duration_mean else 0.0,
        "iterations": float(len(iterations)),
    }
