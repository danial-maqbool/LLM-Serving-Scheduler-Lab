from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .models import Request


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    n_requests: int = 200
    arrival_rate_rps: float = 8.0
    prompt_median_tokens: int = 512
    prompt_sigma: float = 0.9
    output_median_tokens: int = 128
    output_sigma: float = 0.55
    min_prompt_tokens: int = 16
    max_prompt_tokens: int = 8192
    min_output_tokens: int = 8
    max_output_tokens: int = 1024
    seed: int = 7


def _clamped_lognormal(rng: random.Random, median: int, sigma: float, lo: int, hi: int) -> int:
    value = int(round(rng.lognormvariate(math.log(median), sigma)))
    return max(lo, min(hi, value))


def generate_workload(config: WorkloadConfig) -> list[Request]:
    if config.n_requests <= 0:
        raise ValueError("n_requests must be positive")
    if config.arrival_rate_rps <= 0:
        raise ValueError("arrival_rate_rps must be positive")
    rng = random.Random(config.seed)
    arrival_ms = 0.0
    requests: list[Request] = []
    for i in range(config.n_requests):
        if i > 0:
            arrival_ms += rng.expovariate(config.arrival_rate_rps) * 1000.0
        requests.append(
            Request(
                request_id=i,
                arrival_ms=arrival_ms,
                prompt_tokens=_clamped_lognormal(
                    rng,
                    config.prompt_median_tokens,
                    config.prompt_sigma,
                    config.min_prompt_tokens,
                    config.max_prompt_tokens,
                ),
                output_tokens=_clamped_lognormal(
                    rng,
                    config.output_median_tokens,
                    config.output_sigma,
                    config.min_output_tokens,
                    config.max_output_tokens,
                ),
            )
        )
    return requests
