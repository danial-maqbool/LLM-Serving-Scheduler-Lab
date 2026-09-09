"""Reproducible workload descriptions and strict CSV replay."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import Request, finite, integer


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
    min_output_tokens: int = 1
    max_output_tokens: int = 1024
    seed: int = 7
    arrival_mode: str = 'poisson'
    regime: str = 'custom'
    burst_size: int = 16
    concurrency: int = 16
    think_time_ms: float = 0.0

    def __post_init__(self) -> None:
        integer('n_requests', self.n_requests, 0)
        integer('seed', self.seed, 0)
        for name in ('prompt_median_tokens', 'output_median_tokens', 'min_prompt_tokens',
                     'max_prompt_tokens', 'min_output_tokens', 'max_output_tokens', 'burst_size', 'concurrency'):
            integer(name, getattr(self, name))
        finite('arrival_rate_rps', self.arrival_rate_rps, positive=True)
        for name in ('prompt_sigma', 'output_sigma', 'think_time_ms'):
            finite(name, getattr(self, name))
        if self.min_prompt_tokens > self.max_prompt_tokens or self.min_output_tokens > self.max_output_tokens:
            raise ValueError('minimum token length exceeds maximum')
        if self.arrival_mode not in ('poisson', 'burst', 'fixed_concurrency', 'head_of_line'):
            raise ValueError('unknown arrival_mode')
        if self.regime not in ('custom', 'short_short', 'long_short', 'short_long', 'mixed'):
            raise ValueError('unknown workload regime')


def _length(rng: random.Random, median: int, sigma: float, lo: int, hi: int) -> int:
    # Saturate before exponentiation for extreme but finite configuration values.
    value = rng.normalvariate(math.log(median), sigma)
    return max(lo, min(hi, round(math.exp(min(math.log(hi), value)))))


def generate_workload(config: WorkloadConfig) -> list[Request]:
    # Changing arrival mode/rate must not alter token lengths for the same seed.
    arrivals = random.Random(config.seed ^ 0x5A17)
    lengths = random.Random(config.seed ^ 0xB47C)
    time_ms = 0.0
    rows = []
    for i in range(config.n_requests):
        if config.arrival_mode == 'poisson' and i:
            time_ms += arrivals.expovariate(config.arrival_rate_rps) * 1000
        elif config.arrival_mode == 'burst':
            time_ms = (i // config.burst_size) * config.burst_size * 1000 / config.arrival_rate_rps
        elif config.arrival_mode == 'head_of_line':
            time_ms = 0.0 if i == 0 else 1.0 + (i - 1) * 0.1
        elif config.arrival_mode == 'fixed_concurrency':
            time_ms = 0.0  # Templates; Simulator releases successors on completions.
        pm, ps = config.prompt_median_tokens, config.prompt_sigma
        om, os = config.output_median_tokens, config.output_sigma
        if config.regime == 'short_short':
            pm, ps, om, os = 96, 0.4, 24, 0.3
        elif config.regime == 'long_short':
            pm, ps, om, os = 4096, 0.5, 24, 0.3
        elif config.regime == 'short_long':
            pm, ps, om, os = 96, 0.4, 256, 0.5
        elif config.regime == 'mixed':
            pm, ps, om, os = (256, 1.4, 64, 1.0) if lengths.random() < 0.8 else (4096, 0.8, 192, 0.7)
        p = _length(lengths, pm, ps, config.min_prompt_tokens, config.max_prompt_tokens)
        o = _length(lengths, om, os, config.min_output_tokens, config.max_output_tokens)
        if config.arrival_mode == 'head_of_line':
            if i == 0:
                p, o = min(32, config.max_prompt_tokens), min(96, config.max_output_tokens)
            elif i == 1:
                p, o = config.max_prompt_tokens, min(8, config.max_output_tokens)
            else:
                p, o = lengths.randint(16, 96), lengths.randint(8, 24)
            p = max(config.min_prompt_tokens, min(config.max_prompt_tokens, p))
            o = max(config.min_output_tokens, min(config.max_output_tokens, o))
        rows.append(Request(i, time_ms, p, o))
    return rows


TRACE_FIELDS = ['request_id', 'arrival_ms', 'prompt_tokens', 'output_tokens']


def write_trace(path: str | Path, requests: list[Request]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in requests)


def read_trace(path: str | Path) -> list[Request]:
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TRACE_FIELDS:
            raise ValueError(f'CSV header must be {TRACE_FIELDS}')
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('ragged CSV row')
                rows.append(Request(int(row['request_id']), float(row['arrival_ms']),
                                    int(row['prompt_tokens']), int(row['output_tokens'])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'invalid trace row {line}: {exc}') from exc
    if len({r.request_id for r in rows}) != len(rows):
        raise ValueError('duplicate trace IDs')
    return sorted(rows, key=lambda r: (r.arrival_ms, r.request_id))


def workload_hash(requests: list[Request]) -> str:
    data = [asdict(r) for r in sorted(requests, key=lambda r: (r.arrival_ms, r.request_id))]
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
