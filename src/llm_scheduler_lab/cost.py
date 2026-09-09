"""Policy-independent synthetic cost models and explicit trace interpolation."""
from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .models import SimulatorConfig, finite


@dataclass(frozen=True, slots=True)
class BatchWork:
    prefill_tokens: int = 0
    decode_sequences: int = 0
    prefill_requests: int = 0
    prefill_attention_pairs: int = 0
    decode_context_tokens: int = 0


class CostModel(Protocol):
    def duration_ms(self, work: BatchWork) -> float: ...
    def metadata(self) -> dict: ...


@dataclass(frozen=True, slots=True)
class AnalyticalCostModel:
    overhead_ms: float = 0.10
    prefill_ms_per_token: float = 0.018
    decode_ms_per_sequence: float = 0.22
    decode_quadratic_ms: float = 0.003
    attention_ms_per_pair: float = 0.0
    context_ms_per_token: float = 0.0

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            finite(key, value)
        if self.prefill_ms_per_token == 0 or self.decode_ms_per_sequence == 0:
            raise ValueError('prefill and decode coefficients must be positive')

    @classmethod
    def from_config(cls, config: SimulatorConfig) -> AnalyticalCostModel:
        return cls(config.fixed_iteration_overhead_ms, config.prefill_ms_per_token,
                   config.decode_ms_per_sequence, config.decode_batch_quadratic_ms)

    def duration_ms(self, work: BatchWork) -> float:
        return (self.overhead_ms + work.prefill_tokens * self.prefill_ms_per_token
                + work.decode_sequences * self.decode_ms_per_sequence
                + work.decode_sequences ** 2 * self.decode_quadratic_ms
                + work.prefill_attention_pairs * self.attention_ms_per_pair
                + work.decode_context_tokens * self.context_ms_per_token)

    def metadata(self) -> dict:
        return {'name': 'analytical', 'source_kind': 'synthetic', 'parameters': asdict(self)}


@dataclass(frozen=True, slots=True)
class PiecewiseCostModel:
    overhead_ms: float = 0.15
    prefill_launch_ms: float = 0.10
    prefill_small_ms: float = 0.025
    prefill_large_ms: float = 0.010
    prefill_knee: int = 256
    decode_shared_ms: float = 0.60
    decode_small_ms: float = 0.040
    decode_large_ms: float = 0.10
    decode_knee: int = 16
    decode_compute_ms: float = 0.020
    prefill_memory_ms: float = 0.001
    interference: float = 0.25
    attention_ms_per_pair: float = 0.0000001
    context_ms_per_token: float = 0.000005

    def __post_init__(self) -> None:
        from .models import integer
        for key, value in asdict(self).items():
            finite(key, value)
        integer('prefill_knee', self.prefill_knee)
        integer('decode_knee', self.decode_knee)
        if not 0 <= self.interference <= 1:
            raise ValueError('interference must be in [0, 1]')
        if self.prefill_small_ms <= 0 or self.prefill_large_ms <= 0 or self.decode_shared_ms <= 0:
            raise ValueError('nonzero service costs are required')

    def duration_ms(self, work: BatchWork) -> float:
        p, d = work.prefill_tokens, work.decode_sequences
        compute = (min(p, self.prefill_knee) * self.prefill_small_ms
                   + max(0, p - self.prefill_knee) * self.prefill_large_ms
                   + work.prefill_attention_pairs * self.attention_ms_per_pair)
        memory = (self.decode_shared_ms * bool(d)
                  + min(d, self.decode_knee) * self.decode_small_ms
                  + max(0, d - self.decode_knee) * self.decode_large_ms
                  + work.decode_context_tokens * self.context_ms_per_token)
        # A roofline-inspired proxy, not an assertion of real kernel overlap.
        return (self.overhead_ms + work.prefill_requests * self.prefill_launch_ms
                + max(compute + d * self.decode_compute_ms, memory + p * self.prefill_memory_ms)
                + self.interference * min(compute, memory))

    def metadata(self) -> dict:
        return {'name': 'piecewise', 'source_kind': 'synthetic', 'parameters': asdict(self)}


class TraceCalibratedCostModel:
    """Bilinear interpolation on a complete (prefill tokens, decode sequences) grid.

    Context and prefill-request count are not axes. This limitation is explicit.
    No extrapolation, nearest-neighbor substitution, or silent fallback is allowed.
    """
    def __init__(self, rows: list[dict], source_kind: str, description: str) -> None:
        from .models import integer
        if source_kind not in ('synthetic', 'measured') or not description.strip():
            raise ValueError('trace provenance is required')
        self.source_kind, self.description = source_kind, description
        self.table: dict[tuple[int, int], float] = {}
        for row in rows:
            p, d, t = row['prefill_tokens'], row['decode_sequences'], row['duration_ms']
            integer('prefill_tokens', p, 0); integer('decode_sequences', d, 0)
            finite('duration_ms', t, positive=True)
            if (p, d) in self.table:
                raise ValueError('duplicate trace coordinate')
            self.table[p, d] = float(t)
        self.p = sorted({p for p, _ in self.table})
        self.d = sorted({d for _, d in self.table})
        if len(self.p) < 2 or len(self.d) < 2 or len(self.table) != len(self.p) * len(self.d):
            raise ValueError('trace must form a complete grid with at least two points per axis')

    @classmethod
    def from_file(cls, path: str | Path) -> TraceCalibratedCostModel:
        path = Path(path)
        if path.suffix.lower() == '.csv':
            meta = json.loads(path.with_suffix('.metadata.json').read_text())
            with path.open(newline='') as handle:
                rows = [{'prefill_tokens': int(r['prefill_tokens']),
                         'decode_sequences': int(r['decode_sequences']),
                         'duration_ms': float(r['duration_ms'])} for r in csv.DictReader(handle)]
        else:
            meta = json.loads(path.read_text()); rows = meta['rows']
        return cls(rows, meta['source_kind'], meta['description'])

    @classmethod
    def synthetic_demo(cls) -> TraceCalibratedCostModel:
        model = AnalyticalCostModel()
        rows = [dict(prefill_tokens=p, decode_sequences=d,
                     duration_ms=model.duration_ms(BatchWork(p, d, int(p > 0))))
                for p in (0, 32, 128, 512, 2048, 8192, 16384)
                for d in (0, 1, 4, 8, 16, 32)]
        return cls(rows, 'synthetic', 'Analytical grid fixture; not hardware calibration')

    @staticmethod
    def _bracket(axis: list[int], value: int) -> tuple[int, int, float]:
        if value < axis[0] or value > axis[-1]:
            raise ValueError('batch outside trace range; extrapolation disabled')
        i = min(max(1, bisect.bisect_left(axis, value)), len(axis) - 1)
        lo, hi = axis[i - 1], axis[i]
        return lo, hi, (value - lo) / (hi - lo)

    def duration_ms(self, work: BatchWork) -> float:
        p0, p1, x = self._bracket(self.p, work.prefill_tokens)
        d0, d1, y = self._bracket(self.d, work.decode_sequences)
        return ((1-x)*(1-y)*self.table[p0, d0] + x*(1-y)*self.table[p1, d0]
                + (1-x)*y*self.table[p0, d1] + x*y*self.table[p1, d1])

    def metadata(self) -> dict:
        return {'name': 'trace', 'source_kind': self.source_kind, 'description': self.description,
                'rows': [dict(prefill_tokens=p, decode_sequences=d, duration_ms=t)
                         for (p, d), t in sorted(self.table.items())],
                'limitations': 'Two axes only; no context-length or prompt-count calibration'}


def make_cost_model(spec: dict | None, config: SimulatorConfig) -> CostModel:
    spec = dict(spec or {'name': 'analytical'})
    name = spec.pop('name', 'analytical')
    if name == 'analytical':
        return AnalyticalCostModel(**spec) if spec else AnalyticalCostModel.from_config(config)
    if name == 'piecewise':
        return PiecewiseCostModel(**spec)
    if name == 'trace':
        return TraceCalibratedCostModel.from_file(spec['path']) if 'path' in spec else TraceCalibratedCostModel.synthetic_demo()
    raise ValueError(f'unknown cost model: {name}')
