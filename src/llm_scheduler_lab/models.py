"""Validated request and resource models. All times are simulated milliseconds."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def finite(name: str, value: float, minimum: float = 0.0, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < minimum or (positive and value == minimum):
        raise ValueError(f'{name} is outside its permitted range')


@dataclass(frozen=True, slots=True)
class Request:
    request_id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        integer('request_id', self.request_id, 0)
        finite('arrival_ms', self.arrival_ms)
        integer('prompt_tokens', self.prompt_tokens)
        integer('output_tokens', self.output_tokens)


@dataclass(slots=True)
class RequestState:
    request: Request
    prefill_remaining: int = field(init=False)
    decode_remaining: int = field(init=False)
    admitted_ms: float | None = None
    prefill_started_ms: float | None = None
    prefill_finished_ms: float | None = None
    first_token_ms: float | None = None
    completed_ms: float | None = None
    output_timestamps_ms: list[float] = field(default_factory=list)
    decode_tokens_emitted: int = 0
    decode_service_ms: float = 0.0
    prefill_service_ms: float = 0.0
    last_service_end_ms: float | None = None
    max_service_gap_ms: float = 0.0

    def __post_init__(self) -> None:
        self.prefill_remaining = self.request.prompt_tokens
        self.decode_remaining = self.request.output_tokens

    @property
    def ready_to_decode(self) -> bool:
        return self.prefill_remaining == 0 and self.decode_remaining > 0

    @property
    def completed(self) -> bool:
        return self.completed_ms is not None

    def emit(self, timestamp_ms: float, *, decode: bool) -> None:
        self.output_timestamps_ms.append(timestamp_ms)
        self.decode_remaining -= 1
        self.decode_tokens_emitted += int(decode)
        if self.first_token_ms is None:
            self.first_token_ms = timestamp_ms
        if self.decode_remaining == 0:
            self.completed_ms = timestamp_ms


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    # Retained for the starter's analytical timing model.
    fixed_iteration_overhead_ms: float = 0.10
    prefill_ms_per_token: float = 0.018
    decode_ms_per_sequence: float = 0.22
    decode_batch_quadratic_ms: float = 0.003
    max_batch_size: int = 32
    max_tokens_per_iteration: int = 16384
    max_resident_requests: int = 32
    max_prefill_requests_per_iteration: int = 1
    max_iterations: int = 1_000_000
    first_token_source: str = 'prefill'
    long_wait_threshold_ms: float = 1000.0
    kv_capacity_tokens: int | None = None
    output_token_reservation: int = 1024

    def __post_init__(self) -> None:
        for name in ('fixed_iteration_overhead_ms', 'decode_batch_quadratic_ms'):
            finite(name, getattr(self, name))
        for name in ('prefill_ms_per_token', 'decode_ms_per_sequence', 'long_wait_threshold_ms'):
            finite(name, getattr(self, name), positive=True)
        for name in ('max_batch_size', 'max_tokens_per_iteration', 'max_resident_requests',
                     'max_prefill_requests_per_iteration', 'max_iterations', 'output_token_reservation'):
            integer(name, getattr(self, name))
        if self.kv_capacity_tokens is not None:
            integer('kv_capacity_tokens', self.kv_capacity_tokens)
        if self.first_token_source not in ('prefill', 'decode'):
            raise ValueError('first_token_source must be prefill or decode')
