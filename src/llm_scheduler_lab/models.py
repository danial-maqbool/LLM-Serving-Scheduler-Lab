from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class Request:
    request_id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms must be non-negative")
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")


@dataclass(slots=True)
class RequestState:
    request: Request
    prefill_remaining: int = field(init=False)
    decode_remaining: int = field(init=False)
    admitted_ms: Optional[float] = None
    prefill_started_ms: Optional[float] = None
    prefill_finished_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    completed_ms: Optional[float] = None
    decode_tokens_emitted: int = 0
    decode_service_ms: float = 0.0

    def __post_init__(self) -> None:
        self.prefill_remaining = self.request.prompt_tokens
        self.decode_remaining = self.request.output_tokens

    @property
    def ready_to_decode(self) -> bool:
        return self.prefill_remaining == 0 and self.decode_remaining > 0

    @property
    def completed(self) -> bool:
        return self.decode_remaining == 0 and self.completed_ms is not None


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    fixed_iteration_overhead_ms: float = 0.10
    prefill_ms_per_token: float = 0.018
    decode_ms_per_sequence: float = 0.22
    decode_batch_quadratic_ms: float = 0.003
    max_batch_size: int = 32
    max_tokens_per_iteration: int = 2048
    max_iterations: int = 1_000_000

    def __post_init__(self) -> None:
        if self.fixed_iteration_overhead_ms < 0:
            raise ValueError("fixed_iteration_overhead_ms must be non-negative")
        if self.prefill_ms_per_token <= 0 or self.decode_ms_per_sequence <= 0:
            raise ValueError("timing coefficients must be positive")
        if self.max_batch_size <= 0 or self.max_tokens_per_iteration <= 0:
            raise ValueError("capacity values must be positive")
