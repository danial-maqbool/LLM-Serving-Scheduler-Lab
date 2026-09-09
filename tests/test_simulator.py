from llm_scheduler_lab.models import Request, SimulatorConfig
from llm_scheduler_lab.policies import ChunkedPrefillPolicy, ContinuousBatchingPolicy, FCFSPolicy
from llm_scheduler_lab.simulator import Simulator


def tiny_workload():
    return [
        Request(0, 0.0, 100, 5),
        Request(1, 1.0, 400, 6),
        Request(2, 2.0, 50, 4),
    ]


def test_all_policies_complete_every_request():
    cfg = SimulatorConfig(max_tokens_per_iteration=2048, max_batch_size=8)
    policies = [FCFSPolicy(), ContinuousBatchingPolicy(), ChunkedPrefillPolicy(64)]
    for policy in policies:
        result = Simulator(cfg).run(tiny_workload(), policy)
        assert all(s.completed for s in result.states)
        assert result.metrics["requests"] == 3
        assert result.metrics["ttft_p95_ms"] >= 0


def test_chunking_bounds_prefill_work_per_iteration():
    cfg = SimulatorConfig(max_tokens_per_iteration=2048, max_batch_size=8)
    result = Simulator(cfg).run([Request(0, 0.0, 1000, 2)], ChunkedPrefillPolicy(64))
    assert max(i.prefill_tokens for i in result.iterations) <= 64


def test_monolithic_prefill_can_create_longer_iterations_than_chunking():
    cfg = SimulatorConfig(max_tokens_per_iteration=2048, max_batch_size=8)
    reqs = [Request(0, 0.0, 1000, 4), Request(1, 0.0, 20, 4)]
    mono = Simulator(cfg).run(reqs, ContinuousBatchingPolicy())
    chunked = Simulator(cfg).run(reqs, ChunkedPrefillPolicy(64))
    assert max(i.duration_ms for i in mono.iterations) > max(i.duration_ms for i in chunked.iterations)
