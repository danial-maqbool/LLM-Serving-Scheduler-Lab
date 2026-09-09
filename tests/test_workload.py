from llm_scheduler_lab.workload import WorkloadConfig, generate_workload


def test_workload_is_deterministic_and_sorted():
    cfg = WorkloadConfig(n_requests=10, seed=123)
    a = generate_workload(cfg)
    b = generate_workload(cfg)
    assert a == b
    assert [r.arrival_ms for r in a] == sorted(r.arrival_ms for r in a)
    assert all(r.prompt_tokens > 0 and r.output_tokens > 0 for r in a)
