import pytest
from llm_scheduler_lab import Simulator, Request
from llm_scheduler_lab.cost import AnalyticalCostModel
from llm_scheduler_lab.policies import FCFSPolicy, ChunkedPrefillPolicy, BatchPlan
from llm_scheduler_lab.workload import WorkloadConfig, generate_workload


class BrokenPolicy:
    name = 'broken'
    monolithic = False
    def __init__(self, output): self.output = output
    def plan(self, *args): return self.output


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),0,-1,True])
def test_invalid_cost_duration_is_rejected(bad):
    class Invalid(AnalyticalCostModel):
        def duration_ms(self,w):return bad
    with pytest.raises(ValueError):Simulator(cost_model=Invalid()).run([Request(0,0,1,1)],FCFSPolicy())


def test_boolean_request_id_in_plan_is_rejected():
    with pytest.raises(ValueError):
        Simulator().run([Request(1,0,2,1)],BrokenPolicy(BatchPlan({True:1})))


def test_same_workload_and_policy_are_exactly_replayable():
    rows=generate_workload(WorkloadConfig(n_requests=12,regime='mixed',max_output_tokens=16))
    a=Simulator().run(rows,ChunkedPrefillPolicy(128))
    b=Simulator().run(rows,ChunkedPrefillPolicy(128))
    assert a.metrics==b.metrics
    assert a.iterations==b.iterations
    assert [s.output_timestamps_ms for s in a.states]==[s.output_timestamps_ms for s in b.states]
