from dataclasses import replace
import math
import random
import pytest
from llm_scheduler_lab import Request, Simulator, SimulatorConfig
from llm_scheduler_lab.cost import AnalyticalCostModel, BatchWork, PiecewiseCostModel, TraceCalibratedCostModel
from llm_scheduler_lab.policies import BatchPlan, FCFSPolicy, ContinuousBatchingPolicy, ChunkedPrefillPolicy
from llm_scheduler_lab.metrics import percentile, request_metrics
from llm_scheduler_lab.workload import WorkloadConfig, generate_workload, read_trace, write_trace, workload_hash


@pytest.mark.parametrize('field,value', [('arrival_ms', float('nan')), ('arrival_ms',float('inf')),
    ('prompt_tokens', True), ('output_tokens', 1.2), ('request_id', -1), ('prompt_tokens',0)])
def test_invalid_requests(field,value):
    data = dict(request_id=0, arrival_ms=0, prompt_tokens=4, output_tokens=2)
    data[field] = value
    with pytest.raises(ValueError): Request(**data)


@pytest.mark.parametrize('field,value', [('max_iterations',0), ('max_tokens_per_iteration',True),
    ('max_resident_requests',0), ('decode_batch_quadratic_ms',-1),
    ('prefill_ms_per_token',float('nan')), ('kv_capacity_tokens',0), ('first_token_source','guess')])
def test_invalid_config(field,value):
    with pytest.raises(ValueError): SimulatorConfig(**{field:value})


def test_hand_computed_first_token_and_single_token_tpot():
    model = AnalyticalCostModel(overhead_ms=1, prefill_ms_per_token=2,
                                decode_ms_per_sequence=3, decode_quadratic_ms=0)
    result = Simulator(cost_model=model).run([Request(0,10,4,3)], FCFSPolicy())
    s = result.states[0]
    assert s.output_timestamps_ms == [19,23,27]
    assert request_metrics(s)['ttft_ms'] == 9
    assert request_metrics(s)['tpot_ms'] == 4
    assert result.metrics['processed_tokens'] == 6
    one = Simulator(cost_model=model).run([Request(0,10,4,1)], FCFSPolicy())
    assert one.states[0].completed_ms == 19
    assert one.metrics['tpot_mean_ms'] is None
    assert one.metrics['tpot_count'] == 0
    assert one.metrics['tpot_undefined_requests'] == 1
    legacy = Simulator(SimulatorConfig(first_token_source='decode'), model).run([Request(0,10,4,1)],FCFSPolicy())
    assert legacy.states[0].first_token_ms == 23


def test_iteration_limit_includes_final_work_not_idle_jumps():
    result = Simulator(SimulatorConfig(max_iterations=2)).run([Request(0,0,1,1),Request(1,1000,1,1)],FCFSPolicy())
    assert len(result.iterations) == 2
    with pytest.raises(RuntimeError):
        Simulator(SimulatorConfig(max_iterations=1)).run([Request(0,0,1,2)],FCFSPolicy())


def test_future_arrivals_cannot_enter_running_iteration():
    m = AnalyticalCostModel(1,1,1,0)
    result = Simulator(cost_model=m).run([Request(0,0,10,2),Request(1,1,3,1)],ContinuousBatchingPolicy())
    assert 1 not in result.iterations[0].prefill_allocations
    assert result.states[1].admitted_ms >= 11
    assert result.states[1].admitted_ms == result.states[1].prefill_started_ms


def test_duplicate_ids_and_monolithic_oversize_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        Simulator().run([Request(0,0,1,1),Request(0,1,1,1)], FCFSPolicy())
    for policy in (FCFSPolicy(), ContinuousBatchingPolicy()):
        with pytest.raises(ValueError,match='hard token budget'):
            Simulator(SimulatorConfig(max_tokens_per_iteration=4)).run([Request(0,0,8,2)],policy)
    result=Simulator(SimulatorConfig(max_tokens_per_iteration=4)).run([Request(0,0,8,2)],ChunkedPrefillPolicy(4))
    assert result.states[0].completed


class BrokenPolicy:
    name='broken'
    monolithic=False
    def __init__(self, plan): self.output=plan
    def plan(self,*args): return self.output


@pytest.mark.parametrize('plan,error', [
    (BatchPlan(), RuntimeError),
    (BatchPlan({999:1}),ValueError),
    (BatchPlan({0:1.5}),ValueError),
    (BatchPlan({0:1},[0]),ValueError),
    (BatchPlan({},[0,0]),ValueError),
    (BatchPlan({0:5}),ValueError),
    (BatchPlan({},[0]),ValueError),
    (BatchPlan({0:3,1:3}),ValueError),
])
def test_invalid_plans_rejected(plan,error):
    with pytest.raises(error):
        Simulator(SimulatorConfig(max_tokens_per_iteration=4,max_prefill_requests_per_iteration=2)).run(
            [Request(0,0,4,2),Request(1,0,4,2)],BrokenPolicy(plan))


def test_combined_sequence_limit():
    with pytest.raises(ValueError,match='sequence budget'):
        Simulator(SimulatorConfig(max_batch_size=1,max_prefill_requests_per_iteration=2)).run(
            [Request(0,0,2,1),Request(1,0,2,1)],BrokenPolicy(BatchPlan({0:2,1:2})))


@pytest.mark.parametrize('seed',range(12))
def test_randomized_conservation_and_capacities(seed):
    rng=random.Random(seed)
    rows=[Request(i,rng.randrange(20),rng.randrange(1,65),rng.randrange(1,9)) for i in range(20)]
    cfg=SimulatorConfig(max_batch_size=5,max_tokens_per_iteration=80,max_resident_requests=7,
                        max_prefill_requests_per_iteration=3)
    for model in [AnalyticalCostModel(),PiecewiseCostModel(),TraceCalibratedCostModel.synthetic_demo()]:
        for policy in [FCFSPolicy(),ContinuousBatchingPolicy(),ChunkedPrefillPolicy(8),
                       ChunkedPrefillPolicy(8,priority='prefill_first',prefill_order='aging')]:
            result=Simulator(cfg,model).run(rows,policy)
            assert len(result.states)==len(rows)
            for s in result.states:
                r=s.request
                assert s.completed and s.prefill_remaining==s.decode_remaining==0
                assert len(s.output_timestamps_ms)==r.output_tokens
                assert s.decode_tokens_emitted==r.output_tokens-1
                assert r.arrival_ms<=s.admitted_ms==s.prefill_started_ms<s.prefill_finished_ms
                assert s.first_token_ms==s.prefill_finished_ms<=s.completed_ms
                assert sum(it.prefill_allocations.get(r.request_id,0) for it in result.iterations)==r.prompt_tokens
            for it in result.iterations:
                assert it.prefill_tokens+it.decode_sequences<=cfg.max_tokens_per_iteration
                assert len(it.prefill_allocations)+it.decode_sequences<=cfg.max_batch_size
                assert it.resident_requests<=cfg.max_resident_requests
                assert not set(it.prefill_allocations)&set(it.decode_request_ids)
            assert result.metrics['server_busy_fraction']<=1
            assert 0<result.metrics['batch_slot_occupancy']<=1


def test_kv_reservation_is_hard_and_does_not_use_actual_output_for_admission():
    cfg=SimulatorConfig(kv_capacity_tokens=28,output_token_reservation=8,max_resident_requests=10)
    rows=[Request(i,0,6,3) for i in range(10)]
    r=Simulator(cfg).run(rows,ChunkedPrefillPolicy(2))
    assert r.metrics['max_reserved_kv_tokens']<=28
    assert r.metrics['max_resident_requests']<=2
    with pytest.raises(ValueError): Simulator(cfg).run([Request(0,0,6,9)],FCFSPolicy())
    with pytest.raises(ValueError): Simulator(cfg).run([Request(0,0,21,3)],FCFSPolicy())


def test_closed_loop_arrivals_follow_completions():
    rows=[Request(i,0,2,2) for i in range(5)]
    r=Simulator().run(rows,FCFSPolicy(),concurrency=2,think_time_ms=7)
    assert r.workload_mode=='closed_loop'
    assert r.states[0].request.arrival_ms==r.states[1].request.arrival_ms==0
    assert r.states[2].request.arrival_ms==pytest.approx(r.states[0].completed_ms+7)
    assert all(s.completed for s in r.states)


def test_decode_round_robin_when_token_capacity_is_small():
    cfg=SimulatorConfig(max_tokens_per_iteration=1,max_batch_size=1,max_resident_requests=4)
    r=Simulator(cfg).run([Request(i,0,1,6) for i in range(4)],ChunkedPrefillPolicy(1,priority='prefill_first'))
    assert all(s.completed for s in r.states)
    assert [it.decode_request_ids[0] for it in r.iterations if it.decode_request_ids][:4]==[0,1,2,3]


def test_trace_interpolation_and_rejection(tmp_path):
    rows=[dict(prefill_tokens=p,decode_sequences=d,duration_ms=1+p+2*d) for p in [0,10] for d in [0,4]]
    model=TraceCalibratedCostModel(rows,'synthetic','affine oracle')
    assert model.duration_ms(BatchWork(5,2))==10
    with pytest.raises(ValueError): model.duration_ms(BatchWork(11,1))
    with pytest.raises(ValueError): TraceCalibratedCostModel(rows[:-1],'synthetic','missing corner')
    with pytest.raises(ValueError): TraceCalibratedCostModel(rows+rows[:1],'synthetic','duplicate')
    import json
    p=tmp_path/'grid.json';p.write_text(json.dumps(dict(rows=rows,source_kind='synthetic',description='test')))
    assert TraceCalibratedCostModel.from_file(p).duration_ms(BatchWork(5,2))==10


def test_context_pairs_are_independent_of_chunk_partition():
    cfg=SimulatorConfig(max_tokens_per_iteration=128)
    class Recording(AnalyticalCostModel):
        work=[]
        def duration_ms(self,w): self.work.append(w);return super().duration_ms(w)
    m=Recording();m.work.clear()
    Simulator(cfg,m).run([Request(0,0,100,1)],ChunkedPrefillPolicy(7))
    assert sum(w.prefill_attention_pairs for w in m.work)==5050


def test_percentiles_validate_even_on_empty_samples():
    assert percentile([],0.5) is None
    assert percentile([0,10],0.95)==9.5
    with pytest.raises(ValueError): percentile([],2)
    with pytest.raises(ValueError): percentile([float('nan')],0.5)


@pytest.mark.parametrize('mode',['poisson','burst','fixed_concurrency','head_of_line'])
def test_workload_roundtrip(mode,tmp_path):
    cfg=WorkloadConfig(n_requests=20,arrival_mode=mode,regime='mixed')
    a=generate_workload(cfg)
    assert a==generate_workload(cfg)
    path=tmp_path/'trace.csv';write_trace(path,a)
    assert read_trace(path)==a
    assert workload_hash(read_trace(path))==workload_hash(a)


def test_independent_arrival_and_length_streams():
    cfg=WorkloadConfig(n_requests=30,seed=12,regime='mixed')
    a=generate_workload(cfg)
    for mode in ['poisson','burst','fixed_concurrency']:
        b=generate_workload(replace(cfg,arrival_mode=mode,arrival_rate_rps=100))
        assert [(r.prompt_tokens,r.output_tokens) for r in a]==[(r.prompt_tokens,r.output_tokens) for r in b]


@pytest.mark.parametrize('row',['0,nan,2,3','0,0,1.5,3','0,0,2,0','0,0,2','0,0,2,3,extra'])
def test_malformed_traces(row,tmp_path):
    p=tmp_path/'bad.csv';p.write_text('request_id,arrival_ms,prompt_tokens,output_tokens\n'+row+'\n')
    with pytest.raises(ValueError):read_trace(p)


def test_empty_workload_and_idle_time():
    assert Simulator().run([],FCFSPolicy()).metrics['requests']==0
    r=Simulator().run([Request(0,0,1,1),Request(1,1000,1,1)],FCFSPolicy())
    assert r.metrics['server_busy_fraction']<0.01
    assert r.metrics['request_throughput_rps']<2
