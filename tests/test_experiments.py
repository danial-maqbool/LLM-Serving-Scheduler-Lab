import json
from pathlib import Path
import pytest
from llm_scheduler_lab.experiments import expand_suite,run_suite,read_config
from llm_scheduler_lab.statistics import describe,paired_comparison


def suite():
    return {'schema_version':1,'seeds':[1,2],
            'defaults':{'workload':{'n_requests':4,'prompt_median_tokens':8,'output_median_tokens':3,
                                    'min_prompt_tokens':1,'max_prompt_tokens':16,'max_output_tokens':5}},
            'policies':[{'id':'mono','name':'continuous_batching'},
                        {'id':'chunk_4','name':'chunked_prefill','chunk_size':4}],
            'experiments':[{'id':'smoke','sweep':{'workload.arrival_rate_rps':[2,20]}}]}


def test_expand_and_duplicate_seed_rejection():
    s=suite();assert len(expand_suite(s))==8
    s['seeds']=[1,1]
    with pytest.raises(ValueError):expand_suite(s)


def test_exact_pairing_and_statistics():
    p=paired_comparison({1:2,2:4},{1:4,2:6},resamples=100)
    assert p['mean_delta']==p['ci95_low']==p['ci95_high']==2
    assert p['relative_change_pct']==pytest.approx(100*2/3)
    assert p['paired_dz'] is None
    with pytest.raises(ValueError): paired_comparison({1:1},{2:2})
    assert paired_comparison({1:0},{1:1},resamples=10)['relative_change_pct'] is None
    assert describe([4,4,4],resamples=10)['ci95_low']==4


def test_verified_cache_resume_and_corruption_repair(tmp_path):
    config=tmp_path/'suite.json';config.write_text(json.dumps(suite()))
    out=tmp_path/'results'
    a=run_suite(config,out,allow_dirty=True,quiet=True)
    data=(out/'summary.csv').read_bytes()
    b=run_suite(config,out,allow_dirty=True,quiet=True)
    assert a['trial_count']==b['cached_trials']==8
    assert data==(out/'summary.csv').read_bytes()
    path=next((out/'.runs').glob('*.json'))
    invalid=json.loads(path.read_text());invalid['payload']['summary']['ttft_mean_ms']=123456
    path.write_text(json.dumps(invalid))
    c=run_suite(config,out,allow_dirty=True,quiet=True)
    assert c['repaired_cache_entries']==1 and c['cached_trials']==7
    rows=json.loads((out/'runs.json').read_text())
    assert len(rows)==8


def test_yaml_and_json_expand_identically(tmp_path):
    yaml=pytest.importorskip('yaml')
    p=tmp_path/'suite.yml';p.write_text(yaml.safe_dump(suite()))
    assert expand_suite(read_config(p))==expand_suite(suite())


def test_closed_loop_and_trace_artifacts(tmp_path):
    s=suite();s['experiments']=[{'id':'closed','workload':{'arrival_mode':'fixed_concurrency'},'trace_seeds':[1]}]
    p=tmp_path/'s.json';p.write_text(json.dumps(s))
    out=tmp_path/'out';run_suite(p,out,allow_dirty=True,quiet=True)
    traces=list((out/'traces').glob('*.json'));assert len(traces)==2
    t=json.loads(traces[0].read_text())
    assert t['iterations'] and t['requests']
    assert t['summary']['result_kind']=='SIMULATED'
