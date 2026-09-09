import json
import pytest
from llm_scheduler_lab.experiments import run_suite
from llm_scheduler_lab.analysis import analyze,read_results
from llm_scheduler_lab.plotting import plot_all


def config(tmp_path):
    data={'schema_version':1,'seeds':[1,2],
          'defaults':{'workload':{'n_requests':6,'prompt_median_tokens':4,'output_median_tokens':2,
                                  'max_prompt_tokens':8,'min_prompt_tokens':1,'max_output_tokens':3}},
          'policies':[{'id':'mono','name':'continuous_batching'},
                      {'id':'chunk_4','name':'chunked_prefill','chunk_size':4}],
          'experiments':[{'id':'smoke'}]}
    p=tmp_path/'suite.json';p.write_text(json.dumps(data));return p


def test_analysis_and_plot_smoke(tmp_path):
    pytest.importorskip('matplotlib')
    out=tmp_path/'out';run_suite(config(tmp_path),out,allow_dirty=True,quiet=True)
    analyze(out,out/'stats')
    assert (out/'stats/paired.csv').stat().st_size>0
    files=plot_all(out,out/'figures')
    assert files==['smoke_overview.svg']
    import xml.etree.ElementTree as ET
    root=ET.parse(out/'figures'/files[0]).getroot()
    assert root.tag.endswith('svg')
    assert 'SIMULATED' in (out/'figures'/files[0]).read_text()


def test_artifact_tampering_is_rejected(tmp_path):
    out=tmp_path/'out';run_suite(config(tmp_path),out,allow_dirty=True,quiet=True)
    with (out/'summary.csv').open('a') as h:h.write('corrupt\n')
    with pytest.raises(ValueError,match='checksum'):read_results(out)


def test_dirty_cache_cannot_be_promoted_as_clean(tmp_path,monkeypatch):
    import llm_scheduler_lab.experiments as e
    out=tmp_path/'out';p=config(tmp_path)
    monkeypatch.setattr(e,'git_state',lambda root:{'git_commit':'test-dirty-commit','git_dirty':True})
    e.run_suite(p,out,allow_dirty=True,quiet=True)
    monkeypatch.setattr(e,'git_state',lambda root:{'git_commit':'test-clean-commit','git_dirty':False})
    result=e.run_suite(p,out,quiet=True)
    assert result['cached_trials']==0
    assert result['simulation_commits']==['test-clean-commit']
