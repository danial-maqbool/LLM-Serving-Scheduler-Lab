"""Build report tables directly from verified simulation artifacts."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from llm_scheduler_lab.analysis import read_results
from llm_scheduler_lab.statistics import describe, paired_comparison

ROOT = Path(__file__).resolve().parents[1]


def build(folder: Path, destination: Path = ROOT) -> None:
    rows, manifest = read_results(folder)
    if manifest['trial_count'] != 955 or manifest.get('selected_experiment'):
        raise ValueError('the report template requires the complete predefined 955-trial study')
    def pick(experiment, model='piecewise', **filters):
        return [r for r in rows if r['experiment_id'] == experiment and r['cost_id'] == model
                and all((float(r[k]) == float(v) if k == 'offered_load_rps' else str(r[k]) == str(v))
                        for k,v in filters.items())]
    def average(block, metric):
        if not block:
            raise ValueError(f'empty report condition for {metric}')
        values = [float(r[metric]) for r in block if r.get(metric) not in ('', None)]
        return fmean(values) if values else None
    def number(value, digits=2):
        return 'N/A' if value is None else f'{value:,.{digits}f}'
    def interval(block, metric):
        d=describe([float(r[metric]) for r in block if r.get(metric) not in ('', None)])
        return f"{d['mean']:,.2f} [{d['ci95_low']:,.2f}, {d['ci95_high']:,.2f}]"
    def table(headers, data):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                          *['| '+' | '.join(map(str,r))+' |' for r in data]])
    def compare(experiment, model, policy, metric):
        a={int(r['seed']):float(r[metric]) for r in pick(experiment,model,policy_id='mono')}
        b={int(r['seed']):float(r[metric]) for r in pick(experiment,model,policy_id=policy)}
        return paired_comparison(a,b)

    base=[]
    for model in ('analytical','piecewise'):
        for policy in ('fcfs','mono','chunk_128','chunk_512'):
            b=pick('E1',model,policy_id=policy)
            base.append([model,policy,interval(b,'ttft_p95_ms'),number(average(b,'tpot_p95_ms')),
                         number(average(b,'itl_p99_ms')),number(average(b,'request_throughput_rps'))])
    baseline_table=table(['Cost model','Policy','TTFT P95 mean [95% CI], ms','TPOT P95, ms','ITL P99, ms','Cohort requests/s'],base)
    chunks=[]
    for policy in ('mono',*[f'chunk_{n}' for n in (32,64,128,256,512,1024,2048)]):
        b=pick('E2',policy_id=policy)
        chunks.append([policy,*[number(average(b,m)) for m in ('ttft_p95_ms','tpot_p95_ms','itl_p99_ms',
                                                              'request_throughput_rps','iteration_cv')]])
    chunk_table=table(['Policy','TTFT P95, ms','TPOT P95, ms','ITL P99, ms','Cohort requests/s','Iteration CV'],chunks)
    tradeoffs=[]
    for metric in ('ttft_p95_ms','tpot_p95_ms','itl_p99_ms','request_throughput_rps'):
        c=compare('E2','piecewise','chunk_128',metric)
        tradeoffs.append([metric,number(c['mean_delta']),number(c['relative_change_pct'])+'%',
                          f"[{c['relative_ci95_low']:.2f}%, {c['relative_ci95_high']:.2f}%]"])
    tradeoff_table=table(['Metric','Mean paired delta','Relative change','Paired 95% interval'],tradeoffs)
    loads=[]
    for rate in (2,8,16,32,64,128):
        for policy in ('fcfs','mono','chunk_512'):
            b=pick('E3',offered_load_rps=f'{rate}.0',policy_id=policy)
            loads.append([rate,policy,*[number(average(b,m)) for m in ('observed_request_throughput_rps',
                'ttft_p95_ms','tpot_p95_ms','backlog_at_arrival_end','drain_time_ms')]])
    load_table=table(['Offered req/s','Policy','Arrival-window req/s','TTFT P95, ms','TPOT P95, ms','Final-arrival backlog','Drain, ms'],loads)
    regimes=[]
    for regime in ('short_short','long_short','short_long','mixed'):
        for policy in ('fcfs','mono','chunk_512'):
            b=pick('E4',regime=regime,policy_id=policy)
            regimes.append([regime,policy,*[number(average(b,m)) for m in ('ttft_p95_ms','tpot_p95_ms','request_throughput_rps')]])
    regime_table=table(['Regime','Policy','TTFT P95, ms','TPOT P95, ms','Cohort requests/s'],regimes)
    hol=[]
    for model in ('analytical','piecewise'):
        for policy in ('mono','chunk_64','chunk_64_aging'):
            b=pick('E5',model,policy_id=policy)
            hol.append([model,policy,*[number(average(b,m)) for m in ('primer_max_itl_ms','short_prompt_ttft_p95_ms','long_prompt_ttft_p95_ms')]])
    hol_table=table(['Model','Policy','Stream max ITL, ms','Short-prompt TTFT P95, ms','Large-prompt TTFT, ms'],hol)
    fairness=[]
    for policy in ('fcfs','mono','chunk_512','chunk_512_aging','chunk_512_prefill_first'):
        b=pick('E7',policy_id=policy)
        fairness.append([policy,*[number(average(b,m),3) for m in ('waiting_satisfaction_jain','max_wait_ms','starvation_proxy_count')]])
    fairness_table=table(['Policy','Waiting-satisfaction Jain','Maximum wait, ms','Requests with gap >1 s'],fairness)
    sensitivity=[]
    for model in sorted({r['cost_id'] for r in rows if r['experiment_id']=='E8'}):
        sensitivity.append([model,*[number(compare('E8',model,'chunk_512',m)['relative_change_pct'])+'%'
                                     for m in ('ttft_p95_ms','tpot_p95_ms','request_throughput_rps')]])
    sensitivity_table=table(['Cost configuration','TTFT change','TPOT change','Throughput change'],sensitivity)
    capacity=[]
    runs={r['run_id']:r for r in json.loads((folder/'runs.json').read_text(encoding='utf-8'))}
    for residents in (8,32):
        for prefills in (1,4):
            for policy in ('mono','chunk_512'):
                b=[r for r in pick('E10','analytical',policy_id=policy)
                   if runs[r['run_id']]['trial']['simulator']['max_resident_requests']==residents
                   and runs[r['run_id']]['trial']['simulator']['max_prefill_requests_per_iteration']==prefills]
                capacity.append([residents,prefills,policy,*[number(average(b,m)) for m in ('ttft_p95_ms','tpot_p95_ms','request_throughput_rps')]])
    capacity_table=table(['Residents','Prefills/iteration','Policy','TTFT P95, ms','TPOT P95, ms','Cohort requests/s'],capacity)
    e2=compare('E2','piecewise','chunk_128','tpot_p95_ms')
    count_table=table(['Experiment','Executed trials'],sorted(Counter(r['experiment_id'] for r in rows).items(),key=lambda x:int(x[0][1:])))
    source=', '.join(manifest['simulation_commits'])
    from string import Template
    context = dict(locals(), source_sha256=manifest['runner']['source_sha256'],
                   trial_count=manifest['trial_count'], e2_tpot_change=f"{e2['relative_change_pct']:.2f}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, template in [('REPORT.md','report_template.md.tmpl'),('README.md','readme_template.md.tmpl')]:
        text = Template((ROOT/'scripts'/template).read_text(encoding='utf-8')).substitute(context)
        (destination/name).write_text(text, encoding='utf-8')
    print(f'Built report and README from {len(rows)} verified trials.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();build(args.results,args.output)
