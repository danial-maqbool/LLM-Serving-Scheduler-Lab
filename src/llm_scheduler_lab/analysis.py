"""Aggregate matched workload seeds; do not pool requests as replicates."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from .experiments import atomic_json, write_csv
from .statistics import describe, paired_comparison

METRICS = ['ttft_p50_ms','ttft_p95_ms','ttft_p99_ms','tpot_p95_ms','tpot_p99_ms',
           'e2e_p95_ms','itl_p95_ms','itl_p99_ms','request_throughput_rps','output_token_throughput_tps',
           'observed_request_throughput_rps','iteration_cv','max_wait_ms','starvation_proxy_count',
           'waiting_satisfaction_jain','short_prompt_ttft_p95_ms','long_prompt_ttft_p95_ms',
           'primer_max_itl_ms','backlog_at_arrival_end','server_busy_fraction','batch_slot_occupancy']
LABELS = ['experiment_id','group_id','policy_id','cost_id','regime','arrival_mode','offered_load_rps','concurrency']


def read_results(folder: Path) -> tuple[list[dict],dict]:
    manifest=json.loads((folder/'manifest.json').read_text())
    for name, expected in manifest['outputs'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'artifact checksum mismatch: {name}')
    with (folder/'summary.csv').open(newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=manifest['trial_count'] or len({r['run_id'] for r in rows})!=len(rows):
        raise ValueError('missing or duplicate result rows')
    if any(r['result_kind']!='SIMULATED' for r in rows):
        raise ValueError('this report accepts simulated results only')
    return rows,manifest


def analyze(folder: Path, output: Path) -> None:
    rows,manifest=read_results(folder)
    grouped=defaultdict(list)
    for row in rows: grouped[(row['group_id'],row['policy_id'])].append(row)
    aggregates,comparisons=[],[]
    expected=set(manifest['suite']['seeds'])
    for (group,policy), block in sorted(grouped.items()):
        by_seed={int(r['seed']):r for r in block}
        if len(by_seed)!=len(block) or set(by_seed)!=expected:
            raise ValueError('missing or duplicate seeds within a condition')
        meta={k:block[0][k] for k in LABELS}
        for metric in METRICS:
            values=[float(r[metric]) for r in block if r.get(metric) not in ('',None)]
            if values: aggregates.append({**meta,'metric':metric,**describe(values)})
        baseline=grouped.get((group,'mono'))
        if not baseline or policy=='mono': continue
        a={int(r['seed']):r for r in baseline}
        if set(a)!=set(by_seed): raise ValueError('unpaired seeds')
        for seed in a:
            if a[seed]['workload_sha256']!=by_seed[seed]['workload_sha256']:
                raise ValueError('unpaired workload content')
            if block[0]['arrival_mode']!='fixed_concurrency' and a[seed]['actual_arrivals_sha256']!=by_seed[seed]['actual_arrivals_sha256']:
                raise ValueError('open-loop arrival mismatch')
        for metric in METRICS:
            valid=[s for s in a if a[s].get(metric) not in ('',None) and by_seed[s].get(metric) not in ('',None)]
            if valid:
                comparisons.append({**meta,'baseline_policy':'mono','metric':metric,
                    **paired_comparison({s:float(a[s][metric]) for s in valid},
                                        {s:float(by_seed[s][metric]) for s in valid})})
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'aggregate.csv',aggregates)
    write_csv(output/'paired.csv',comparisons)
    atomic_json(output/'statistics.json',{'result_kind':'SIMULATED','bootstrap_resamples':2000,
        'resampling_unit':'matched workload seed','confidence_interval':'percentile bootstrap 95%',
        'seed_count':len(expected),'aggregate_rows':len(aggregates),'paired_rows':len(comparisons),
        'limitations':['Five seeds give limited uncertainty resolution.',
                       'Intervals reflect workload randomness, not cost-model error.',
                       'No significance or universal scheduler ranking is claimed.'],
        'input_summary_sha256':manifest['outputs']['summary.csv']})
    print(f'Wrote {len(aggregates)} aggregate rows and {len(comparisons)} paired comparisons.')
