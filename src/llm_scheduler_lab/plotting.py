"""Static figures from verified artifacts. Matplotlib defaults; one axes per file."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from .analysis import read_results
from .statistics import describe


def plot_all(folder: Path, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows,manifest=read_results(folder)
    output.mkdir(parents=True,exist_ok=True)
    files=[]
    plt.rcParams['svg.fonttype']='none'
    plt.rcParams['svg.hashsalt']='llm-scheduler-lab'

    def select(exp,model='piecewise'):
        return [r for r in rows if r['experiment_id']==exp and r['cost_id']==model]

    def new(title,xlabel,ylabel):
        fig,ax=plt.subplots(figsize=(8.4,5.2))
        ax.set_title(title,fontsize=12,pad=12)
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
        ax.grid(axis='y',alpha=.25)
        return fig,ax

    def save(fig,ax,name,subset):
        sizes=sorted({r['requests'] for r in subset},key=int)
        seeds=sorted({r['seed'] for r in subset},key=int)
        commits=sorted({r['git_commit'][:8] for r in subset})
        fig.text(.01,.014,f"SIMULATED · n={','.join(sizes)} per run · seeds={len(seeds)} · source={','.join(commits)}",
                 fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1))
        fig.savefig(output/f'{name}.svg',metadata={'Date':None})
        fig.savefig(output/f'{name}.png',dpi=160)
        plt.close(fig);files.append(name+'.svg')

    def curve(ax,subset,xfield,metric,label):
        xs=sorted({float(r[xfield]) for r in subset if r[xfield]})
        means,lows,highs=[],[],[]
        for x in xs:
            vals=[float(r[metric]) for r in subset if float(r[xfield])==x and r.get(metric)]
            d=describe(vals,resamples=2000)
            means.append(d['mean']);lows.append(d['mean']-d['ci95_low']);highs.append(d['ci95_high']-d['mean'])
        ax.errorbar(xs,means,yerr=[lows,highs],fmt='o-',capsize=3,label=label)

    if not select('E2'):
        fig,ax=new('Smoke experiment (not the full study)','Policy','Mean TTFT P95 (ms)')
        policies=sorted({r['policy_id'] for r in rows})
        ax.bar(policies,[fmean(float(r['ttft_p95_ms']) for r in rows if r['policy_id']==p) for p in policies])
        save(fig,ax,'smoke_overview',rows)
        return files

    sweep=select('E2')
    chunks=[r for r in sweep if int(r['chunk_size'])>0]
    for name,metric,ylabel in [('01_ttft_chunk','ttft_p95_ms','Mean request TTFT P95 (ms)'),
                               ('02_tpot_chunk','tpot_p95_ms','Mean request TPOT P95 (ms)')]:
        fig,ax=new('Chunk-size trade-off · piecewise cost model','Chunk limit (tokens)',ylabel)
        curve(ax,chunks,'chunk_size',metric,'Chunked prefill · 95% seed CI')
        for policy in ('mono','fcfs'):
            y=fmean(float(r[metric]) for r in sweep if r['policy_id']==policy)
            ax.axhline(y,linestyle='--' if policy=='mono' else ':',label=policy)
        ax.set_xscale('log',base=2);ax.legend(fontsize=8)
        save(fig,ax,name,sweep)

    load=select('E3')
    for name,metric,ylabel in [('03_throughput_load','observed_request_throughput_rps','Arrival-window completions (requests/s)'),
                               ('12_ttft_load','ttft_p95_ms','Mean cohort TTFT P95, including drain (ms)')]:
        fig,ax=new('Finite offered-load sweep · piecewise cost model','Offered load (requests/s)',ylabel)
        for policy in ('fcfs','mono','chunk_512'):
            curve(ax,[r for r in load if r['policy_id']==policy],'offered_load_rps',metric,policy)
        if metric=='ttft_p95_ms': ax.set_yscale('log');ax.set_ylabel(ylabel+' · log scale')
        ax.set_xscale('log',base=2);ax.legend(fontsize=8)
        save(fig,ax,name,load)

    baseline=[r for r in select('E1') if int(r['seed'])==101]
    runs={r['run_id']:r for r in json.loads((folder/'runs.json').read_text())}
    fig,ax=new('Iteration duration ECDF · piecewise · seed 101','Iteration duration (ms) · log scale','Cumulative fraction of iterations')
    for row in baseline:
        hist=runs[row['run_id']]['histogram']
        durations=[h[0] for h in hist];counts=np.array([h[1] for h in hist])
        ax.step(durations,np.cumsum(counts)/counts.sum(),where='post',label=row['policy_id'])
    ax.set_xscale('log');ax.set_ylim(0,1.02);ax.legend(fontsize=8)
    save(fig,ax,'04_iteration_ecdf',baseline)

    fig,ax=new('Iteration variation depends on the cost model','Chunk limit (tokens)','Iteration coefficient of variation (unitless)')
    all_cv=[]
    for model in ('analytical','piecewise'):
        data=[r for r in select('E2',model) if int(r['chunk_size'])>0]
        all_cv.extend(data);curve(ax,data,'chunk_size','iteration_cv',model)
    ax.set_xscale('log',base=2);ax.legend(fontsize=8)
    save(fig,ax,'05_iteration_cv',all_cv)

    fig,ax=new('Latency-throughput trade-off · piecewise · offered 64 requests/s',
               'Mean TTFT P95 (ms)','Finite-cohort throughput including drain (requests/s)')
    points=[]
    for policy in sorted({r['policy_id'] for r in sweep}):
        data=[r for r in sweep if r['policy_id']==policy]
        points.append((fmean(float(r['ttft_p95_ms']) for r in data),
                       fmean(float(r['request_throughput_rps']) for r in data),policy))
    ax.scatter([p[0] for p in points],[p[1] for p in points])
    for i,(x,y,label) in enumerate(points):
        ax.annotate(label.replace('chunk_','c'),(x,y),xytext=(5,6+(i%2)*8),textcoords='offset points',fontsize=8)
    frontier=sorted(p for p in points if not any(q[0]<=p[0] and q[1]>=p[1] and (q[0]<p[0] or q[1]>p[1]) for q in points))
    if len(frontier)>1:ax.plot([p[0] for p in frontier],[p[1] for p in frontier],linestyle='--',label='Nondominated means')
    ax.set_xscale('log');ax.set_xlabel('Mean TTFT P95 (ms) · log scale')
    if len(frontier)>1:ax.legend(fontsize=8)
    save(fig,ax,'06_pareto',sweep)

    hol=[r for r in select('E5','analytical') if int(r['seed'])==101 and r['policy_id']!='fcfs']
    fig,ax=new('Head-of-line case · same arrivals · analytical cost model','Simulated time since first arrival (ms)','Policy and request')
    labels=[];y=0
    for row in sorted(hol,key=lambda r:r['policy_id']):
        trace=json.loads((folder/'traces'/f"{row['run_id']}.json").read_text())
        for rid,title in [(0,'stream'),(1,'large prompt'),(2,'short request')]:
            intervals=[(it['start_ms'],it['duration_ms']) for it in trace['iterations']
                       if str(rid) in it['prefill_allocations']]
            token_times=[it['end_ms'] for it in trace['iterations'] if rid in it['decode_request_ids']]
            ax.broken_barh(intervals,(y-.22,.44),hatch='//',label='Prefill service' if y==0 else None)
            ax.scatter(token_times,[y]*len(token_times),marker='|',s=35,label='Decode output' if y==0 else None)
            labels.append(row['policy_id'].replace('chunk_','c')+' · '+title);y+=1
    ax.set_yticks(range(y),labels,fontsize=8);ax.invert_yaxis();ax.legend(fontsize=8)
    save(fig,ax,'07_hol_timeline',hol)

    fig,ax=new('Cost sensitivity · chunk 512 vs monolithic prefill','Prefill cost multiplier','Change in mean TPOT P95 (%) · negative is lower')
    sensitivity=[]
    for d in (.5,1,2):
        vals=[]
        for p in (.5,1,2):
            data=select('E8',f'a_p{p}_d{d}');sensitivity.extend(data)
            mono=fmean(float(r['tpot_p95_ms']) for r in data if r['policy_id']=='mono')
            chunked=fmean(float(r['tpot_p95_ms']) for r in data if r['policy_id']=='chunk_512')
            vals.append(100*(chunked/mono-1))
        ax.plot([.5,1,2],vals,'o-',label=f'Decode multiplier {d}')
    ax.axhline(0,linestyle=':');ax.legend(fontsize=8)
    save(fig,ax,'08_cost_sensitivity',sensitivity)

    fairness=select('E7')
    fig,ax=new('Waiting satisfaction · piecewise · offered 64 requests/s','Policy','Jain index (unitless; higher is more equal)')
    names=sorted({r['policy_id'] for r in fairness})
    stats=[describe([float(r['waiting_satisfaction_jain']) for r in fairness if r['policy_id']==p]) for p in names]
    ax.errorbar(range(len(names)),[d['mean'] for d in stats],
                yerr=[[d['mean']-d['ci95_low'] for d in stats],[d['ci95_high']-d['mean'] for d in stats]],fmt='o',capsize=4)
    ax.set_xticks(range(len(names)),[p.replace('chunk_','c').replace('_','\n') for p in names],fontsize=8)
    ax.set_ylim(0,1.02);save(fig,ax,'09_fairness',fairness)

    burst=select('E6')
    fig,ax=new('Arrival burst sensitivity · piecewise · configured 16 requests/s','Policy / arrival process','Mean TTFT P95 (ms)')
    names=[];values=[]
    for policy in ('fcfs','mono','chunk_512'):
        for mode in ('poisson','burst'):
            data=[r for r in burst if r['policy_id']==policy and r['arrival_mode']==mode]
            names.append(policy.replace('chunk_','c')+'\n'+mode)
            values.append(fmean(float(r['ttft_p95_ms']) for r in data))
    ax.bar(range(len(names)),values);ax.set_xticks(range(len(names)),names,fontsize=8)
    save(fig,ax,'10_burst',burst)

    closed=select('E9')
    fig,ax=new('Closed-loop submission · piecewise · policy-dependent arrival times','Concurrent clients','Finite-cohort throughput (requests/s)')
    for policy in ('fcfs','mono','chunk_512'):
        curve(ax,[r for r in closed if r['policy_id']==policy],'concurrency','request_throughput_rps',policy)
    ax.legend(fontsize=8);save(fig,ax,'11_closed_loop',closed)
    (output/'manifest.json').write_text(json.dumps({'result_kind':'SIMULATED','files':files,
        'input_summary_sha256':manifest['outputs']['summary.csv'],'source_commits':manifest['simulation_commits']},indent=2)+'\n')
    return files
