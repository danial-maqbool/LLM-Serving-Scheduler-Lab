"""Generate the committed study design. This script does not inspect results."""
from pathlib import Path
import json


def chunk(n: int, **extra) -> dict:
    suffix = '_aging' if extra.get('prefill_order') == 'aging' else '_prefill_first' if extra.get('priority') == 'prefill_first' else ''
    return {'id':f'chunk_{n}{suffix}','name':'chunked_prefill','chunk_size':n,**extra}


def build_suite() -> dict:
    fcfs={'id':'fcfs','name':'fcfs'}
    mono={'id':'mono','name':'continuous_batching'}
    default=[fcfs,mono,chunk(512)]
    models=[{'id':'analytical','name':'analytical'},{'id':'piecewise','name':'piecewise'}]
    sensitivity=[{'id':f'a_p{p}_d{d}','name':'analytical','prefill_ms_per_token':.018*p,
                  'decode_ms_per_sequence':.22*d,'decode_quadratic_ms':.003*d}
                 for p in [.5,1,2] for d in [.5,1,2]]
    sensitivity += [{'id':f'piecewise_mix_{f}','name':'piecewise','interference':f} for f in [0,.25,1]]
    sensitivity += [{'id':'trace_demo','name':'trace'}]
    return {'schema_version':1,'seeds':[101,202,303,404,505],
            'defaults':{'workload':{'n_requests':128,'arrival_rate_rps':16,'regime':'mixed',
                                    'max_prompt_tokens':8192,'max_output_tokens':512},
                        'simulator':{'max_batch_size':32,'max_resident_requests':32,
                                     'max_tokens_per_iteration':16384,'max_prefill_requests_per_iteration':1}},
            'policies':default,'cost_models':models,
            'experiments':[
                {'id':'E1','workload':{'n_requests':256},'policies':[fcfs,mono,chunk(128),chunk(512)]},
                {'id':'E2','workload':{'arrival_rate_rps':64},
                 'policies':[fcfs,mono,*[chunk(n) for n in [32,64,128,256,512,1024,2048]]]},
                {'id':'E3','workload':{'n_requests':256},'sweep':{'workload.arrival_rate_rps':[2,8,16,32,64,128]}},
                {'id':'E4','sweep':{'workload.regime':['short_short','long_short','short_long','mixed']}},
                {'id':'E5','workload':{'n_requests':64,'arrival_mode':'head_of_line'},'trace_seeds':[101],
                 'policies':[fcfs,mono,chunk(64),chunk(64,prefill_order='aging')]},
                {'id':'E6','workload':{'n_requests':160,'arrival_rate_rps':16},
                 'sweep':{'workload.arrival_mode':['poisson','burst']}},
                {'id':'E7','workload':{'n_requests':192,'arrival_rate_rps':64},
                 'policies':[fcfs,mono,chunk(512),chunk(512,prefill_order='aging'),chunk(512,priority='prefill_first')]},
                {'id':'E8','workload':{'arrival_rate_rps':32},'cost_models':sensitivity},
                {'id':'E9','workload':{'arrival_mode':'fixed_concurrency'},
                 'sweep':{'workload.concurrency':[1,4,16,32]}},
                {'id':'E10','workload':{'arrival_rate_rps':64},'cost_models':[models[0]],
                 'sweep':{'simulator.max_resident_requests':[8,32],
                          'simulator.max_prefill_requests_per_iteration':[1,4]}}
            ]}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    data=build_suite()
    (root/'configs/study.json').write_text(json.dumps(data,indent=2)+'\n')
    smoke=build_suite();smoke['seeds']=[101,202]
    smoke['defaults']['workload'].update(n_requests=12,max_prompt_tokens=128,max_output_tokens=8)
    smoke['experiments']=[{'id':'smoke'}]
    (root/'configs/smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
