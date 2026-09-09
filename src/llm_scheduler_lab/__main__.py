from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description='LLM serving scheduler simulation lab (no model inference)')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run',help='Run or resume a JSON/YAML experiment suite')
    run.add_argument('--config',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--experiment')
    run.add_argument('--allow-dirty',action='store_true')
    trace=sub.add_parser('trace',help='Generate a replayable synthetic CSV workload')
    trace.add_argument('--output',type=Path,required=True)
    trace.add_argument('--seed',type=int,default=101)
    trace.add_argument('--requests',type=int,default=128)
    trace.add_argument('--mode',default='poisson')
    trace.add_argument('--regime',default='mixed')
    for name in ('summarize','plot'):
        p=sub.add_parser(name)
        p.add_argument('--results',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='run':
            from .experiments import run_suite
            m=run_suite(args.config,args.output,experiment=args.experiment,allow_dirty=args.allow_dirty)
            print(json.dumps({k:m[k] for k in ('trial_count','cached_trials','repaired_cache_entries','complete')}))
        elif args.command=='trace':
            from .workload import WorkloadConfig,generate_workload,write_trace
            write_trace(args.output,generate_workload(WorkloadConfig(n_requests=args.requests,seed=args.seed,
                                                                   arrival_mode=args.mode,regime=args.regime)))
            print(args.output)
        elif args.command=='summarize':
            from .analysis import analyze
            analyze(args.results,args.output)
        elif args.command=='plot':
            from .plotting import plot_all
            plot_all(args.results,args.output)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':main()
