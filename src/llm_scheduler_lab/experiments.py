"""Config expansion, atomic verified caches, and reproducible experiment artifacts."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import platform
import subprocess
import time
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cost import make_cost_model
from .metrics import percentile, request_metrics
from .models import SimulatorConfig
from .policies import ChunkedPrefillPolicy, ContinuousBatchingPolicy, FCFSPolicy
from .simulator import Simulator
from .workload import WorkloadConfig, generate_workload, read_trace, workload_hash

SCHEMA_VERSION = 2


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(canonical(value) + b'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_config(path: Path) -> dict:
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('YAML input requires the analysis extra (PyYAML)') from exc
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('expected a schema_version=1 suite mapping')
    return value


def source_fingerprint() -> str:
    # Plotting and documentation changes do not invalidate simulation caches.
    root = Path(__file__).parent
    names = ['__init__.py', 'models.py', 'cost.py', 'metrics.py', 'simulator.py', 'workload.py', 'experiments.py']
    names += [p.relative_to(root).as_posix() for p in sorted((root / 'policies').glob('*.py'))]
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode() + b'\0' + (root / name).read_bytes() + b'\0')
    return h.hexdigest()


def git_state(root: Path) -> dict:
    def command(*args: str) -> str:
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
    try:
        return {'git_commit': command('rev-parse', 'HEAD'),
                'git_dirty': bool(command('status', '--porcelain', '--untracked-files=normal'))}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': 'unavailable', 'git_dirty': True}


def make_policy(spec: dict):
    values = dict(spec)
    values.pop('id', None)
    name = values.pop('name')
    if name == 'fcfs':
        if values: raise ValueError('FCFS has no extra policy parameters')
        return FCFSPolicy()
    if values.get('priority', 'decode_first') not in ('decode_first', 'prefill_first'):
        raise ValueError('invalid priority')
    if values.get('prefill_order', 'fcfs') not in ('fcfs', 'aging'):
        raise ValueError('invalid prefill order')
    if name == 'continuous_batching': return ContinuousBatchingPolicy(**values)
    if name == 'chunked_prefill': return ChunkedPrefillPolicy(**values)
    raise ValueError(f'unknown policy: {name}')


def expand_suite(data: dict, experiment: str | None = None) -> list[dict]:
    if data.get('schema_version') != 1:
        raise ValueError('unsupported suite schema')
    seeds = data.get('seeds', [])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('provide distinct nonnegative integer seeds')
    defaults = data.get('defaults', {})
    experiment_ids = [e['id'] for e in data['experiments']]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError('experiment IDs must be unique')
    if experiment and experiment not in experiment_ids:
        raise ValueError(f'unknown experiment: {experiment}')
    trials = []
    for exp in data['experiments']:
        if experiment and exp['id'] != experiment: continue
        sweep = exp.get('sweep', {})
        axes = sorted(sweep)
        if any(not isinstance(sweep[k], list) or not sweep[k] for k in axes):
            raise ValueError('sweep axes must contain nonempty lists')
        policies = exp.get('policies', data.get('policies', []))
        if not policies or len({p['id'] for p in policies}) != len(policies):
            raise ValueError('policy IDs must be nonempty and unique')
        models = exp.get('cost_models', data.get('cost_models', [{'id':'analytical','name':'analytical'}]))
        if not models or len({m['id'] for m in models}) != len(models):
            raise ValueError('cost model IDs must be nonempty and unique')
        for values in itertools.product(*(sweep[k] for k in axes)):
            work = {**defaults.get('workload', {}), **exp.get('workload', {})}
            sim = {**defaults.get('simulator', {}), **exp.get('simulator', {})}
            coordinate = dict(zip(axes, values))
            for key, value in coordinate.items():
                namespace, field = key.split('.', 1)
                if namespace not in ('workload', 'simulator'):
                    raise ValueError('sweep keys must start with workload. or simulator.')
                (work if namespace == 'workload' else sim)[field] = value
            cfg = SimulatorConfig(**sim)
            for cost in models:
                cost_spec = {k:v for k,v in cost.items() if k != 'id'}
                group = exp['id'] + '-' + digest({'workload':work, 'simulator':asdict(cfg), 'cost':cost})[:12]
                for seed, policy in itertools.product(seeds, policies):
                    wc = WorkloadConfig(**{**work, 'seed':seed})
                    make_policy(policy)
                    trials.append({'experiment_id':exp['id'], 'group_id':group, 'coordinate':coordinate,
                                   'workload':asdict(wc), 'simulator':asdict(cfg), 'cost':cost_spec,
                                   'cost_id':cost['id'], 'policy':policy,
                                   'trace':seed in exp.get('trace_seeds', []),
                                   'input_trace':exp.get('input_trace')})
    if len({digest(t) for t in trials}) != len(trials):
        raise ValueError('duplicate trial generated by sweep')
    return trials


def run_trial(trial: dict, provenance: dict, input_root: Path) -> dict:
    cfg, wc = SimulatorConfig(**trial['simulator']), WorkloadConfig(**trial['workload'])
    requests = (read_trace(input_root / trial['input_trace']) if trial['input_trace']
                else generate_workload(wc))
    cost_spec = dict(trial['cost'])
    if 'path' in cost_spec:
        cost_spec['path'] = str((input_root / cost_spec['path']).resolve())
    model = make_cost_model(cost_spec, cfg)
    policy = make_policy(trial['policy'])
    closed = wc.arrival_mode == 'fixed_concurrency'
    result = Simulator(cfg, model).run(requests, policy, concurrency=wc.concurrency if closed else None,
                                       think_time_ms=wc.think_time_ms if closed else 0)
    metrics = result.metrics
    per_request = [request_metrics(s) for s in result.states]
    for label, subset in [('short_prompt', [r for r in per_request if r['prompt_tokens']<=256]),
                          ('long_prompt', [r for r in per_request if r['prompt_tokens']>=2048])]:
        metrics[f'{label}_count'] = len(subset)
        metrics[f'{label}_ttft_p95_ms'] = percentile([r['ttft_ms'] for r in subset], .95)
    by_arrival = sorted(per_request, key=lambda r:(r['arrival_ms'],r['request_id']))
    mid = len(by_arrival)//2
    for label, subset in [('early',by_arrival[:mid]),('late',by_arrival[mid:])]:
        metrics[f'{label}_ttft_p95_ms'] = percentile([r['ttft_ms'] for r in subset], .95)
    metrics['primer_max_itl_ms'] = per_request[0]['itl_max_ms'] if per_request else None
    row = {'experiment_id':trial['experiment_id'], 'group_id':trial['group_id'],
           'policy_id':trial['policy']['id'], 'policy_name':policy.name,
           'chunk_size':getattr(policy,'chunk_size',0),
           'prefill_order':getattr(policy,'prefill_order','fcfs'),
           'priority':getattr(policy,'priority','serial'), 'cost_id':trial['cost_id'],
           'cost_source_kind':model.metadata()['source_kind'], 'result_kind':'SIMULATED',
           'seed':wc.seed, 'regime':wc.regime, 'arrival_mode':wc.arrival_mode,
           'offered_load_rps':wc.arrival_rate_rps if not closed else None,
           'concurrency':wc.concurrency if closed else None,
           'workload_sha256':workload_hash(requests),
           'actual_arrivals_sha256':workload_hash([s.request for s in result.states]),
           **provenance, **metrics}
    record = {'schema_version':SCHEMA_VERSION, 'summary':row, 'trial':trial,
              'cost_model':model.metadata(), 'request_metrics':per_request,
              'iteration_histogram_ms':sorted(Counter(round(it.duration_ms,3) for it in result.iterations).items())}
    if trial['trace']:
        record['iterations'] = [asdict(it) for it in result.iterations]
        record['requests'] = [asdict(s.request) for s in result.states]
    return record


def external_fingerprints(trial: dict, root: Path) -> dict:
    paths = [trial.get('input_trace'), trial['cost'].get('path')]
    result = {}
    for name in paths:
        if name:
            path = root / name
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            sidecar = path.with_suffix('.metadata.json')
            if sidecar.exists(): result[name + ':metadata'] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    return result


def run_suite(config_path: Path, output: Path, *, experiment: str | None = None,
              allow_dirty: bool = False, quiet: bool = False) -> dict:
    data = read_config(config_path)
    trials = expand_suite(data, experiment)
    root = Path(__file__).resolve().parents[2]
    provenance = {**git_state(root), 'source_sha256':source_fingerprint(),
                  'python_version':platform.python_version(), 'platform':platform.platform()}
    if provenance['git_dirty'] and not allow_dirty:
        raise ValueError('working tree is dirty or Git metadata is unavailable; commit first or use --allow-dirty')
    output.mkdir(parents=True, exist_ok=True)
    cache = output / '.runs'
    records, reused, repaired = [], 0, 0
    started = time.perf_counter()
    for index, trial in enumerate(trials):
        key = digest({'schema':SCHEMA_VERSION, 'source':provenance['source_sha256'], 'trial':trial,
                      'external':external_fingerprints(trial, config_path.parent)})
        path = cache / f'{key}.json'
        record = None
        if path.exists():
            try:
                envelope = json.loads(path.read_text())
                candidate = envelope['payload']
                if envelope['key'] != key or envelope['sha256'] != digest(candidate):
                    raise ValueError('cache digest mismatch')
                if candidate['schema_version'] != SCHEMA_VERSION or candidate['summary']['source_sha256'] != provenance['source_sha256']:
                    raise ValueError('cache schema or source mismatch')
                if not allow_dirty and candidate['summary']['git_dirty']:
                    raise ValueError('dirty-worktree cache cannot certify a clean run')
                record = candidate
                reused += 1
            except (ValueError, KeyError, TypeError):
                repaired += 1
        if record is None:
            record = run_trial(trial, {**provenance,'created_utc':datetime.now(timezone.utc).isoformat()}, config_path.parent)
            record['summary']['run_id'] = key
            atomic_json(path, {'key':key, 'sha256':digest(record), 'payload':record})
        records.append(record)
        if not quiet and (index % 25 == 0 or index+1 == len(trials)):
            print(f'{index+1}/{len(trials)} trials; {reused} verified cache hits; {time.perf_counter()-started:.1f}s', flush=True)
    rows = [r['summary'] for r in records]
    write_csv(output / 'summary.csv', rows)
    atomic_json(output / 'runs.json', [{'run_id':r['summary']['run_id'], 'trial':r['trial'],
                                       'cost_model':r['cost_model'], 'histogram':r['iteration_histogram_ms']}
                                      for r in records])
    for r in records:
        if 'iterations' in r:
            atomic_json(output / 'traces' / f"{r['summary']['run_id']}.json", r)
    manifest = {'schema_version':SCHEMA_VERSION, 'result_kind':'SIMULATED', 'complete':True,
                'suite_sha256':digest(data), 'suite':data, 'selected_experiment':experiment,
                'trial_count':len(rows), 'cached_trials':reused, 'repaired_cache_entries':repaired,
                'runner':provenance, 'simulation_commits':sorted({r['git_commit'] for r in rows}),
                'seed_count':len(data['seeds']), 'elapsed_wall_seconds':time.perf_counter()-started,
                'outputs':{name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                           for name in ('summary.csv','runs.json')}}
    atomic_json(output / 'manifest.json', manifest)
    return manifest
