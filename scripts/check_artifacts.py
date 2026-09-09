"""Independently validate result coverage, accounting, and artifact integrity."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from llm_scheduler_lab.analysis import read_results
from llm_scheduler_lab.experiments import expand_suite


def check(folder: Path) -> dict:
    rows, manifest = read_results(folder)
    if not manifest.get('complete') or manifest.get('result_kind') != 'SIMULATED':
        raise ValueError('a complete simulated manifest is required')
    planned = expand_suite(manifest['suite'], manifest.get('selected_experiment'))
    if len(rows) != len(planned):
        raise ValueError('experiment coverage differs from the declared suite')
    expected = {(r['experiment_id'], r['group_id'], r['policy']['id'], r['workload']['seed']) for r in planned}
    actual = {(r['experiment_id'], r['group_id'], r['policy_id'], int(r['seed'])) for r in rows}
    if actual != expected:
        raise ValueError('missing or unexpected experiment coordinates')
    sources = {r['source_sha256'] for r in rows}
    if sources != {manifest['runner']['source_sha256']}:
        raise ValueError('mixed source implementations')
    runs = json.loads((folder/'runs.json').read_text(encoding='utf-8'))
    if {r['run_id'] for r in runs} != {r['run_id'] for r in rows} or len(runs) != len(rows):
        raise ValueError('run metadata coverage mismatch')
    for r in rows:
        for field in ('requests', 'iterations', 'wall_time_ms', 'ttft_p95_ms'):
            value = float(r[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid {field}')
        if r['git_dirty'] != 'False':
            raise ValueError('release results must come from a clean source commit')
        if int(r['unfinished_requests']) != 0:
            raise ValueError('unfinished requests must not be silently dropped')
        if not 0 < float(r['server_busy_fraction']) <= 1 or not 0 < float(r['batch_slot_occupancy']) <= 1:
            raise ValueError('occupancy proxy outside physical bounds')
        if not 0 < float(r['waiting_satisfaction_jain']) <= 1.0000000001:
            raise ValueError('Jain index outside bounds')
    count = 0
    for path in sorted((folder/'traces').glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        cfg = record['trial']['simulator']
        prompts, emitted, completed, admitted = {}, {}, set(), set()
        for it in record['iterations']:
            prefill, decode = it['prefill_allocations'], it['decode_request_ids']
            if sum(prefill.values()) + len(decode) > cfg['max_tokens_per_iteration']:
                raise ValueError('trace token budget violation')
            if len(prefill)+len(decode) > cfg['max_batch_size'] or set(map(int, prefill)) & set(decode):
                raise ValueError('trace sequence budget violation')
            if it['start_ms'] >= it['end_ms']:
                raise ValueError('trace does not advance time')
            for rid in it['admitted_ids']:
                if rid in admitted: raise ValueError('duplicate admission')
                admitted.add(rid)
            for rid, tokens in prefill.items():
                prompts[int(rid)] = prompts.get(int(rid), 0) + tokens
            for rid in decode: emitted[rid] = emitted.get(rid, 0) + 1
            for rid in it['completed_ids']:
                if rid in completed: raise ValueError('duplicate completion')
                completed.add(rid)
        for r in record['requests']:
            rid = r['request_id']
            if prompts.get(rid) != r['prompt_tokens'] or rid not in completed or rid not in admitted:
                raise ValueError('trace lost a request or prompt tokens')
            first = int(cfg['first_token_source'] == 'prefill')
            if emitted.get(rid, 0) + first != r['output_tokens']:
                raise ValueError('trace output conservation failure')
        count += 1
    result = {'trials_checked':len(rows), 'traces_checked':count,
              'source_sha256':next(iter(sources)), 'result_kind':'SIMULATED'}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    check(parser.parse_args().results)
