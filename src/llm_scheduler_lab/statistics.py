"""Seed-level uncertainty. Requests within a run are not independent replicas."""
from __future__ import annotations

import math
import random
from statistics import fmean, median, stdev
from .metrics import percentile


def describe(values: list[float], *, resamples: int = 2000, seed: int = 2026) -> dict:
    if resamples <= 0 or not values or any(not math.isfinite(v) for v in values):
        raise ValueError('finite nonempty samples and positive resamples are required')
    rng = random.Random(seed)
    means = [fmean(rng.choices(values,k=len(values))) for _ in range(resamples)]
    return {'n':len(values),'mean':fmean(values),'median':median(values),
            'std':stdev(values) if len(values)>1 else 0,
            'ci95_low':percentile(means,.025),'ci95_high':percentile(means,.975),
            'resampling_unit':'workload_seed','resamples':resamples}


def paired_comparison(baseline: dict[int,float], candidate: dict[int,float], *,
                      resamples: int = 2000, seed: int = 2026) -> dict:
    if set(baseline)!=set(candidate) or not baseline:
        raise ValueError('paired comparisons require exactly the same nonempty seed set')
    keys=sorted(baseline)
    a,b=[baseline[k] for k in keys],[candidate[k] for k in keys]
    delta=[y-x for x,y in zip(a,b)]
    result=describe(delta,resamples=resamples,seed=seed)
    result.update({'baseline_mean':fmean(a),'candidate_mean':fmean(b),
                   'mean_delta':result['mean'],
                   'relative_change_pct':100*(fmean(b)/fmean(a)-1) if fmean(a)!=0 else None,
                   'paired_dz':fmean(delta)/stdev(delta) if len(delta)>1 and stdev(delta)>0 else None})
    # Ratio-of-means intervals retain the within-seed pairing.
    rng=random.Random(seed)
    relative=[]
    for _ in range(resamples):
        ids=rng.choices(range(len(keys)),k=len(keys))
        ma=fmean(a[i] for i in ids)
        if ma!=0:relative.append(100*(fmean(b[i] for i in ids)/ma-1))
    result.update({'relative_ci95_low':percentile(relative,.025),
                   'relative_ci95_high':percentile(relative,.975)})
    return result
