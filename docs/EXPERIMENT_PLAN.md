# Experiment design

The committed design predates the formal experiment run. The generator is
[make_suite.py](../scripts/make_suite.py). It does not inspect results.
The expanded [study configuration](../configs/study.json) contains **955 trials** across five seeds:
101, 202, 303, 404, and 505. No seed is selected based on its outcome.

| ID | Question | Trials |
| --- | --- | ---: |
| E1 | Baselines and chunks 128/512; 256 requests; analytical and piecewise | 40 |
| E2 | Chunks 32/64/128/256/512/1024/2048 at 64 requests/s | 90 |
| E3 | Offered load 2/8/16/32/64/128 requests/s; 256-request finite cohorts | 180 |
| E4 | Short/short, long/short, short/long, mixed lengths | 120 |
| E5 | A streaming request, a large prompt, then short arrivals; compare FCFS vs aging | 40 |
| E6 | Poisson versus burst arrivals at the same configured mean rate | 60 |
| E7 | Waiting fairness and prefill priority at high load | 50 |
| E8 | Nine analytical coefficient pairs, three piecewise interference settings, synthetic trace | 195 |
| E9 | Closed-loop concurrency 1/4/16/32 | 120 |
| E10 | Resident capacity 8/32 and prefill request capacity 1/4 | 60 |

E1-E8 satisfy the main study. E9 and E10 are supplemental controlled experiments, not hardware calibration.
Unless specified otherwise, each trial uses 128 requests, mixed lengths, a 16 requests/s Poisson
source, and the common limits in [methodology](METHODOLOGY.md). Output length is clamped at 512.
E5 uses 64 requests; E6 uses 160; E7 uses 192. E2/E7/E10 use 64 requests/s; E8 uses 32.

## Workload construction

Open-loop comparisons replay identical arrival and length traces across policies. Independent seeded
random streams keep lengths fixed when arrival processes or rates change. Burst mode submits groups
of 16 at deterministic intervals. Finite realized arrival rates need not match exactly; the configured
mean does. Lognormal lengths are clamped to explicit limits, so these are bounded heavy-tailed mixtures.

E5 starts a small-prompt, long-output stream. The next request has an 8,192-token prompt, followed
by many short requests. Seed 101 retains complete traces for all four policies and both models.
This tests decode interference and queue order separately.

Closed-loop E9 reuses the same ordered request templates, concurrency, and think time. It releases
replacement requests on completions. Arrival timestamps therefore differ between policies. Those
comparisons measure a common client-submission rule, not an identical arrival trace.

## Acceptance and reporting

All 955 trials must finish. Missing seeds, changed workload hashes, corrupted cache records, dropped
requests, and resource violations are failures. Report negative and neutral outcomes. A universal
best chunk size is not a required outcome. Finite overload curves must show backlog/drain behavior.
The [artifact checker](../scripts/check_artifacts.py) independently validates coverage and saved traces.
