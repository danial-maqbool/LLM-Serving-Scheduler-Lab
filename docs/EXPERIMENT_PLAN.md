# Experiment Plan

## E1 — Baseline policy comparison

Compare FCFS, continuous batching, and chunked prefill at chunk sizes 128/256/512/1024 on the same workload seeds.

Report TTFT P50/P95/P99, TPOT P50/P95, E2E P95, throughput, and iteration CV.

## E2 — Chunk-size sweep

Sweep chunk size:

```text
32, 64, 128, 256, 512, 1024, 2048
```

Goal: expose the TTFT/TPOT/throughput trade-off rather than claiming one universal optimum.

## E3 — Arrival-rate saturation curve

Sweep request arrival rate from underloaded to overloaded. Plot throughput and tail latency versus offered load.

## E4 — Prompt-length regimes

At minimum:

- short prompts / short outputs;
- long prompts / short outputs;
- short prompts / long outputs;
- mixed heavy-tail workload.

## E5 — Adversarial head-of-line workload

Inject a very long prompt before many short requests. Quantify head-of-line blocking and decode disruption.

## E6 — Burst arrivals

Compare Poisson arrivals against synchronized bursts.

## E7 — Fairness / starvation

Track per-request waiting time. Confirm no policy except intentionally serial FCFS creates accidental starvation.

## E8 — Cost-model sensitivity

Vary prefill/decode coefficients over plausible ranges. Conclusions that disappear under small coefficient changes must be labeled model-sensitive.

## E9 — Hardware calibration (optional but preferred)

If a CUDA GPU is available, collect simple prefill/decode timing traces for one open model and fit/compare the analytical model. Keep measured and simulated results separate.

## Required figures

1. TTFT P95 vs policy/chunk size
2. TPOT P95 vs policy/chunk size
3. Request throughput vs offered load
4. Iteration duration distribution
5. Iteration CV vs chunk size
6. Latency-throughput Pareto frontier
7. Head-of-line blocking case study timeline
8. Sensitivity heatmap or small-multiple plot
