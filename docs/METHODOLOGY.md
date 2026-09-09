# Methodology and metric definitions

## Scope and prior work

This is an independent discrete-iteration simulator. It executes no transformer or GPU kernel.
All bundled timing coefficients and the example timing trace are synthetic. A simulated millisecond
is a model output, not elapsed host time. The lab does not reproduce published hardware speedups.

[Orca (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) motivates iteration-level
admission and batching. [Sarathi-Serve (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/agrawal)
motivates separating prefill work into chunks and measuring the latency-throughput trade-off.
These sources provide background. The simulator, numerical coefficients, workloads, and results are this lab's choices.

## Time and request lifecycle

Requests have an ID, arrival time, prompt length P, and required output length O. Counts are positive
integers; arrival times are finite and nonnegative. The engine knows O to terminate the request.
The schedulers do not use O for admission or prioritization.

At each iteration boundary, the engine moves arrived requests to the waiting queue. A scheduler
selects ready decodes and prefill work. The engine checks the plan before changing any state.
Each selected request occupies one sequence slot. Each selected decode processes one token.
A new arrival cannot join an iteration already in progress. An empty server jumps directly to the
next arrival. Idle time incurs no batch overhead but remains in throughput denominators.

Admission means first prefill service, not simply reaching the waiting queue. An admitted request
remains resident until it completes. Prefill can take several iterations. By default, the final
prefill iteration produces output token 1. Later decode iterations produce tokens 2 through O.
Thus O=1 requires no decode iteration. `first_token_source=decode` retains the starter's alternative
convention for explicit sensitivity tests. The report uses `prefill` throughout.

All outputs in one batch receive the batch-end timestamp. The engine models no within-batch
streaming, CPU overlap, networking, tokenization, or client-side transport delays.

## Shared hard resource limits

The default study uses 32 sequence slots per batch, 32 resident requests, 16,384 processed tokens
per iteration, and one prefill request per iteration. The largest generated prompt is 8,192 tokens.
These limits apply to every policy. A monolithic prompt above the hard token budget fails validation;
it does not receive a special exemption. A whole prompt that does not fit the residual budget waits.

Optional KV capacity uses a conservative reservation of `P + output_token_reservation` tokens per
resident. This is a capacity proxy, not a paged allocator. The output reservation is a configuration
limit independent of actual O. Inputs above the configured output limit fail validation when KV
capacity is enabled. The published primary sweeps leave this optional limit disabled; unit tests
exercise reservation and over-capacity behavior.

## Policies

| Policy | Selection rule | Purpose |
| --- | --- | --- |
| `fcfs` | Serve the oldest request alone to completion. Prefill is monolithic. | Serial baseline, not a production engine |
| `mono` | Schedule ready decodes first, then whole prompt work within residual limits. | Iteration-level batching baseline |
| `chunk_N` | Same selection order and resource limits; prefill work per selected request is at most N tokens. | Isolate chunk size |
| `chunk_N_aging` | Order prefills by oldest last-service time, or arrival if not yet served. | Test service rotation separately from chunking |
| `chunk_N_prefill_first` | Allocate prefills before decodes. | Test scheduling priority |

When decode capacity is limited, ready requests rotate by oldest last-service end time, then ID.
Default FCFS prefill ordering can retain head-of-line blocking across chunks. Chunking does not
promise that short waiting prompts bypass a large earlier prompt. The aging variant changes that
ordering explicitly. Candidate waiting lookahead is bounded by the sequence budget. Existing
partial prefills can progress even when a new request is blocked by resident or KV capacity.

## Policy-independent timing models

A cost model receives only batch work, never the scheduler name. Let P_b be total selected prefill
tokens, D_b selected decode requests, and R_b selected prefill requests.

For each prefill chunk of n tokens after k prompt tokens, attention-pair work is
`n*k + n*(n+1)/2`. Summed over a prompt, this equals `P*(P+1)/2` regardless of chunk partition.
Decode context work sums prompt length plus the number of emitted outputs for selected decodes.
These counts permit context sensitivity without simulating kernels.

### Analytical

`T = h + a*P_b + b*D_b + q*D_b^2 + c*attention_pairs + d*decode_context_tokens`.

Defaults: h=0.10 ms, a=0.018 ms/token, b=0.22 ms/sequence, q=0.003 ms/sequence²,
c=d=0. Its additive service and quadratic decode pressure do not inherently reward chunking.
Repeated small chunks can add overhead or change decode batch pressure.

### Piecewise

Compute demand C uses 0.025 ms per prefill token for the first 256 selected tokens and 0.010 ms
thereafter, plus `1e-7 * attention_pairs` ms. Memory demand M uses a shared 0.60 ms decode cost,
0.040 ms/sequence for the first 16 sequences, 0.10 thereafter, and `5e-6 * decode_context_tokens` ms.
Shared decode cost applies only when at least one decode is present.

`T = 0.15 + 0.10*R_b + max(C + 0.020*D_b, M + 0.001*P_b) + f*min(C,M)`.

Default f=0.25. The sweep also uses f=0 and f=1. This is a roofline-inspired accounting proxy.
It is not evidence that two real kernels overlap by these amounts. Small-chunk penalties, context
costs, and decode sharing are assumptions to inspect, not calibrated hardware parameters.

### Trace interpolation

`TraceCalibratedCostModel` interpolates a complete rectangular `(P_b,D_b)` timing grid bilinearly.
Input provenance must say `synthetic` or `measured`. Out-of-grid batches, duplicate coordinates,
and missing grid corners fail. No extrapolation or nearest-neighbor substitution occurs.

The bundled grid is generated from the analytical equation and is explicitly synthetic.
Its interpolation approximates the quadratic D_b term between grid points. Context length and
prefill-request count are absent from its axes. Even a measured grid would generate simulated
scheduler results; it would not turn the simulator into a hardware benchmark.
See [calibration format](CALIBRATION.md).

## Metrics

For request i, let a_i be arrival, s_i first prefill start, f_i prefill finish, t_i,j output token j,
and c_i completion. All latency fields use milliseconds.

| Metric | Definition |
| --- | --- |
| TTFT | `t_i,1 - a_i` |
| TPOT | `(c_i - t_i,1)/(O_i - 1)` for O_i>1; null otherwise |
| ITL | Each individual `t_i,j - t_i,j-1`; pooled across token gaps |
| E2E | `c_i - a_i` |
| Prefill wait | `s_i - a_i` |
| Decode service | Sum of complete batch durations in which i was selected for decode |
| Decode elapsed | `c_i - f_i` |
| Decode descheduled | `max(0, decode_elapsed - decode_service)` |
| Maximum service gap | Largest interval from arrival/previous service end to next selected service start |

Decode service includes the full mixed-batch duration. It is not exclusive GPU time attributable
to one request. The descheduled metric excludes interference while selected; ITL captures that
interference. Means and P50/P95/P99 are reported separately for request-level metrics and pooled ITL.
Percentiles use linear interpolation at `(N-1)*q`. Undefined values are excluded, with counts retained.
P95 TPOT is a percentile of per-request averages, not a token-level P95 stall.

Finite-cohort throughput divides completed requests or logical tokens by
`max(completion) - min(arrival)`, including idle time and final drain. Logical token throughput
counts P+O. Processed-token throughput counts P+O-1 with the default first-token convention.

Arrival-window throughput counts completions through the final arrival and divides by that arrival
window. Backlog at the final arrival and drain time are reported separately. Finite traces do not
establish steady-state capacity. At high offered load, latency can grow while finite-cohort throughput
looks stable. Closed-loop runs have policy-dependent arrivals and no prescribed offered load.

Iteration mean, population standard deviation, and coefficient of variation describe batch durations.
Histograms round durations to 0.001 ms for plotting only. Request metrics retain full precision.
Server busy fraction is total service time / cohort duration. Batch-slot occupancy is the
service-time-weighted selected sequence count / batch capacity. Neither is measured GPU utilization.
No pipeline is modeled, so neither metric is called a pipeline bubble.

## Fairness and statistics

The long-wait count is a **starvation proxy**: requests whose maximum service gap exceeds 1,000 ms.
It is not proof of starvation under an infinite workload. All finite requests must finish or the
engine fails at its iteration limit. The waiting-satisfaction Jain index uses
`q_i = 1/(1 + prefill_wait_i/1000)` and `(sum q_i)^2/(N*sum q_i^2)`. A high score can coexist with
unacceptable absolute waits. Inspect both the index and the wait metrics.

Five fixed workload seeds are the replicate units. Reported tables average per-seed metrics.
Uncertainty uses 2,000 bootstrap resamples of seeds, with 95% percentile intervals. Paired differences
and ratio-of-means changes resample matching seeds together. Request rows from one run are not
independent experiment replicates. Paired d_z is null when difference variance is zero.
No significance or universal ranking follows from these intervals. They omit cost-model uncertainty.
