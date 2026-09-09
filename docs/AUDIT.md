# Correctness audit

## Preserved starting point

The supplied archive was inspected before modification. Its seven tests passed locally.
Commit `5d74d582610e690954dac328f7fabbd092f57b78` preserves the executable starter.
The starter was not a validated hardware model. Its baseline numbers are not final results.

## Changes with a technical reason

| Finding | Correction | Regression evidence |
| --- | --- | --- |
| Monolithic prefill could exceed the nominal token budget | Enforce one hard combined token budget for every policy. Reject oversized monolithic inputs before running. | Oversize rejection and randomized token accounting |
| Only decode sequences were counted against the batch limit | Count prefill requests and decode requests together. | Invalid combined-plan test |
| No resident request limit | Keep admitted requests resident until completion. Validate every admission. | Randomized resident-capacity tests |
| First output always required an extra decode iteration | Default to the first output at final-prefill completion. Retain an explicit legacy `decode` convention. | Hand-calculated timestamps |
| One-output-token TPOT was zero | Return null and exclude it from TPOT aggregates. | One-token oracle |
| Repeated IDs could overwrite the state lookup | Reject duplicate IDs. | Duplicate-input test |
| NaN and fractional counts could pass validation | Validate finite times and integer token/resource counts. | Invalid-input parameterization |
| Idle jumps consumed the iteration limit | Count only executed work; permit completion at the exact limit. | Idle-jump and limit-boundary test |
| Arrival and length random draws were coupled | Use separate seeded streams. | Same token lengths across arrival modes and rates |
| Aggregate TPOT hid individual stalls | Also report pooled inter-token latency and maximum gaps. | Timestamp conservation |

The original comparison fixtures now use a common budget large enough for their largest
monolithic prompt. This changes the invalid fixture, not the intended assertions. Separate
negative tests require rejection when a monolithic prompt does not fit.

## Explicit modeling choices

The final-prefill first-token convention follows the request lifecycle described in
[the Sarathi-Serve paper overview](https://www.usenix.org/conference/osdi24/presentation/agrawal).
This is a correction relative to the starter, not a result reproduced from that paper.

Timing models receive only batch work, never a policy name. All coefficients are synthetic.
The trace adapter requires provenance, a complete grid, and in-range queries.
The bundled trace is an analytical demonstration, not measured calibration.

Optional KV capacity is a conservative reservation proxy: prompt tokens plus a configured
output limit. Admission does not inspect a request's actual future output length. Requests
above that configured limit are rejected. This is not a paged KV allocator.

Closed-loop experiments replay the same request lengths and think time. Arrival times
necessarily depend on each policy's completion times. They must not be described as
identical arrival-trace comparisons.

A finite drain cannot prove absence of starvation under infinite arrivals. The reported
starvation proxy is a configurable long-service-gap count. Iteration variation and slot
occupancy are not pipeline bubbles or measured GPU utilization.
