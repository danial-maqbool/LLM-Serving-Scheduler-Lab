from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_scheduler_lab.models import SimulatorConfig
from llm_scheduler_lab.policies import ChunkedPrefillPolicy, ContinuousBatchingPolicy, FCFSPolicy
from llm_scheduler_lab.simulator import Simulator
from llm_scheduler_lab.workload import WorkloadConfig, generate_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline LLM serving scheduler experiments")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output", default="results/baseline.csv")
    args = parser.parse_args()

    config_path = Path(args.config)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    workload_cfg = WorkloadConfig(**data["workload"])
    sim_cfg = SimulatorConfig(**data["simulator"])
    requests = generate_workload(workload_cfg)

    policies = [FCFSPolicy(), ContinuousBatchingPolicy()]
    policies.extend(ChunkedPrefillPolicy(chunk_size=size) for size in data["chunk_sizes"])

    rows: list[dict[str, object]] = []
    for policy in policies:
        result = Simulator(sim_cfg).run(requests, policy)
        row: dict[str, object] = {"policy": policy.name}
        if isinstance(policy, ChunkedPrefillPolicy):
            row["chunk_size"] = policy.chunk_size
        else:
            row["chunk_size"] = ""
        row.update(result.metrics)
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['policy']:>20} chunk={str(row['chunk_size']):>4} "
            f"TTFT_p95={row['ttft_p95_ms']:.2f}ms "
            f"TPOT_p95={row['tpot_p95_ms']:.2f}ms "
            f"RPS={row['request_throughput_rps']:.2f}"
        )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
