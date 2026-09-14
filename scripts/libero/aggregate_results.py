"""Aggregate disjoint TurboVLA LIBERO task-result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.inputs]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    suites = {payload["task_suite_name"] for payload in payloads}
    checkpoints = {payload["ckpt_path"] for payload in payloads}
    if len(suites) != 1 or len(checkpoints) != 1:
        raise ValueError("all inputs must use one suite and one checkpoint")
    protocol_fields = (
        "seed",
        "precision",
        "num_open_loop_steps",
        "num_trials_per_task",
    )
    protocol = {field: payloads[0].get(field) for field in protocol_fields}
    for payload in payloads[1:]:
        mismatched = {
            field: (protocol[field], payload.get(field))
            for field in protocol_fields
            if payload.get(field) != protocol[field]
        }
        if mismatched:
            raise ValueError(f"evaluation protocol mismatch across inputs: {mismatched}")

    tasks = []
    seen_task_ids = set()
    for payload in payloads:
        for task in payload["tasks"]:
            task_id = int(task["task_id"])
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate task id across inputs: {task_id}")
            seen_task_ids.add(task_id)
            tasks.append(task)
    tasks.sort(key=lambda item: int(item["task_id"]))
    total_episodes = sum(int(task["episodes"]) for task in tasks)
    total_successes = sum(int(task["successes"]) for task in tasks)
    expected_trials = protocol["num_trials_per_task"]
    if expected_trials is not None:
        incomplete = [
            (int(task["task_id"]), int(task["episodes"]))
            for task in tasks
            if int(task["episodes"]) != int(expected_trials)
        ]
        if incomplete:
            raise ValueError(f"incomplete task results: {incomplete}")
    result = {
        "script": "turbovla_libero_evaluation_aggregate",
        "ckpt_path": next(iter(checkpoints)),
        "task_suite_name": next(iter(suites)),
        "source_results": [str(path) for path in paths],
        **protocol,
        "tasks": tasks,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / max(total_episodes, 1),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{result['task_suite_name']}: {total_successes}/{total_episodes} "
        f"({100.0 * result['overall_success_rate']:.2f}%)"
    )


if __name__ == "__main__":
    main()
