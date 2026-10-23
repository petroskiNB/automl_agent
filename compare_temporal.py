"""Run a predefined three-way temporal ablation, without an API key or search."""
import argparse
import json
import os
from pathlib import Path

from solution import discover_files, compatible_manifest, fingerprint
from temporal_research import TemporalResearch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base-id", type=int, required=True)
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--key-columns", nargs="+", required=True)
    parser.add_argument("--group-columns", nargs="+", required=True)
    parser.add_argument("--validation-periods", nargs="+", type=float, required=True)
    parser.add_argument("--export", action="store_true", help="Refit winner to a separate candidate file")
    parser.add_argument("--history-gate", action="store_true", help="Also test history for seen keys, baseline otherwise")
    args = parser.parse_args()
    folder, run = args.task.resolve().parent, args.run.resolve()
    if not run.is_relative_to(folder / "agent_runs"):
        raise ValueError("Run must belong to the given task")
    files = discover_files(folder)
    state = json.loads((run / "research/state.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if not compatible_manifest(manifest, fingerprint(args.task.resolve(), files), state):
        raise ValueError("Source run fingerprint does not match current inputs/engine")
    lock = folder / ".agent.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(str(os.getpid()))
        worker = TemporalResearch(run / "research", files, folder)
        report = worker.run_temporal_suite(args.base_id, args.time_column, args.key_columns,
                                           args.group_columns, args.validation_periods, args.history_gate)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.export:
            print(json.dumps(worker.finalize_temporal(report["suite_id"]), ensure_ascii=False, indent=2), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
