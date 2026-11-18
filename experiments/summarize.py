"""Print a compact summary of the experiment CSVs.

Usage:
    python -m experiments.summarize --results-dir experiments/results --shot 1 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "experiments" / "results"))
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = report.load_results(results_dir, args.shot, args.seed)
    indexed = {g: report.index_rows(results[g]) for g in report.GROUPS}

    print("== Baseline ==")
    for name in ("A0_mean", "C1_off", "C0_crf_dynamic"):
        for g in report.GROUPS:
            val = report.avg_metric(indexed.get(g, {}), name, "mIoU")
            if val is not None:
                fb = report.avg_metric(indexed[g], name, "FB-IoU")
                fg = report.avg_metric(indexed[g], name, "FG_pred")
                print(f"  {name:18s} group={g}  mIoU={val:.2f}  FB={fb:.2f}  %FG={fg:.2f}")
                break

    for group in report.GROUPS:
        by_bench = indexed.get(group, {})
        if not by_bench:
            continue
        names: List[str] = []
        for b in report.BENCHMARKS:
            for name in by_bench.get(b, {}):
                if name not in names:
                    names.append(name)
        scored = []
        for name in names:
            miou = report.avg_metric(by_bench, name, "mIoU")
            fb = report.avg_metric(by_bench, name, "FB-IoU")
            fg = report.avg_metric(by_bench, name, "FG_pred")
            if miou is not None:
                scored.append((miou, fb, fg, name))
        scored.sort(reverse=True)
        print(f"\n== Group {group}: top 8 by average mIoU ==")
        for miou, fb, fg, name in scored[:8]:
            print(f"  {name:55s} mIoU={miou:6.2f}  FB={fb:6.2f}  %FG={fg:6.2f}")
        scored.sort(key=lambda x: (x[1] or 0), reverse=True)
        print(f"== Group {group}: top 5 by average FB-IoU ==")
        for miou, fb, fg, name in scored[:5]:
            print(f"  {name:55s} mIoU={miou:6.2f}  FB={fb:6.2f}  %FG={fg:6.2f}")


if __name__ == "__main__":
    main()
