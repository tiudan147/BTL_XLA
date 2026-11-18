"""End-to-end experiment pipeline: cache -> groups A/B/C/X -> docs/exp.md.

This is the entry point used by the Colab notebook.  It is resumable: cached
episodes and finished group CSVs are reused unless --force is given.

Example (Colab):
    python -m experiments.pipeline --datapath "$DATA_ROOT" \
        --benchmarks deepglobe,isic,lung,fss,suim \
        --shot 1 --seed 0 --max-episodes 600 --crf-max-episodes 200 \
        --groups A,B,C,X --commit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import cache_features, common, report, run_experiments  # noqa: E402

BENCHMARKS = ["deepglobe", "isic", "lung", "fss", "suim"]
SUBDIRS = {
    "deepglobe": "deepglobe",
    "isic": "isic2018-classwise",
    "lung": "lungsegmentation",
    "fss": "fss1000",
    "suim": "suim",
}


def resolve_datapath(benchmark: str, datapath: str) -> str:
    """Accept either the Kaggle root (containing all five datasets) or a
    benchmark-specific directory."""
    p = Path(datapath).expanduser()
    sub = SUBDIRS[benchmark]
    if benchmark == "fss":
        if (p / "fss1000").exists():
            return str(p / "fss1000")
        if p.name == "fss1000":
            return str(p)
        if p.name == "FSS-1000":
            return str(p.parent)
        return str(p)
    if (p / sub).exists():
        return str(p / sub)
    return str(p)


def _cache_complete(cache_dir: Path, benchmark: str, shot: int, seed: int) -> bool:
    root = cache_dir / f"{benchmark}_shot{shot}_seed{seed}"
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    expected = meta.get("n_episodes", 0)
    have = len(list(root.glob("ep*.npz")))
    return have >= expected > 0


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _commit(paths: List[str], message: str, commit: bool) -> Optional[str]:
    if not commit:
        return None
    for p in paths:
        if not (REPO_ROOT / p).exists():
            print(f"[git] skip missing path {p}")
            continue
        subprocess.run(["git", "add", p], cwd=str(REPO_ROOT), check=True)
    # Nothing staged? do not create an empty commit.
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=str(REPO_ROOT)
    ).returncode
    if staged == 0:
        print(f"[git] no changes for: {message}")
        return common.git_commit()
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=nhanlx05-dev",
            "-c",
            "user.email=nhanlx05-dev@users.noreply.github.com",
            "commit",
            "-m",
            message,
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    sha = common.git_commit()
    print(f"[git] {sha[:8]} {message}")
    return sha


def pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for b in benchmarks:
        if b not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark {b!r}; choose from {BENCHMARKS}")
    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
    for g in groups:
        if g not in {"A", "B", "C", "X"}:
            raise ValueError(f"Unknown group {g!r}; choose from A,B,C,X")

    cache_dir = Path(args.cache_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    commits: Dict[str, str] = {}
    commits_path = results_dir / "commits.json"
    if commits_path.exists() and not args.force:
        with open(commits_path) as f:
            commits = json.load(f)

    resolved = {b: resolve_datapath(b, args.datapath) for b in benchmarks}
    run_info = {
        "benchmarks": benchmarks,
        "groups": groups,
        "shot": args.shot,
        "seed": args.seed,
        "max_episodes": args.max_episodes,
        "crf_max_episodes": args.crf_max_episodes,
        "datapath": args.datapath,
        "resolved_datapaths": resolved,
        "cache_dir": str(cache_dir),
        "results_dir": str(results_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # ------------------------------------------------------------------
    # 1. Cache every benchmark once.
    # ------------------------------------------------------------------
    for b in benchmarks:
        if _cache_complete(cache_dir, b, args.shot, args.seed) and not args.force:
            print(f"[pipeline] cache for {b} complete; skipping", flush=True)
            continue
        cache_features.cache_benchmark(
            benchmark=b,
            datapath=resolve_datapath(b, args.datapath),
            shot=args.shot,
            seed=args.seed,
            cache_dir=cache_dir,
            max_episodes=args.max_episodes or None,
            resume=not args.force,
            device=args.device,
            report_every=args.report_every,
        )

    # ------------------------------------------------------------------
    # 2. Run each group across all benchmarks, then commit that group.
    # ------------------------------------------------------------------
    for group in groups:
        for b in benchmarks:
            stem = f"{b}_shot{args.shot}_seed{args.seed}_group{group}"
            csv_path = results_dir / f"{stem}.csv"
            if csv_path.exists() and not args.force:
                print(f"[pipeline] {stem} exists; skipping (use --force to rerun)")
                continue
            run_experiments.run_group(
                benchmark=b,
                shot=args.shot,
                seed=args.seed,
                group=group,
                cache_dir=cache_dir,
                results_dir=results_dir,
                crf_max_episodes=args.crf_max_episodes,
                max_episodes=0,
                device="cpu",
                report_every=args.report_every,
            )
        paths = [
            str((results_dir / f"{b}_shot{args.shot}_seed{args.seed}_group{group}.csv").relative_to(REPO_ROOT))
            for b in benchmarks
        ] + [
            str((results_dir / f"{b}_shot{args.shot}_seed{args.seed}_group{group}.json").relative_to(REPO_ROOT))
            for b in benchmarks
        ]
        sha = _commit(
            [p for p in paths if (REPO_ROOT / p).exists()],
            f"exp({group.lower()}): group {group} results (shot={args.shot}, seed={args.seed}, "
            f"episodes={args.max_episodes or 'all'})",
            args.commit,
        )
        if sha:
            commits[group] = sha
            with open(commits_path, "w") as f:
                json.dump(commits, f, indent=2)
            # Include the updated commit map in the next group's commit.
            _commit([str(commits_path.relative_to(REPO_ROOT))],
                    f"exp(meta): record group {group} commit", args.commit)

    # ------------------------------------------------------------------
    # 3. Generate the experiment log and commit it with the commit map.
    # ------------------------------------------------------------------
    with open(commits_path, "w") as f:
        json.dump(commits, f, indent=2)
    report.generate(
        results_dir=results_dir,
        out_path=Path(args.out),
        shot=args.shot,
        seed=args.seed,
    )
    run_info["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_info["commits"] = commits
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(run_info, f, indent=2)
    final_sha = _commit(
        [
            "docs/exp.md",
            str(results_dir.relative_to(REPO_ROOT) / "run_config.json"),
            str(results_dir.relative_to(REPO_ROOT) / "commits.json"),
        ],
        f"docs(exp): experiment log (shot={args.shot}, seed={args.seed})",
        args.commit,
    )
    if final_sha:
        print("[pipeline] final commit:", common.commit_link(final_sha), flush=True)

    if args.push and args.commit:
        try:
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(REPO_ROOT), check=True)
            print("[pipeline] pushed to origin", flush=True)
        except Exception as exc:
            print(f"[pipeline] git push failed (run it manually): {exc}", flush=True)

    print("\n=== DONE ===")
    print(f"exp.md: {Path(args.out)}")
    for g, sha in commits.items():
        print(f"group {g}: {common.commit_link(sha)}")
    return {"commits": commits, "out": str(args.out)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=str, default="deepglobe,isic,lung,fss,suim")
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="0 = all episodes; 600 is a good Colab compromise")
    parser.add_argument("--crf-max-episodes", type=int, default=200,
                        help="0 = all episodes (very slow)")
    parser.add_argument("--groups", type=str, default="A,B,C,X")
    parser.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "experiments" / "cache"))
    parser.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "experiments" / "results"))
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "docs" / "exp.md"))
    parser.add_argument("--device", type=str, default=None,
                        help="device for adapter caching (default: cuda if available)")
    parser.add_argument("--force", action="store_true",
                        help="recompute cache and results")
    parser.add_argument("--commit", action="store_true",
                        help="commit each group's results with a linkable hash")
    parser.add_argument("--push", action="store_true",
                        help="push after the final commit (needs credentials)")
    parser.add_argument("--report-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline(args)


if __name__ == "__main__":
    main()
