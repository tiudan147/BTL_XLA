"""Run one experiment group (A, B, C or a combined set) from a cached dataset.

Usage:
    python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 \
        --group A --cache-dir experiments/cache --results-dir experiments/results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import common  # noqa: E402


def _load_meta(cache_dir: Path, benchmark: str, shot: int, seed: int) -> Dict[str, Any]:
    path = cache_dir / f"{benchmark}_shot{shot}_seed{seed}" / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at {path}. Run experiments.cache_features first."
        )
    with open(path) as f:
        return json.load(f)


def _episode_path(cache_dir: Path, benchmark: str, shot: int, seed: int, idx: int) -> Path:
    return cache_dir / f"{benchmark}_shot{shot}_seed{seed}" / f"ep{idx:06d}.npz"


def _load_episode(path: Path) -> Dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "layers": torch.from_numpy(data["layers"]).float(),
            "pseudo_layers": torch.from_numpy(data["pseudo_layers"]).float(),
            "q_img": torch.from_numpy(data["q_img"]).float(),
            "s_img": torch.from_numpy(data["s_img"]).float(),
            "q_mask": torch.from_numpy(data["q_mask"]).float(),
            "s_mask": torch.from_numpy(data["s_mask"]).float(),
            "s_mask_aug_mean": float(data["s_mask_aug_mean"]),
            "class_id": int(data["class_id"]),
        }


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _top_names(rows: List[Dict[str, Any]], key: str, k: int = 2,
               exclude: Optional[Sequence[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    candidates = [r for r in rows if r["name"] not in exclude]
    candidates.sort(key=lambda r: float(r[key]), reverse=True)
    return [r["name"] for r in candidates[:k]]


def build_combo_configs(results_dir: Path, benchmark: str, shot: int, seed: int) -> List[Dict[str, Any]]:
    """Pick promising A/B/C variants from group results and combine them."""
    stem = f"{benchmark}_shot{shot}_seed{seed}_group"
    rows_a = _read_csv(results_dir / f"{stem}A.csv")
    rows_b = _read_csv(results_dir / f"{stem}B.csv")
    rows_c = _read_csv(results_dir / f"{stem}C.csv")
    if not (rows_a and rows_b and rows_c):
        print("[combo] group A/B/C results missing; nothing to combine")
        return []

    by_name: Dict[str, Dict[str, Any]] = {}
    for cfgs in (
        common.group_a_configs(),
        common.group_b_configs(),
        common.group_c_configs(),
    ):
        for cfg in cfgs:
            by_name[cfg["name"]] = cfg

    fusions = list(dict.fromkeys(
        _top_names(rows_a, "mIoU", 2) + _top_names(rows_a, "FB-IoU", 2)
    ))
    thresholds = list(dict.fromkeys(
        _top_names(rows_b, "mIoU", 2) + _top_names(rows_b, "FB-IoU", 2)
    ))
    # Keep the combination cheap: prefer non-CRF post-processing, but allow the
    # best CRF variant if it is clearly at the top.
    postprocs = list(dict.fromkeys(
        _top_names(rows_c, "mIoU", 2) + _top_names(rows_c, "FB-IoU", 2)
    ))

    combos: List[Dict[str, Any]] = []
    seen = set()

    def add(fusion_name: str, threshold_name: str, postproc_name: str) -> None:
        fcfg = by_name.get(fusion_name)
        tcfg = by_name.get(threshold_name)
        pcfg = by_name.get(postproc_name)
        if not (fcfg and tcfg and pcfg):
            return
        name = f"X_{fusion_name}__{threshold_name}__{postproc_name}"
        if name in seen:
            return
        seen.add(name)
        combos.append(
            common._cfg(
                name,
                "X",
                fusion=fcfg["fusion"],
                fusion_params=fcfg["fusion_params"],
                threshold=tcfg["threshold"],
                threshold_params=tcfg["threshold_params"],
                postproc=pcfg["postproc"],
                postproc_params=pcfg["postproc_params"],
                note=f"combo {fusion_name} + {threshold_name} + {postproc_name}",
            )
        )

    for f in fusions:
        for t in thresholds:
            add(f, t, "C1_off")
            for p in postprocs:
                add(f, t, p)
    # A compact fallback set even if the result files are tiny.
    if not combos:
        for f in fusions[:1]:
            for t in thresholds[:1]:
                for p in postprocs[:1]:
                    add(f, t, p)
    return combos[:18]


def run_group(
    benchmark: str,
    shot: int,
    seed: int,
    group: str,
    cache_dir: Path,
    results_dir: Path,
    crf_max_episodes: int = 200,
    max_episodes: int = 0,
    device: str = "cpu",
    report_every: int = 50,
) -> Dict[str, Any]:
    meta = _load_meta(cache_dir, benchmark, shot, seed)
    all_indices: List[int] = list(meta["episode_indices"])
    if max_episodes and max_episodes > 0:
        all_indices = common.select_subset(all_indices, max_episodes)
    class_ids = list(meta["class_ids"])
    nclass = int(meta["nclass"])

    if group.upper() == "X":
        configs = build_combo_configs(results_dir, benchmark, shot, seed)
        group_label = "X"
    else:
        configs = common.configs_for_group(group)
        group_label = group.upper()
    if not configs:
        raise ValueError(f"No configurations for group {group!r}")

    if any(common.is_crf_config(c) for c in configs):
        try:
            import pydensecrf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "This group needs pydensecrf. Run experiments/install_pydensecrf.py first."
            ) from exc

    crf_indices = all_indices
    if crf_max_episodes and crf_max_episodes > 0:
        crf_indices = common.select_subset(all_indices, crf_max_episodes)
    # Group C and combined runs that contain a CRF variant use the CRF subset
    # for *all* variants so that C1/C0/C2 are paired on exactly the same
    # episodes.  Groups A/B are unaffected.
    if group_label in {"C", "X"} and crf_max_episodes and crf_max_episodes > 0:
        eval_indices = crf_indices
    else:
        eval_indices = all_indices
    crf_index_set = set(crf_indices)

    acc = common.MetricAccumulator(nclass)
    per_config_seconds: Dict[str, float] = {c["name"]: 0.0 for c in configs}
    per_config_episodes: Dict[str, int] = {c["name"]: 0 for c in configs}

    print(
        f"[run] benchmark={benchmark} shot={shot} seed={seed} group={group_label} "
        f"configs={len(configs)} episodes={len(eval_indices)} device={device}",
        flush=True,
    )

    start = time.time()
    for n, idx in enumerate(eval_indices):
        path = _episode_path(cache_dir, benchmark, shot, seed, idx)
        if not path.exists():
            print(f"[run] WARNING missing cache file {path}; skipping")
            continue
        ep = _load_episode(path)
        for cfg in configs:
            if common.is_crf_config(cfg) and idx not in crf_index_set:
                continue
            t0 = time.perf_counter()
            try:
                pred = common.apply_config(
                    ep["layers"],
                    ep["q_img"],
                    ep["s_mask"],
                    ep["s_mask_aug_mean"],
                    ep["pseudo_layers"],
                    ep["s_img"],
                    cfg,
                )
            except Exception as exc:
                print(f"[run] ERROR cfg={cfg['name']} idx={idx}: {exc}")
                raise
            dt = time.perf_counter() - t0
            acc.update(cfg["name"], pred, ep["q_mask"], ep["class_id"], dt)
            per_config_seconds[cfg["name"]] += dt
            per_config_episodes[cfg["name"]] += 1
        del ep
        if report_every and ((n + 1) % report_every == 0 or n == len(eval_indices) - 1):
            elapsed = time.time() - start
            rate = (n + 1) / max(elapsed, 1e-6)
            eta = (len(eval_indices) - n - 1) / max(rate, 1e-6)
            print(
                f"[run] {n + 1}/{len(eval_indices)} {rate:.2f} ep/s "
                f"ETA={eta / 60:.1f} min",
                flush=True,
            )

    summary = acc.summarize(class_ids)
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        name = cfg["name"]
        if name not in summary:
            continue
        s = summary[name]
        row = {
            "group": group_label,
            "shot": int(shot),
            "seed": int(seed),
            "name": name,
            "fusion": cfg["fusion"],
            "fusion_params": json.dumps(cfg.get("fusion_params", {})),
            "threshold": cfg["threshold"],
            "threshold_params": json.dumps(cfg.get("threshold_params", {})),
            "postproc": cfg["postproc"],
            "postproc_params": json.dumps(cfg.get("postproc_params", {})),
            "mIoU": round(s["mIoU"], 4),
            "FB-IoU": round(s["FB-IoU"], 4),
            "FG_pred": round(s["FG_pred"], 4),
            "FG_gt": round(s["FG_gt"], 4),
            "n_episodes": s["n_episodes"],
            "seconds": round(s["seconds"], 3),
            "per_class_iou": json.dumps(s["per_class_iou"]),
            "note": cfg.get("note", ""),
        }
        rows.append(row)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{benchmark}_shot{shot}_seed{seed}_group{group_label}"
    csv_path = results_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(results_dir / f"{stem}.json", "w") as f:
        json.dump(
            {
                "meta": meta,
                "group": group_label,
                "eval_indices": eval_indices,
                "results": rows,
                "elapsed_seconds": time.time() - start,
            },
            f,
            indent=2,
        )
    print(f"[run] wrote {csv_path} ({len(rows)} configs)", flush=True)
    return {"rows": rows, "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True,
                        choices=["deepglobe", "isic", "lung", "fss", "suim"])
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group", type=str, default="A",
                        help="A, B, C or X (combined)")
    parser.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "experiments" / "cache"))
    parser.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "experiments" / "results"))
    parser.add_argument("--crf-max-episodes", type=int, default=200,
                        help="0 = evaluate CRF variants on all cached episodes")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="optionally subsample the cached episodes further")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--report-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_group(
        benchmark=args.benchmark,
        shot=args.shot,
        seed=args.seed,
        group=args.group,
        cache_dir=Path(args.cache_dir),
        results_dir=Path(args.results_dir),
        crf_max_episodes=args.crf_max_episodes,
        max_episodes=args.max_episodes,
        device=args.device,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
