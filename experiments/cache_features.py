"""Cache the 13 layer predictions for every episode of a CD-FSS benchmark.

The test-time adapter is by far the most expensive part of ABCDFSS.  This
script runs it once per (benchmark, shot, seed) and stores everything that the
output-stage experiments need:

  layers          [13, 100, 100] float16  layer predictions at h0/w0
  pseudo_layers   [k, 13, 100, 100] float16  support->augmented-support maps
  q_img           [3, 400, 400] float16  ImageNet-normalised query image
  s_img           [k, 3, 400, 400] float16  ImageNet-normalised support images
  q_mask          [400, 400] uint8
  s_mask          [k, 400, 400] uint8
  s_mask_aug_mean scalar float32
  class_id        int64

The layer predictions are stored at the 100x100 resolution at which the
baseline fuses them.  Fusing there and upsampling the fused map to 400x400 is
exactly equivalent to the baseline for all linear fusion rules and preserves
the baseline's non-linear fusion semantics as well.

Usage:
    python -m experiments.cache_features --benchmark lung \
        --datapath /path/to/lungsegmentation --shot 1 --seed 0 \
        --cache-dir experiments/cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import runner  # noqa: E402
from experiments import common  # noqa: E402
from utils import crfhelper  # noqa: E402


def build_args(ns: argparse.Namespace) -> argparse.Namespace:
    ns.backbone = "resnet50"
    ns.nworker = 0  # deterministic episode sampling
    ns.bsz = 1
    ns.fold = 0
    ns.adapt_to = getattr(ns, "adapt_to", "first-episode")
    ns.postprocessing = "off"
    ns.fusion_method = "mean"
    ns.threshold_method = "pred_mean"
    ns.logpath = "cache_only"
    ns.verbosity = 0
    return ns


def _episode_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"ep{idx:06d}.npz"


def _save_episode(path: Path, arrays: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _load_meta(cache_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _write_meta(cache_dir: Path, meta: Dict[str, Any]) -> None:
    tmp = cache_dir / "meta.json.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, cache_dir / "meta.json")


def cache_benchmark(
    benchmark: str,
    datapath: str,
    shot: int,
    seed: int,
    cache_dir: Path,
    max_episodes: Optional[int] = None,
    resume: bool = True,
    device: Optional[str] = None,
    report_every: int = 25,
) -> Dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cache_dir = Path(cache_dir) / f"{benchmark}_shot{shot}_seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ns = argparse.Namespace(
        benchmark=benchmark,
        datapath=datapath,
        nshot=shot,
        adapt_to="first-episode",
        logpath="cache_only",
        verbosity=0,
    )
    ns = build_args(ns)
    runner.set_args(ns)

    from data.dataset import FSSDataset
    FSSDataset.initialize(img_size=common.IMG_SIZE, datapath=datapath)
    dataset = FSSDataset.datasets[benchmark](
        datapath,
        fold=0,
        transform=FSSDataset.transform,
        split="test",
        shot=shot,
    )
    indices = common.select_indices(len(dataset), max_episodes)
    nclass = common.nclass_for_dataset(dataset)
    class_ids = common.class_ids_for_dataset(dataset)

    existing_meta = _load_meta(cache_dir)
    if existing_meta and existing_meta.get("episode_indices") != indices:
        if resume:
            raise RuntimeError(
                f"Cache {cache_dir} exists with a different episode selection. "
                "Delete it or use --no-resume."
            )
    if not resume and cache_dir.exists():
        for p in cache_dir.glob("ep*.npz"):
            p.unlink()

    config = runner.makeConfig()
    config.featext.fit_every_episode = ns.adapt_to == "every-episode"
    feat_maker = runner.makeFeatureMaker(dataset, config, device=device)
    fit_every_episode = bool(config.featext.fit_every_episode)

    print(
        f"[cache] benchmark={benchmark} shot={shot} seed={seed} "
        f"episodes={len(indices)}/{len(dataset)} classes={len(class_ids)} device={device}",
        flush=True,
    )

    first_idx = common.first_index_per_class(dataset)
    if not fit_every_episode:
        # Fit the per-class adapter on a canonical, deterministic episode so
        # that cache generation is resumable and reproducible.
        print(f"[cache] fitting {len(first_idx)} per-class adapters", flush=True)
        for j, cid in enumerate(sorted(first_idx, key=lambda c: first_idx[c])):
            idx = first_idx[cid]
            common.seed_everything(seed + idx)
            batch = default_collate([dataset[idx]])
            sseval = runner.SingleSampleEval(batch, feat_maker)
            sseval.taskAdapt()
            del sseval, batch
            if device == "cuda" and (j + 1) % 50 == 0:
                torch.cuda.empty_cache()

    start = time.time()
    written = 0
    skipped = 0
    for n, idx in enumerate(indices):
        path = _episode_path(cache_dir, idx)
        if resume and path.exists():
            skipped += 1
            continue
        common.seed_everything(seed + idx)
        batch = default_collate([dataset[idx]])
        sseval = runner.SingleSampleEval(batch, feat_maker)
        sseval.taskAdapt()

        layers = sseval.damat_comp.calc_q_pred_coarses(
            sseval.task_adapted[0], sseval.task_adapted[1], sseval.s_mask
        )[0]

        # Pseudo episode used by Algorithm 1 (support as query, augmented
        # support as support).  The augmentations are the ones created when the
        # adapter was fitted, exactly as in utils/crfhelper.crf_is_good.
        s_feat = [f[0] for f in sseval.task_adapted[1] if f is not None]
        s_feataug, s_maskaug = crfhelper.get_augmentations_and_features(sseval)
        # s_feat/s_feataug already had the first l0 layers removed by
        # getTaskAdaptedFeats, hence l0=0 here (13 layers -> 13 maps).
        pseudo_layers = sseval.damat_comp.calc_q_pred_coarses(
            s_feat, s_feataug, s_maskaug, l0=0
        )

        q_mask = batch["query_mask"][0].detach().cpu().numpy().astype(np.uint8)
        s_mask = sseval.s_mask[0].detach().cpu().numpy().astype(np.uint8)
        q_img = sseval.q_img[0].detach().half().cpu().numpy()
        s_img = sseval.s_img[0].detach().half().cpu().numpy()

        arrays = {
            "layers": layers.detach().half().cpu().numpy(),
            "pseudo_layers": pseudo_layers.detach().half().cpu().numpy(),
            "q_img": q_img,
            "s_img": s_img,
            "q_mask": q_mask,
            "s_mask": s_mask,
            "s_mask_aug_mean": np.asarray(
                float(s_maskaug.detach().float().mean().item()), dtype=np.float32
            ),
            "class_id": np.asarray(int(sseval.class_id), dtype=np.int64),
            "query_name": np.asarray(str(batch.get("query_name", ""))),
            "support_names": np.asarray([str(x) for x in batch.get("support_names", [])]),
        }
        _save_episode(path, arrays)
        written += 1
        del sseval, batch, layers, pseudo_layers
        if device == "cuda" and (n + 1) % 100 == 0:
            torch.cuda.empty_cache()
        if written % report_every == 0 or n == len(indices) - 1:
            elapsed = max(time.time() - start, 1e-6)
            rate = written / elapsed
            remaining = max(len(indices) - n - 1, 0)
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"[cache] {n + 1}/{len(indices)} written={written} skipped={skipped} "
                f"{rate:.2f} ep/s ETA={eta / 60:.1f} min",
                flush=True,
            )

    meta = {
        "benchmark": benchmark,
        "datapath": str(datapath),
        "shot": int(shot),
        "seed": int(seed),
        "img_size": common.IMG_SIZE,
        "n_episodes": len(indices),
        "episode_indices": indices,
        "nclass": nclass,
        "class_ids": class_ids,
        "code_commit": common.git_commit(),
        "torch": torch.__version__,
        "device": device,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_meta(cache_dir, meta)
    total_size = sum(p.stat().st_size for p in cache_dir.glob("ep*.npz"))
    print(
        f"[cache] done {benchmark}: {written} written, {skipped} skipped, "
        f"{total_size / 1e9:.2f} GB in {cache_dir}",
        flush=True,
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True,
                        choices=["deepglobe", "isic", "lung", "fss", "suim"])
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="0 or >= dataset size means all episodes")
    parser.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / "experiments" / "cache"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--report-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_benchmark(
        benchmark=args.benchmark,
        datapath=args.datapath,
        shot=args.shot,
        seed=args.seed,
        cache_dir=Path(args.cache_dir),
        max_episodes=args.max_episodes or None,
        resume=not args.no_resume,
        device=args.device,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
