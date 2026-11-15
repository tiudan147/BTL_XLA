from data.dataset import FSSDataset

benchmarks = [
    ("isic", "./datasets/isic2018-classwise"),
    ("lung", "./datasets/lungsegmentation"),
    ("fss", "./datasets/fss1000-a-1000-class-fewshot-segmentation"),
    ("deepglobe", "./datasets/deepglobe"),
    ("suim", "./datasets/suim-merged/suim_merged")
]

print("=== Checking all 5 datasets ===")
for name, path in benchmarks:
    try:
        FSSDataset.initialize(img_size=400, datapath=path)
        loader = FSSDataset.build_dataloader(name, bsz=1, nworker=0, fold=0, split='test', shot=1)
        batch = next(iter(loader))
        q_shape = tuple(batch["query_img"].shape)
        s_shape = tuple(batch["support_imgs"].shape)
        total_episodes = len(loader.dataset)
        print(f"[OK] {name.upper():10s} | Episodes: {total_episodes:5d} | Query: {q_shape} | Support: {s_shape}")
    except Exception as e:
        print(f"[FAIL] {name.upper():10s} | Error: {e}")

