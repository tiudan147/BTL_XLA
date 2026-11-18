# Output-stage experiments (ABCDFSS)

Thư mục này chứa pipeline chạy **toàn bộ** thí nghiệm khâu đầu ra trong
`docs/README.md`: gộp tầng (A), cắt ngưỡng (B), hậu xử lý (C) và kết hợp (X).

## Vì sao có cache?

Adapter test-time của ABCDFSS là phần đắt nhất. Pipeline chạy adapter **một lần**
cho mỗi `(benchmark, shot, seed)` rồi lưu:

- `layers`: 13 bản đồ `q̂_pred_l` ở 100×100 (đúng độ phân giải baseline gộp),
- `pseudo_layers`: 13 bản đồ của pseudo-episode dùng cho Thuật toán 1,
- ảnh query/support đã chuẩn hoá ImageNet, mặt nạ GT/support,
- `class_id` và tỉ lệ foreground của support.

Mọi biến thể A/B/C sau đó chạy offline trên **cùng** đặc trưng đã thích ứng.

## Chạy nhanh

```bash
# Toàn bộ 5 tập, 600 episode phân tầng/tập, 1-shot, seed 0
python -m experiments.pipeline \
  --datapath /path/to/cdfss-benchmark-dataset \
  --benchmarks deepglobe,isic,lung,fss,suim \
  --shot 1 --seed 0 \
  --max-episodes 600 --crf-max-episodes 200 \
  --groups A,B,C,X --commit
```

Pipeline tự:

1. cache adapter + layer predictions (bỏ qua nếu đã có),
2. chạy nhóm A, B, C, X,
3. commit kết quả từng nhóm,
4. sinh `docs/exp.md` với link commit.

`--max-episodes 0` chạy toàn bộ test split. `--crf-max-episodes 0` chạy CRF trên
toàn bộ episode (rất chậm vì DenseCRF chạy CPU).

## Chạy từng phần

```bash
python -m experiments.cache_features --benchmark lung \
  --datapath /path/to/lungsegmentation --shot 1 --seed 0 --max-episodes 600

python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group A
python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group B
python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group C
python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group X

python -m experiments.report --shot 1 --seed 0 --out docs/exp.md
```

## Colab

Mở `experiments/colab_experiments.ipynb`, chọn GPU (T4), chạy lần lượt các cell.
Notebook tự clone repo, cài `pydensecrf` (có patch cho Python ≥ 3.11), tải dữ
liệu qua `kagglehub`, chạy pipeline và in link commit.

## Cấu trúc file

| File | Chức năng |
|---|---|
| `common.py` | fusion / threshold / post-processing / metrics |
| `cache_features.py` | sinh cache adapter + 13 layer predictions |
| `run_experiments.py` | chạy một nhóm trên cache |
| `pipeline.py` | orchestration + commit |
| `report.py` | sinh `docs/exp.md` |
| `install_pydensecrf.py` | build `pydensecrf` trên Python 3.11+ |
| `colab_experiments.ipynb` | notebook Colab một chạm |

## Lưu ý

- Kết quả trong `docs/exp.md` là **tự chạy trên tập con phân tầng**, không phải
  Bảng 4 của bài báo. Luôn so sánh với baseline tự chạy (A0+B0+C1).
- Nhóm C/CRF mặc định chỉ 200 episode/tập; nhóm A/B chạy 600 episode/tập.
- Cần báo cáo thêm seed và 5-shot nếu muốn kết luận chắc chắn.
