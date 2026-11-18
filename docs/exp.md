## 6. Nhật ký thí nghiệm

> Kết quả dưới đây được tổng hợp từ `experiments/results/*.csv`; các file CSV/JSON gốc được giữ nguyên. Mỗi dòng có link tới commit chứa kết quả tương ứng. Các biến thể thắng đã được hợp nhất vào code chính (`core/denseaffinity.py`, `main.py`).

### 6.0 Giao thức chạy

- **Shot:** 1; **seed:** 0; **số episode/tập:** 600 (lấy mẫu phân tầng đều trên toàn bộ test split).
- **Thiết bị sinh cache adapter:** `cuda` (NVIDIA GeForce GTX 1650); code commit `21a47578f85759d84faa15a046aae4f362a79c47`.
- **Adapter:** đóng băng ResNet50 + adapter 2 lớp Conv 1×1, SGD 25 epoch, lr 0.01, 2 augmentation shear tối đa 20° (đúng cấu hình gốc). Adapter được fit một lần cho mỗi lớp (first-episode) và **dùng lại** cho mọi biến thể đầu ra.
- **Cache:** 13 bản đồ `q̂_pred_l` ở độ phân giải 100×100 (đúng nơi baseline gộp), cùng ảnh query/support, mặt nạ GT/support và pseudo-episode của Thuật toán 1. Nhờ vậy mọi biến thể được so sánh trên **cùng đặc trưng đã thích ứng**.
- **Chỉ số:** mIoU (trung bình IoU foreground theo lớp), FB-IoU (trung bình IoU background và foreground toàn cục), %FG (tỉ lệ foreground dự đoán). `mIoU TB`/`FB-IoU TB` là trung bình trên 5 tập (Deepglobe, ISIC, Chest X-ray, FSS-1000, SUIM).
- **Nhóm C và các biến thể CRF:** DenseCRF chạy CPU rất chậm (~1.7 s/ảnh cho 5 vòng), nên nhóm C được đánh giá trên một tập con phân tầng (200 episode/tập) và **mọi biến thể trong nhóm C dùng cùng tập con** để so sánh có ghép cặp. Nhóm A/B chạy trên toàn bộ episode đã cache.


> **Lưu ý:** đây là kết quả tự chạy trên một tập con phân tầng, không phải con số trong Bảng 4 của bài báo. Dùng cột Δ so với baseline tự chạy, không so trực tiếp với 60.7/69.2 của bài báo.


### 6.1 Baseline tự chạy

| Cấu hình | Deepglobe (m/FB) | ISIC (m/FB) | Chest X-ray (m/FB) | FSS-1000 (m/FB) | SUIM (m/FB) | TB 5 tập (m/FB) |
|---|---|---|---|---|---|---|
| Baseline no-pp (A0+B0+C1) | 41.9 / 47.0 | 45.2 / 59.1 | 79.8 / 86.4 | 70.7 / 80.2 | 34.5 / 51.6 | 54.4 / 64.9 |
| Baseline đầy đủ (A0+B0+C0) † | 42.6 / 47.6 | — | — | — | — | 42.6 / 47.6 |

† C0 dùng tập con CRF (xem mục 6.0); A0/A0+B0+C1 dùng toàn bộ episode đã cache.

Baseline no-pp trung bình 5 tập: **54.42 / 64.87**; baseline đầy đủ: **42.63 / 47.57**.


### 6.2 Nhóm A — Gộp tầng (Layer Fusion)

| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|---|
| `A0_mean` | baseline fusion (13-layer mean) | 1 | 0 | 54.42 | 64.87 | 33.54 | +0.00 | +0.00 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A1_low` | only Low (layers 1-4) | 1 | 0 | 43.61 | 55.55 | 38.00 | -10.81 | -9.33 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A2_mid` | only Mid (layers 5-10) | 1 | 0 | 52.14 | 63.39 | 32.99 | -2.28 | -1.49 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A3_high` | only High (layers 11-13) | 1 | 0 | 47.01 | 59.23 | 34.37 | -7.41 | -5.65 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A4_low_mid` | Low + Mid | 1 | 0 | 52.96 | 63.69 | 33.85 | -1.46 | -1.18 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A4_mid_high` | Mid + High | 1 | 0 | 53.41 | 64.26 | 32.94 | -1.01 | -0.62 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A4_low_high` | Low + High | 1 | 0 | 52.09 | 62.95 | 34.74 | -2.33 | -1.93 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A6_adaptive_otsu` | per-episode stage weights from Otsu separability | 1 | 0 | 51.01 | 62.27 | 34.21 | -3.41 | -2.61 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A6_adaptive_contrast` | per-episode stage weights from map contrast | 1 | 0 | 49.45 | 61.44 | 32.97 | -4.97 | -3.44 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A6_adaptive_entropy` | per-episode stage weights from histogram entropy | 1 | 0 | 48.14 | 60.31 | 32.50 | -6.29 | -4.56 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A7_norm_minmax` | per-layer min-max normalisation before mean | 1 | 0 | 54.25 | 64.34 | 34.94 | -0.18 | -0.53 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | thay đổi nhỏ |
| `A7_norm_zscore` | per-layer z-score normalisation before mean | 1 | 0 | 54.15 | 64.33 | 34.75 | -0.27 | -0.54 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | thay đổi nhỏ |
| `A8_max` | layer-wise max | 1 | 0 | 45.29 | 60.58 | 24.69 | -9.13 | -4.29 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A8_geomean` | layer-wise geometric mean | 1 | 0 | 54.61 | 64.65 | 34.96 | +0.19 | -0.23 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A8_median` | layer-wise median | 1 | 0 | 53.73 | 64.01 | 34.36 | -0.69 | -0.86 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w0.5_0.5_0.5` | stage weights L/M/H=0.5/0.5/0.5 | 1 | 0 | 54.18 | 64.68 | 33.70 | -0.24 | -0.19 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A5_w0.5_0.5_1` | stage weights L/M/H=0.5/0.5/1 | 1 | 0 | 52.88 | 63.74 | 33.71 | -1.54 | -1.13 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w0.5_0.5_2` | stage weights L/M/H=0.5/0.5/2 | 1 | 0 | 50.94 | 62.28 | 33.88 | -3.48 | -2.59 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w0.5_1_0.5` | stage weights L/M/H=0.5/1/0.5 | 1 | 0 | 54.33 | 64.84 | 33.42 | -0.09 | -0.04 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A5_w0.5_1_1` | stage weights L/M/H=0.5/1/1 | 1 | 0 | 53.73 | 64.41 | 33.38 | -0.69 | -0.46 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w0.5_1_2` | stage weights L/M/H=0.5/1/2 | 1 | 0 | 52.21 | 63.26 | 33.55 | -2.21 | -1.61 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w0.5_2_0.5` | stage weights L/M/H=0.5/2/0.5 | 1 | 0 | 53.97 | 64.64 | 33.21 | -0.45 | -0.23 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w0.5_2_1` | stage weights L/M/H=0.5/2/1 | 1 | 0 | 53.98 | 64.62 | 33.19 | -0.44 | -0.25 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w0.5_2_2` | stage weights L/M/H=0.5/2/2 | 1 | 0 | 53.33 | 64.13 | 33.24 | -1.10 | -0.75 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w1_0.5_0.5` | stage weights L/M/H=1/0.5/0.5 | 1 | 0 | 54.15 | 64.52 | 34.16 | -0.27 | -0.36 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | thay đổi nhỏ |
| `A5_w1_0.5_1` | stage weights L/M/H=1/0.5/1 | 1 | 0 | 53.58 | 64.20 | 34.01 | -0.84 | -0.67 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w1_0.5_2` | stage weights L/M/H=1/0.5/2 | 1 | 0 | 51.92 | 62.99 | 33.99 | -2.50 | -1.88 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w1_1_0.5` | stage weights L/M/H=1/1/0.5 | 1 | 0 | 54.42 | 64.80 | 33.80 | -0.00 | -0.07 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A5_w1_1_1` | stage weights L/M/H=1/1/1 | 1 | 0 | 54.18 | 64.68 | 33.70 | -0.24 | -0.19 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A5_w1_1_2` | stage weights L/M/H=1/1/2 | 1 | 0 | 52.88 | 63.74 | 33.71 | -1.54 | -1.13 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w1_2_0.5` | stage weights L/M/H=1/2/0.5 | 1 | 0 | 54.24 | 64.78 | 33.45 | -0.18 | -0.10 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A5_w1_2_1` | stage weights L/M/H=1/2/1 | 1 | 0 | 54.33 | 64.84 | 33.42 | -0.09 | -0.04 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A5_w1_2_2` | stage weights L/M/H=1/2/2 | 1 | 0 | 53.73 | 64.41 | 33.38 | -0.69 | -0.46 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w2_0.5_0.5` | stage weights L/M/H=2/0.5/0.5 | 1 | 0 | 52.74 | 63.28 | 34.88 | -1.68 | -1.60 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w2_0.5_1` | stage weights L/M/H=2/0.5/1 | 1 | 0 | 53.54 | 64.01 | 34.46 | -0.89 | -0.86 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w2_0.5_2` | stage weights L/M/H=2/0.5/2 | 1 | 0 | 52.89 | 63.64 | 34.28 | -1.53 | -1.23 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w2_1_0.5` | stage weights L/M/H=2/1/0.5 | 1 | 0 | 53.68 | 64.09 | 34.38 | -0.74 | -0.78 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w2_1_1` | stage weights L/M/H=2/1/1 | 1 | 0 | 54.15 | 64.52 | 34.16 | -0.27 | -0.36 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | thay đổi nhỏ |
| `A5_w2_1_2` | stage weights L/M/H=2/1/2 | 1 | 0 | 53.58 | 64.20 | 34.01 | -0.84 | -0.67 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | kém hơn baseline |
| `A5_w2_2_0.5` | stage weights L/M/H=2/2/0.5 | 1 | 0 | 54.13 | 64.56 | 33.88 | -0.29 | -0.31 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | thay đổi nhỏ |
| `A5_w2_2_1` | stage weights L/M/H=2/2/1 | 1 | 0 | 54.42 | 64.80 | 33.80 | -0.00 | -0.07 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |
| `A5_w2_2_2` | stage weights L/M/H=2/2/2 | 1 | 0 | 54.18 | 64.68 | 33.70 | -0.24 | -0.19 | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) | tương đương baseline |

### 6.3 Nhóm B — Cắt ngưỡng (Thresholding)

| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|---|
| `B0_pred_mean` | max(mean, Otsu) baseline | 1 | 0 | 54.42 | 64.87 | 33.54 | +0.00 | +0.00 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | tương đương baseline |
| `B1_mean` | mean threshold | 1 | 0 | 47.68 | 58.00 | 42.19 | -6.74 | -6.87 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | kém hơn baseline |
| `B2_otsus` | pure Otsu | 1 | 0 | 54.78 | 64.64 | 35.48 | +0.36 | -0.23 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | cải thiện (mIoU và FB-IoU không giảm) |
| `B3_multi_otsu` | 3-class Otsu, highest class | 1 | 0 | 46.19 | 61.57 | 21.59 | -8.23 | -3.30 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | kém hơn baseline |
| `B4_quantile_support` | quantile at support foreground ratio | 1 | 0 | 45.30 | 60.61 | 28.14 | -9.12 | -4.27 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | kém hơn baseline |
| `B5_adaptive_local` | local Gaussian adaptive threshold | 1 | 0 | 26.18 | 42.07 | 36.24 | -28.24 | -22.80 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | kém hơn baseline |
| `B6_lambda0` | lambda*Otsu + (1-lambda)*mean, lambda=0 | 1 | 0 | 47.68 | 58.00 | 42.19 | -6.74 | -6.87 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | kém hơn baseline |
| `B6_lambda0.25` | lambda*Otsu + (1-lambda)*mean, lambda=0.25 | 1 | 0 | 51.10 | 61.39 | 39.38 | -3.33 | -3.49 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | kém hơn baseline |
| `B6_lambda0.5` | lambda*Otsu + (1-lambda)*mean, lambda=0.5 | 1 | 0 | 53.31 | 63.33 | 37.66 | -1.11 | -1.55 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | kém hơn baseline |
| `B6_lambda0.75` | lambda*Otsu + (1-lambda)*mean, lambda=0.75 | 1 | 0 | 54.48 | 64.32 | 36.44 | +0.06 | -0.55 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | thay đổi nhỏ |
| `B6_lambda1` | lambda*Otsu + (1-lambda)*mean, lambda=1 | 1 | 0 | 54.78 | 64.64 | 35.48 | +0.36 | -0.23 | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) | cải thiện (mIoU và FB-IoU không giảm) |

### 6.4 Nhóm C — Hậu xử lý (Post-processing)

| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|---|
| `C1_off` | no post-processing | 1 | 0 | 41.07 | 46.08 | 44.10 | -1.56 | -1.49 | — | kém hơn baseline |
| `C0_crf_dynamic` | DenseCRF with Algorithm 1 gate (baseline) | 1 | 0 | 42.63 | 47.57 | 43.95 | +0.00 | +0.00 | — | tương đương baseline |
| `C2_crf_always` | DenseCRF always on | 1 | 0 | 42.87 | 48.30 | 42.39 | +0.23 | +0.73 | — | thay đổi nhỏ |
| `C3_guided` | guided filter before thresholding | 1 | 0 | 41.28 | 46.29 | 43.92 | -1.35 | -1.28 | — | kém hơn baseline |
| `C4_morph_open` | morphological opening | 1 | 0 | 41.06 | 46.09 | 44.06 | -1.57 | -1.48 | — | kém hơn baseline |
| `C4_morph_close` | morphological closing | 1 | 0 | 41.09 | 46.09 | 44.14 | -1.54 | -1.48 | — | kém hơn baseline |
| `C4_morph_open_close` | opening then closing | 1 | 0 | 41.08 | 46.09 | 44.10 | -1.55 | -1.48 | — | kém hơn baseline |
| `C4_morph_remove_small` | remove components < 0.5% of image | 1 | 0 | 40.93 | 46.30 | 43.18 | -1.70 | -1.27 | — | kém hơn baseline |
| `C4_morph_fill_holes` | fill holes | 1 | 0 | 41.88 | 46.39 | 45.49 | -0.75 | -1.18 | — | kém hơn baseline |
| `C4_morph_full` | open + close + remove small + fill holes | 1 | 0 | 41.73 | 46.59 | 44.59 | -0.90 | -0.98 | — | kém hơn baseline |
| `C5_guided_morph` | guided filter then threshold then morphology | 1 | 0 | 41.70 | 46.55 | 44.37 | -0.94 | -1.02 | — | kém hơn baseline |
| `C6_guided_dynamic` | guided filter with Algorithm 1 gate | 1 | 0 | 41.43 | 46.38 | 44.15 | -1.20 | -1.19 | — | kém hơn baseline |
| `C6_morph_full_dynamic` | morphology with Algorithm 1 gate | 1 | 0 | 41.40 | 46.29 | 44.38 | -1.23 | -1.28 | — | kém hơn baseline |

*Nhóm C dùng tập con CRF; `C1_off` trong bảng này cũng chạy trên đúng tập con đó nên so sánh là ghép cặp.*


### 6.6 Lưới trọng số A5 (w_L, w_M, w_H)

| Cấu hình | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB |
|---|---|---|---|---|---|
| `A5_w0.5_0.5_0.5` | 54.18 | 64.68 | 33.70 | -0.24 | -0.19 |
| `A5_w0.5_0.5_1` | 52.88 | 63.74 | 33.71 | -1.54 | -1.13 |
| `A5_w0.5_0.5_2` | 50.94 | 62.28 | 33.88 | -3.48 | -2.59 |
| `A5_w0.5_1_0.5` | 54.33 | 64.84 | 33.42 | -0.09 | -0.04 |
| `A5_w0.5_1_1` | 53.73 | 64.41 | 33.38 | -0.69 | -0.46 |
| `A5_w0.5_1_2` | 52.21 | 63.26 | 33.55 | -2.21 | -1.61 |
| `A5_w0.5_2_0.5` | 53.97 | 64.64 | 33.21 | -0.45 | -0.23 |
| `A5_w0.5_2_1` | 53.98 | 64.62 | 33.19 | -0.44 | -0.25 |
| `A5_w0.5_2_2` | 53.33 | 64.13 | 33.24 | -1.10 | -0.75 |
| `A5_w1_0.5_0.5` | 54.15 | 64.52 | 34.16 | -0.27 | -0.36 |
| `A5_w1_0.5_1` | 53.58 | 64.20 | 34.01 | -0.84 | -0.67 |
| `A5_w1_0.5_2` | 51.92 | 62.99 | 33.99 | -2.50 | -1.88 |
| `A5_w1_1_0.5` | 54.42 | 64.80 | 33.80 | -0.00 | -0.07 |
| `A5_w1_1_1` | 54.18 | 64.68 | 33.70 | -0.24 | -0.19 |
| `A5_w1_1_2` | 52.88 | 63.74 | 33.71 | -1.54 | -1.13 |
| `A5_w1_2_0.5` | 54.24 | 64.78 | 33.45 | -0.18 | -0.10 |
| `A5_w1_2_1` | 54.33 | 64.84 | 33.42 | -0.09 | -0.04 |
| `A5_w1_2_2` | 53.73 | 64.41 | 33.38 | -0.69 | -0.46 |
| `A5_w2_0.5_0.5` | 52.74 | 63.28 | 34.88 | -1.68 | -1.60 |
| `A5_w2_0.5_1` | 53.54 | 64.01 | 34.46 | -0.89 | -0.86 |
| `A5_w2_0.5_2` | 52.89 | 63.64 | 34.28 | -1.53 | -1.23 |
| `A5_w2_1_0.5` | 53.68 | 64.09 | 34.38 | -0.74 | -0.78 |
| `A5_w2_1_1` | 54.15 | 64.52 | 34.16 | -0.27 | -0.36 |
| `A5_w2_1_2` | 53.58 | 64.20 | 34.01 | -0.84 | -0.67 |
| `A5_w2_2_0.5` | 54.13 | 64.56 | 33.88 | -0.29 | -0.31 |
| `A5_w2_2_1` | 54.42 | 64.80 | 33.80 | -0.00 | -0.07 |
| `A5_w2_2_2` | 54.18 | 64.68 | 33.70 | -0.24 | -0.19 |

### 6.7 Chi tiết theo tập dữ liệu (các cấu hình đáng chú ý)

| Cấu hình | Deepglobe (m/FB) | ISIC (m/FB) | Chest X-ray (m/FB) | FSS-1000 (m/FB) | SUIM (m/FB) | TB 5 tập (m/FB) |
|---|---|---|---|---|---|---|
| Baseline no-pp (A0+B0+C1) | 41.9 / 47.0 | 45.2 / 59.1 | 79.8 / 86.4 | 70.7 / 80.2 | 34.5 / 51.6 | 54.4 / 64.9 |
| Baseline đầy đủ (A0+B0+C0) † | 42.6 / 47.6 | — | — | — | — | 42.6 / 47.6 |
| Best A: A8_geomean | 42.4 / 47.1 | 45.7 / 58.8 | 80.7 / 86.9 | 70.7 / 80.1 | 33.6 / 50.3 | 54.6 / 64.6 |
| Best B: B2_otsus | 43.7 / 47.8 | 44.2 / 57.2 | 79.8 / 86.4 | 70.7 / 80.3 | 35.4 / 51.5 | 54.8 / 64.6 |
| Best C: C2_crf_always | 42.9 / 48.3 | — | — | — | — | 42.9 / 48.3 |

### 6.8 Nhận xét & hạn chế

- Mọi biến thể đều là heuristic không học; kết quả phụ thuộc vào chất lượng đặc trưng đã thích ứng và vào việc ước lượng đúng tỉ lệ foreground.
- Các biến thể làm tăng mIoU nhưng giảm FB-IoU hoặc tăng mạnh %FG cần được coi là **dự đoán thừa foreground**, không phải cải thiện thực sự.
- Nhóm C (đặc biệt CRF) mới chạy trên tập con; cần chạy lại toàn bộ episode nếu muốn kết luận chắc chắn. Nhóm A/B đã chạy trên toàn bộ episode đã cache.
- Kết quả 1 seed; để kết luận chắc chắn cần chạy thêm seed và báo cáo trung bình ± độ lệch chuẩn (xem mục 7 của README).
- Deepglobe có nhãn nhiễu; chênh lệch nhỏ trên tập này không nên diễn giải quá mức.


### 6.9 Commit của từng nhóm

| Nhóm | Commit |
|---|---|
| Nhóm A — Gộp tầng (Layer Fusion) | [83e61f4b](https://github.com/tiudan147/BTL_XLA/commit/83e61f4bac721b17cf5d1073f5579bf3004684c6) |
| Nhóm B — Cắt ngưỡng (Thresholding) | [43bfd243](https://github.com/tiudan147/BTL_XLA/commit/43bfd24395187ab73d5ecf145898902c96201b92) |
| Code pipeline | [21a47578](https://github.com/tiudan147/BTL_XLA/commit/21a47578f85759d84faa15a046aae4f362a79c47) |

### 6.10 Cách chạy lại

Các biến thể thắng baseline đã được hợp nhất vào code chính. Chạy lại trực tiếp:

```bash
# Baseline (F0 / T-PredMean / P0)
python main.py --benchmark fss --datapath ./datasets/fss/ --nshot 1

# A8_geomean (gộp trung bình hình học 13 tầng)
python main.py --benchmark fss --datapath ./datasets/fss/ --nshot 1 \
  --fusion-method geomean

# B2_otsus (ngưỡng Otsu thuần)
python main.py --benchmark fss --datapath ./datasets/fss/ --nshot 1 \
  --threshold-method otsus

# C2_crf_always (DenseCRF luôn bật; cần cài pydensecrf)
python main.py --benchmark fss --datapath ./datasets/fss/ --nshot 1 \
  --postprocessing always
```

Để sinh lại bảng kết quả từ CSV thô, dùng lại pipeline thí nghiệm đã lưu trong
lịch sử git (commit `5892061`).

