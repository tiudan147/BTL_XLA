## 6. Nhật ký thí nghiệm

> Kết quả dưới đây được sinh tự động bởi `experiments/report.py` từ `experiments/results/*.csv`. Mỗi dòng có link tới commit chứa kết quả tương ứng.

### 6.0 Giao thức chạy

- **Shot:** 1; **seed:** 0; **số episode/tập:** 5 (lấy mẫu phân tầng đều trên toàn bộ test split; đặt `--max-episodes 0` để chạy toàn bộ).
- **Thiết bị sinh cache adapter:** `cuda`; code commit `5892061e29286815eb558896245960d950aac169`.
- **Adapter:** đóng băng ResNet50 + adapter 2 lớp Conv 1×1, SGD 25 epoch, lr 0.01, 2 augmentation shear tối đa 20° (đúng cấu hình gốc). Adapter được fit một lần cho mỗi lớp (first-episode) và **dùng lại** cho mọi biến thể đầu ra.
- **Cache:** 13 bản đồ `q̂_pred_l` ở độ phân giải 100×100 (đúng nơi baseline gộp), cùng ảnh query/support, mặt nạ GT/support và pseudo-episode của Thuật toán 1. Nhờ vậy mọi biến thể được so sánh trên **cùng đặc trưng đã thích ứng**.
- **Chỉ số:** mIoU (trung bình IoU foreground theo lớp), FB-IoU (trung bình IoU background và foreground toàn cục), %FG (tỉ lệ foreground dự đoán). `mIoU TB`/`FB-IoU TB` là trung bình trên 5 tập (Deepglobe, ISIC, Chest X-ray, FSS-1000, SUIM).
- **Nhóm C và các biến thể CRF:** DenseCRF chạy CPU rất chậm (~1.7 s/ảnh cho 5 vòng), nên nhóm C được đánh giá trên một tập con phân tầng (`--crf-max-episodes`, mặc định 200) và **mọi biến thể trong nhóm C dùng cùng tập con** để so sánh có ghép cặp. Nhóm A/B chạy trên toàn bộ episode đã cache.


> **Lưu ý:** đây là kết quả tự chạy trên một tập con phân tầng, không phải con số trong Bảng 4 của bài báo. Dùng cột Δ so với baseline tự chạy, không so trực tiếp với 60.7/69.2 của bài báo.


### 6.1 Baseline tự chạy

| Cấu hình | Deepglobe (m/FB) | ISIC (m/FB) | Chest X-ray (m/FB) | FSS-1000 (m/FB) | SUIM (m/FB) | TB 5 tập (m/FB) |
|---|---|---|---|---|---|---|
| Baseline no-pp (A0+B0+C1) | — | — | 76.1 / 83.8 | — | — | 76.1 / 83.8 |
| Baseline đầy đủ (A0+B0+C0) † | — | — | 69.2 / 79.1 | — | — | 69.2 / 79.1 |

† C0 dùng tập con CRF (xem mục 6.0); A0/A0+B0+C1 dùng toàn bộ episode đã cache.

Baseline no-pp trung bình 5 tập: **76.09 / 83.84**; baseline đầy đủ: **69.17 / 79.11**.


### 6.2 Nhóm A — Gộp tầng (Layer Fusion)

| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|---|
| `A0_mean` | baseline fusion (13-layer mean) | 1 | 0 | 76.09 | 83.84 | 23.47 | +0.00 | +0.00 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A1_low` | only Low (layers 1-4) | 1 | 0 | 61.02 | 71.44 | 33.01 | -15.07 | -12.40 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A2_mid` | only Mid (layers 5-10) | 1 | 0 | 71.64 | 81.12 | 20.62 | -4.45 | -2.72 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A3_high` | only High (layers 11-13) | 1 | 0 | 63.84 | 74.07 | 30.33 | -12.25 | -9.77 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A4_low_mid` | Low + Mid | 1 | 0 | 74.20 | 82.67 | 22.32 | -1.88 | -1.17 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A4_mid_high` | Mid + High | 1 | 0 | 73.77 | 82.31 | 22.86 | -2.32 | -1.53 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A4_low_high` | Low + High | 1 | 0 | 73.24 | 81.33 | 27.76 | -2.85 | -2.51 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A6_adaptive_otsu` | per-episode stage weights from Otsu separability | 1 | 0 | 75.26 | 83.12 | 24.73 | -0.83 | -0.72 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A6_adaptive_contrast` | per-episode stage weights from map contrast | 1 | 0 | 70.55 | 79.90 | 24.12 | -5.54 | -3.94 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A6_adaptive_entropy` | per-episode stage weights from histogram entropy | 1 | 0 | 62.47 | 73.83 | 26.07 | -13.62 | -10.01 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A7_norm_minmax` | per-layer min-max normalisation before mean | 1 | 0 | 77.36 | 84.71 | 23.55 | +1.27 | +0.87 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | cải thiện (mIoU và FB-IoU không giảm) |
| `A7_norm_zscore` | per-layer z-score normalisation before mean | 1 | 0 | 77.35 | 84.71 | 23.49 | +1.26 | +0.87 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | cải thiện (mIoU và FB-IoU không giảm) |
| `A8_max` | layer-wise max | 1 | 0 | 62.70 | 74.37 | 24.00 | -13.38 | -9.48 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A8_geomean` | layer-wise geometric mean | 1 | 0 | 76.99 | 84.42 | 23.96 | +0.90 | +0.57 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | cải thiện (mIoU và FB-IoU không giảm) |
| `A8_median` | layer-wise median | 1 | 0 | 76.59 | 84.21 | 23.26 | +0.50 | +0.37 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | cải thiện (mIoU và FB-IoU không giảm) |
| `A5_w0.5_0.5_0.5` | stage weights L/M/H=0.5/0.5/0.5 | 1 | 0 | 76.15 | 83.79 | 24.35 | +0.06 | -0.05 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A5_w0.5_0.5_1` | stage weights L/M/H=0.5/0.5/1 | 1 | 0 | 74.41 | 82.47 | 25.31 | -1.68 | -1.38 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w0.5_0.5_2` | stage weights L/M/H=0.5/0.5/2 | 1 | 0 | 70.48 | 79.40 | 27.36 | -5.61 | -4.44 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w0.5_1_0.5` | stage weights L/M/H=0.5/1/0.5 | 1 | 0 | 75.65 | 83.56 | 23.28 | -0.44 | -0.28 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w0.5_1_1` | stage weights L/M/H=0.5/1/1 | 1 | 0 | 75.00 | 83.02 | 24.09 | -1.09 | -0.82 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w0.5_1_2` | stage weights L/M/H=0.5/1/2 | 1 | 0 | 72.70 | 81.22 | 25.61 | -3.39 | -2.62 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w0.5_2_0.5` | stage weights L/M/H=0.5/2/0.5 | 1 | 0 | 74.58 | 82.94 | 22.28 | -1.51 | -0.91 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w0.5_2_1` | stage weights L/M/H=0.5/2/1 | 1 | 0 | 74.63 | 82.89 | 22.95 | -1.46 | -0.95 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w0.5_2_2` | stage weights L/M/H=0.5/2/2 | 1 | 0 | 74.16 | 82.45 | 23.98 | -1.93 | -1.40 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w1_0.5_0.5` | stage weights L/M/H=1/0.5/0.5 | 1 | 0 | 75.90 | 83.55 | 24.91 | -0.19 | -0.29 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A5_w1_0.5_1` | stage weights L/M/H=1/0.5/1 | 1 | 0 | 75.73 | 83.38 | 25.44 | -0.36 | -0.47 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w1_0.5_2` | stage weights L/M/H=1/0.5/2 | 1 | 0 | 72.86 | 81.18 | 26.82 | -3.23 | -2.66 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w1_1_0.5` | stage weights L/M/H=1/1/0.5 | 1 | 0 | 76.35 | 84.00 | 23.66 | +0.26 | +0.16 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A5_w1_1_1` | stage weights L/M/H=1/1/1 | 1 | 0 | 76.15 | 83.79 | 24.35 | +0.06 | -0.05 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A5_w1_1_2` | stage weights L/M/H=1/1/2 | 1 | 0 | 74.41 | 82.47 | 25.31 | -1.68 | -1.38 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w1_2_0.5` | stage weights L/M/H=1/2/0.5 | 1 | 0 | 75.44 | 83.49 | 22.62 | -0.64 | -0.35 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w1_2_1` | stage weights L/M/H=1/2/1 | 1 | 0 | 75.65 | 83.56 | 23.28 | -0.44 | -0.28 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w1_2_2` | stage weights L/M/H=1/2/2 | 1 | 0 | 75.00 | 83.02 | 24.09 | -1.09 | -0.82 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w2_0.5_0.5` | stage weights L/M/H=2/0.5/0.5 | 1 | 0 | 74.52 | 82.45 | 26.10 | -1.56 | -1.39 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w2_0.5_1` | stage weights L/M/H=2/0.5/1 | 1 | 0 | 75.19 | 82.93 | 26.00 | -0.90 | -0.91 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w2_0.5_2` | stage weights L/M/H=2/0.5/2 | 1 | 0 | 74.94 | 82.71 | 26.37 | -1.15 | -1.13 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w2_1_0.5` | stage weights L/M/H=2/1/0.5 | 1 | 0 | 75.70 | 83.44 | 24.71 | -0.38 | -0.40 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w2_1_1` | stage weights L/M/H=2/1/1 | 1 | 0 | 75.90 | 83.55 | 24.91 | -0.19 | -0.29 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A5_w2_1_2` | stage weights L/M/H=2/1/2 | 1 | 0 | 75.73 | 83.38 | 25.44 | -0.36 | -0.47 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | kém hơn baseline |
| `A5_w2_2_0.5` | stage weights L/M/H=2/2/0.5 | 1 | 0 | 75.80 | 83.66 | 23.34 | -0.29 | -0.18 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A5_w2_2_1` | stage weights L/M/H=2/2/1 | 1 | 0 | 76.35 | 84.00 | 23.66 | +0.26 | +0.16 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |
| `A5_w2_2_2` | stage weights L/M/H=2/2/2 | 1 | 0 | 76.15 | 83.79 | 24.35 | +0.06 | -0.05 | [19b64913](https://github.com/tiudan147/BTL_XLA/commit/19b64913bc47ddf44468f85030d1bfc10072e201) | tương đương baseline |

### 6.3 Nhóm B — Cắt ngưỡng (Thresholding)

| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|---|
| `B0_pred_mean` | max(mean, Otsu) baseline | 1 | 0 | 76.09 | 83.84 | 23.47 | +0.00 | +0.00 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | tương đương baseline |
| `B1_mean` | mean threshold | 1 | 0 | 68.06 | 76.50 | 34.62 | -8.03 | -7.34 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | kém hơn baseline |
| `B2_otsus` | pure Otsu | 1 | 0 | 76.09 | 83.84 | 23.47 | +0.00 | +0.00 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | tương đương baseline |
| `B3_multi_otsu` | 3-class Otsu, highest class | 1 | 0 | 61.42 | 74.85 | 15.96 | -14.67 | -8.99 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | kém hơn baseline |
| `B4_quantile_support` | quantile at support foreground ratio | 1 | 0 | 63.43 | 74.35 | 27.06 | -12.66 | -9.49 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | kém hơn baseline |
| `B5_adaptive_local` | local Gaussian adaptive threshold | 1 | 0 | 32.74 | 48.89 | 33.59 | -43.35 | -34.95 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | kém hơn baseline |
| `B6_lambda0` | lambda*Otsu + (1-lambda)*mean, lambda=0 | 1 | 0 | 68.06 | 76.50 | 34.62 | -8.03 | -7.34 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | kém hơn baseline |
| `B6_lambda0.25` | lambda*Otsu + (1-lambda)*mean, lambda=0.25 | 1 | 0 | 73.09 | 80.84 | 30.70 | -3.00 | -3.00 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | kém hơn baseline |
| `B6_lambda0.5` | lambda*Otsu + (1-lambda)*mean, lambda=0.5 | 1 | 0 | 75.54 | 82.97 | 27.83 | -0.55 | -0.88 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | kém hơn baseline |
| `B6_lambda0.75` | lambda*Otsu + (1-lambda)*mean, lambda=0.75 | 1 | 0 | 76.55 | 83.94 | 25.48 | +0.46 | +0.10 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | cải thiện (mIoU và FB-IoU không giảm) |
| `B6_lambda1` | lambda*Otsu + (1-lambda)*mean, lambda=1 | 1 | 0 | 76.09 | 83.84 | 23.47 | +0.00 | +0.00 | [e05c495e](https://github.com/tiudan147/BTL_XLA/commit/e05c495e8b67ebf0033b6585fb34a6e08d8c9df6) | tương đương baseline |

### 6.4 Nhóm C — Hậu xử lý (Post-processing)

| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|---|
| `C1_off` | no post-processing | 1 | 0 | 71.79 | 80.74 | 24.17 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C0_crf_dynamic` | DenseCRF with Algorithm 1 gate (baseline) | 1 | 0 | 69.17 | 79.11 | 22.62 | +0.00 | +0.00 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | tương đương baseline |
| `C2_crf_always` | DenseCRF always on | 1 | 0 | 64.06 | 75.17 | 24.76 | -5.10 | -3.94 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | kém hơn baseline |
| `C3_guided` | guided filter before thresholding | 1 | 0 | 71.60 | 80.54 | 24.63 | +2.43 | +1.43 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C4_morph_open` | morphological opening | 1 | 0 | 71.80 | 80.74 | 24.16 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C4_morph_close` | morphological closing | 1 | 0 | 71.79 | 80.74 | 24.17 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C4_morph_open_close` | opening then closing | 1 | 0 | 71.80 | 80.74 | 24.17 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C4_morph_remove_small` | remove components < 0.5% of image | 1 | 0 | 71.80 | 80.74 | 24.17 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C4_morph_fill_holes` | fill holes | 1 | 0 | 71.79 | 80.74 | 24.17 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C4_morph_full` | open + close + remove small + fill holes | 1 | 0 | 71.80 | 80.74 | 24.17 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C5_guided_morph` | guided filter then threshold then morphology | 1 | 0 | 71.59 | 80.54 | 24.60 | +2.43 | +1.43 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C6_guided_dynamic` | guided filter with Algorithm 1 gate | 1 | 0 | 71.60 | 80.54 | 24.63 | +2.43 | +1.43 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |
| `C6_morph_full_dynamic` | morphology with Algorithm 1 gate | 1 | 0 | 71.79 | 80.74 | 24.17 | +2.63 | +1.63 | [e49101bc](https://github.com/tiudan147/BTL_XLA/commit/e49101bc4d4075bd4f7e62c9c822489f38234249) | cải thiện (mIoU và FB-IoU không giảm) |

*Nhóm C dùng tập con CRF; `C1_off` trong bảng này cũng chạy trên đúng tập con đó nên so sánh là ghép cặp.*


### 6.5 Kết hợp A + B + C

| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|---|
| `X_A7_norm_minmax__B6_lambda0.75__C1_off` | combo A7_norm_minmax + B6_lambda0.75 + C1_off | 1 | 0 | 73.29 | 81.50 | 26.48 | -2.80 | -2.34 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_minmax__B6_lambda0.75__C4_morph_open_close` | combo A7_norm_minmax + B6_lambda0.75 + C4_morph_open_close | 1 | 0 | 73.33 | 81.53 | 26.46 | -2.76 | -2.31 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_minmax__B6_lambda0.75__C4_morph_full` | combo A7_norm_minmax + B6_lambda0.75 + C4_morph_full | 1 | 0 | 73.52 | 81.68 | 26.39 | -2.56 | -2.17 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_minmax__B6_lambda0.75__C4_morph_open` | combo A7_norm_minmax + B6_lambda0.75 + C4_morph_open | 1 | 0 | 73.33 | 81.53 | 26.46 | -2.76 | -2.32 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_minmax__B0_pred_mean__C1_off` | combo A7_norm_minmax + B0_pred_mean + C1_off | 1 | 0 | 73.34 | 81.82 | 24.19 | -2.75 | -2.02 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_minmax__B0_pred_mean__C4_morph_open_close` | combo A7_norm_minmax + B0_pred_mean + C4_morph_open_close | 1 | 0 | 73.35 | 81.83 | 24.19 | -2.74 | -2.02 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_minmax__B0_pred_mean__C4_morph_full` | combo A7_norm_minmax + B0_pred_mean + C4_morph_full | 1 | 0 | 73.39 | 81.86 | 24.19 | -2.70 | -1.99 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_minmax__B0_pred_mean__C4_morph_open` | combo A7_norm_minmax + B0_pred_mean + C4_morph_open | 1 | 0 | 73.35 | 81.83 | 24.18 | -2.74 | -2.01 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B6_lambda0.75__C1_off` | combo A7_norm_zscore + B6_lambda0.75 + C1_off | 1 | 0 | 73.40 | 81.59 | 26.39 | -2.69 | -2.26 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B6_lambda0.75__C4_morph_open_close` | combo A7_norm_zscore + B6_lambda0.75 + C4_morph_open_close | 1 | 0 | 73.41 | 81.59 | 26.38 | -2.68 | -2.25 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B6_lambda0.75__C4_morph_full` | combo A7_norm_zscore + B6_lambda0.75 + C4_morph_full | 1 | 0 | 73.74 | 81.85 | 26.21 | -2.35 | -1.99 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B6_lambda0.75__C4_morph_open` | combo A7_norm_zscore + B6_lambda0.75 + C4_morph_open | 1 | 0 | 73.39 | 81.59 | 26.37 | -2.70 | -2.26 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B0_pred_mean__C1_off` | combo A7_norm_zscore + B0_pred_mean + C1_off | 1 | 0 | 73.30 | 81.80 | 24.07 | -2.79 | -2.04 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B0_pred_mean__C4_morph_open_close` | combo A7_norm_zscore + B0_pred_mean + C4_morph_open_close | 1 | 0 | 73.31 | 81.82 | 24.06 | -2.78 | -2.03 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B0_pred_mean__C4_morph_full` | combo A7_norm_zscore + B0_pred_mean + C4_morph_full | 1 | 0 | 73.31 | 81.82 | 24.03 | -2.78 | -2.02 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |
| `X_A7_norm_zscore__B0_pred_mean__C4_morph_open` | combo A7_norm_zscore + B0_pred_mean + C4_morph_open | 1 | 0 | 73.32 | 81.82 | 24.05 | -2.77 | -2.02 | [ce724af9](https://github.com/tiudan147/BTL_XLA/commit/ce724af9bf34c19a2e304f824e2171f767dc6f5f) | kém hơn baseline |

### 6.6 Lưới trọng số A5 (w_L, w_M, w_H)

| Cấu hình | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB |
|---|---|---|---|---|---|
| `A5_w0.5_0.5_0.5` | 76.15 | 83.79 | 24.35 | +0.06 | -0.05 |
| `A5_w0.5_0.5_1` | 74.41 | 82.47 | 25.31 | -1.68 | -1.38 |
| `A5_w0.5_0.5_2` | 70.48 | 79.40 | 27.36 | -5.61 | -4.44 |
| `A5_w0.5_1_0.5` | 75.65 | 83.56 | 23.28 | -0.44 | -0.28 |
| `A5_w0.5_1_1` | 75.00 | 83.02 | 24.09 | -1.09 | -0.82 |
| `A5_w0.5_1_2` | 72.70 | 81.22 | 25.61 | -3.39 | -2.62 |
| `A5_w0.5_2_0.5` | 74.58 | 82.94 | 22.28 | -1.51 | -0.91 |
| `A5_w0.5_2_1` | 74.63 | 82.89 | 22.95 | -1.46 | -0.95 |
| `A5_w0.5_2_2` | 74.16 | 82.45 | 23.98 | -1.93 | -1.40 |
| `A5_w1_0.5_0.5` | 75.90 | 83.55 | 24.91 | -0.19 | -0.29 |
| `A5_w1_0.5_1` | 75.73 | 83.38 | 25.44 | -0.36 | -0.47 |
| `A5_w1_0.5_2` | 72.86 | 81.18 | 26.82 | -3.23 | -2.66 |
| `A5_w1_1_0.5` | 76.35 | 84.00 | 23.66 | +0.26 | +0.16 |
| `A5_w1_1_1` | 76.15 | 83.79 | 24.35 | +0.06 | -0.05 |
| `A5_w1_1_2` | 74.41 | 82.47 | 25.31 | -1.68 | -1.38 |
| `A5_w1_2_0.5` | 75.44 | 83.49 | 22.62 | -0.64 | -0.35 |
| `A5_w1_2_1` | 75.65 | 83.56 | 23.28 | -0.44 | -0.28 |
| `A5_w1_2_2` | 75.00 | 83.02 | 24.09 | -1.09 | -0.82 |
| `A5_w2_0.5_0.5` | 74.52 | 82.45 | 26.10 | -1.56 | -1.39 |
| `A5_w2_0.5_1` | 75.19 | 82.93 | 26.00 | -0.90 | -0.91 |
| `A5_w2_0.5_2` | 74.94 | 82.71 | 26.37 | -1.15 | -1.13 |
| `A5_w2_1_0.5` | 75.70 | 83.44 | 24.71 | -0.38 | -0.40 |
| `A5_w2_1_1` | 75.90 | 83.55 | 24.91 | -0.19 | -0.29 |
| `A5_w2_1_2` | 75.73 | 83.38 | 25.44 | -0.36 | -0.47 |
| `A5_w2_2_0.5` | 75.80 | 83.66 | 23.34 | -0.29 | -0.18 |
| `A5_w2_2_1` | 76.35 | 84.00 | 23.66 | +0.26 | +0.16 |
| `A5_w2_2_2` | 76.15 | 83.79 | 24.35 | +0.06 | -0.05 |

### 6.7 Chi tiết theo tập dữ liệu (các cấu hình đáng chú ý)

| Cấu hình | Deepglobe (m/FB) | ISIC (m/FB) | Chest X-ray (m/FB) | FSS-1000 (m/FB) | SUIM (m/FB) | TB 5 tập (m/FB) |
|---|---|---|---|---|---|---|
| Baseline no-pp (A0+B0+C1) | — | — | 76.1 / 83.8 | — | — | 76.1 / 83.8 |
| Baseline đầy đủ (A0+B0+C0) † | — | — | 69.2 / 79.1 | — | — | 69.2 / 79.1 |
| Best A: A7_norm_minmax | — | — | 77.4 / 84.7 | — | — | 77.4 / 84.7 |
| Best B: B6_lambda0.75 | — | — | 76.6 / 83.9 | — | — | 76.6 / 83.9 |
| Best C: C4_morph_open_close | — | — | 71.8 / 80.7 | — | — | 71.8 / 80.7 |
| Best X: X_A7_norm_zscore__B6_lambda0.75__C4_morph_full | — | — | 73.7 / 81.9 | — | — | 73.7 / 81.9 |

### 6.8 Nhận xét & hạn chế

- Mọi biến thể đều là heuristic không học; kết quả phụ thuộc vào chất lượng đặc trưng đã thích ứng và vào việc ước lượng đúng tỉ lệ foreground.
- Các biến thể làm tăng mIoU nhưng giảm FB-IoU hoặc tăng mạnh %FG cần được coi là **dự đoán thừa foreground**, không phải cải thiện thực sự.
- Nhóm C (đặc biệt CRF) mới chạy trên tập con; cần chạy lại toàn bộ episode nếu muốn kết luận chắc chắn. Nhóm A/B đã chạy trên toàn bộ episode đã cache.
- Kết quả 1 seed; để kết luận chắc chắn cần chạy thêm seed và báo cáo trung bình ± độ lệch chuẩn (xem mục 7 của README).
- Deepglobe có nhãn nhiễu; chênh lệch nhỏ trên tập này không nên diễn giải quá mức.


### 6.9 Cách chạy lại

```bash
# 1. Cache adapter + 13 layer predictions (một lần cho mỗi tập/shot/seed)
python -m experiments.cache_features --benchmark lung --datapath /path/to/lungsegmentation \
    --shot 1 --seed 0 --max-episodes 600

# 2. Chạy từng nhóm offline trên cache
python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group A
python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group B
python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group C
python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group X

# 3. Sinh lại tài liệu
python -m experiments.report --shot 1 --seed 0 --out docs/exp.md
```

