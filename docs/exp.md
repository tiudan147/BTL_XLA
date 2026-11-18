## 6. Nhật ký thí nghiệm

Quy ước đặt tên: `expXX_<nhóm><số>_<mô_tả>` (ví dụ `exp03_A2_mid_only`). Mỗi thí nghiệm một commit Git. Ghi **cả thí nghiệm thất bại**.

| Mã TN | Nhóm | Thay đổi so với baseline | Shot | Seed | mIoU TB | FB-IoU TB | % FG | Commit | Nhận xét |
|---|---|---|---|---|---|---|---|---|---|
| exp00_baseline | — | Không đổi (A0+B0+C0) | 1 | | | | | | Mốc so sánh |
| exp01_C1_nopp | C | Tắt hậu xử lý | 1 | | | | | | |
| exp02_A1_low_only | A | Chỉ tầng L | 1 | | | | | | |
| exp03_A2_mid_only | A | Chỉ tầng M | 1 | | | | | | |
| exp04_A3_high_only | A | Chỉ tầng H | 1 | | | | | | |
| exp05_B1_mean | B | Ngưỡng mean | 1 | | | | | | |
| exp06_B2_otsu | B | Ngưỡng Otsu | 1 | | | | | | |
| exp07_C3_guided | C | Guided Filter thay CRF | 1 | | | | | | |
| exp08_C4_morph | C | Hình thái học | 1 | | | | | | |
| … | | | | | | | | | |

**Bảng chi tiết theo tập dữ liệu** (làm riêng cho các cấu hình đáng chú ý):

| Cấu hình | Deepglobe (m/FB) | ISIC | Chest X-ray | FSS-1000 | SUIM | Avg |
|---|---|---|---|---|---|---|
| Baseline | | | | | | |
| _(điền)_ | | | | | | |
