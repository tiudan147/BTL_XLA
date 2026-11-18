# Cải tiến khâu đầu ra của ABCDFSS: Gộp đa tầng, Cắt ngưỡng & Hậu xử lý

> **Bài báo gốc:** *Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation* (Herzog, arXiv:2402.17614v2)
> **Mã nguồn gốc:** https://github.com/Vision-Kek/ABCDFSS
> **Phạm vi tài liệu:** Mục 3.3, 3.4 và Phụ lục D của bài báo
> **Người phụ trách:** _(điền tên)_  **Cập nhật lần cuối:** _(điền ngày)_

---

## 1. Giới thiệu & mục tiêu

### 1.1 Bối cảnh
ABCDFSS giải bài toán phân đoạn few-shot đa miền (CD-FSS) mà **không huấn luyện mạng phân đoạn nào**. Backbone ResNet50 (pretrain ImageNet) được giữ đóng băng, mỗi tầng trung gian gắn một adapter nhỏ học ngay lúc test. Sau đó:

1. Tính bản đồ tương quan chéo giữa query và support cho từng tầng (Eq. 6).
2. Phóng đại và lấy **trung bình đều** các bản đồ của 13 tầng (Eq. 7).
3. Cắt ngưỡng bằng `max(mean, Otsu)` (Eq. 8).
4. Hậu xử lý bằng DenseCRF, chỉ bật khi cải thiện IoU trên một pseudo-episode (Thuật toán 1).

Thành viên phụ trách **khâu 2–4**: từ khi có đặc trưng đã thích ứng cho đến mặt nạ nhị phân cuối cùng `M̂^q`. Phần adapter (hàm mất mát, huấn luyện test-time) do thành viên khác phụ trách và được xem là **cố định** trong tài liệu này.

### 1.2 Mục tiêu cải tiến
Các khâu này đều là **heuristic không học**, nên có thể thử nghiệm rẻ, không cần huấn luyện lại. Mục tiêu:

- Tăng **mIoU và FB-IoU trung bình trên CD-FSS benchmark** so với baseline (1-shot: 60.7 / 69.2; 5-shot: 65.0 / 73.1).
- Không để model "ăn gian" mIoU bằng cách dự đoán thừa foreground (xem mục 3.2). Mọi kết luận phải được kiểm chứng bằng cả FB-IoU và tỉ lệ foreground dự đoán.
- Giữ nguyên tinh thần của bài báo: không dùng thêm dữ liệu ngoài support/query, không huấn luyện mạng phân đoạn.

### 1.3 Câu hỏi nghiên cứu
| # | Câu hỏi |
|---|---------|
| Q1 | Gộp có chọn lọc/có trọng số theo nhóm tầng (Low/Mid/High) có tốt hơn trung bình đều 13 tầng không? |
| Q2 | Ngưỡng `max(mean, Otsu)` có tối ưu không, hay Otsu thuần / mean thuần / ngưỡng khác tốt hơn trên từng miền? |
| Q3 | Có thể thay DenseCRF bằng Guided Filter hoặc phép toán hình thái học mà không mất (hoặc tăng) hiệu năng không? |

---

## 2. Nguyên lý cần nắm (tóm tắt)

### 2.1 Tương quan chéo (Cross-Correlation) — Mục 3.3, Phụ lục D
Với mỗi tầng `l`, đặc trưng đã thích ứng của query `F̂ᵠ_l` và support `F̂ˢ_l` được làm phẳng chiều không gian:

```
q̂_pred_l = softmax( Q Kᵀ / √d ) · V
   Q = F̂ᵠ_l  (HW × C)       K = F̂ˢ_l  (HW·k × C)       V = Mˢ_l  (HW·k × 1)
```

Ba bước: (1) nhân ma trận tạo affinity map HW×HW giữa mọi cặp pixel query–support; (2) softmax theo chiều support, làm nổi bật các pixel support giống nhất; (3) nhân với mặt nạ support để chỉ giữ lại độ giống với **foreground** support.
Kết quả là bản đồ điểm số nằm trong khoảng con của [0, 1], **phụ thuộc mẫu** và **không phải xác suất**. Với đặc trưng ngẫu nhiên, kỳ vọng của điểm số xấp xỉ `mean(Mˢ)` (tỉ lệ foreground của support).

### 2.2 Gộp 13 tầng — Mục 3.4, Eq. 7
```
q̂_fused = (1/L) · Σ_{l=1..L} upsample(q̂_pred_l),   L = 13
```
- 13 tầng ResNet50 chia thành **Low / Mid / High = 4 / 6 / 3** (theo HSNet và DCAMA).
- Upsample song tuyến tính về kích thước ảnh query. Ở tầng sâu nhất, hệ số phóng đại tới ×32, nên mặt nạ thô.
- Bài báo lập luận phép trung bình đơn giản này vừa **tự điều chuẩn**, vừa **khử nhiễu**, vì bản đồ có độ tự tin cao tự nhiên đóng góp điểm lớn hơn vào tổng.
- Quan sát định tính (Hình 10): tầng **Low** hữu ích cho Deepglobe và ISIC, tầng **High** hợp với FSS-1000 và SUIM, tầng **Mid** hữu ích ở mọi tập. Đây là cơ sở cho các thí nghiệm gộp có chọn lọc.

### 2.3 Cắt ngưỡng — Eq. 8, Phụ lục D
```
M̂^q = q̂_fused > thresh(q̂_fused)
thresh(m) = max( mean(m), otsu(m) )
```
- Otsu (cực tiểu hóa phương sai nội lớp trên histogram) tách cụm foreground khỏi background.
- Nếu Otsu cho ngưỡng thấp hơn `mean`, lấy `mean` để tránh dự đoán thừa foreground (Hình 9c: cụm bên phải quá gần điểm trung bình, thực chất là vùng "không rõ", không phải foreground).
- Bài báo nhấn mạnh ước lượng đúng tỉ lệ foreground là yếu tố chính quyết định hiệu năng FSS.

### 2.4 Hậu xử lý — Mục 3.4, Thuật toán 1
- Mặt nạ thô được tinh chỉnh bằng **DenseCRF** (Krähenbühl & Koltun) dùng ảnh RGB query, với unary là `sigmoid(T·(m̂ − τ))`, `T = 1`.
- Tham số: Gaussian σ = 1 (compat 2); bilateral σ_xy = 35, σ_rgb = 13 (compat 1).
- **Quyết định bật/tắt động:** dựng pseudo-episode (support làm pseudo-query, bản augment của support làm pseudo-support), chỉ áp CRF nếu IoU với mặt nạ support thật **tăng**. Vì vậy với Chest X-ray, CRF hầu như không được bật.

---

## 3. Baseline

### 3.1 Kết quả của bài báo (Bảng 4)
`m` = mIoU, `FB` = FB-IoU.

**1-shot**

| Cấu hình | Deepglobe (m / FB) | ISIC | Chest X-ray | FSS-1000 | **CD-FSS Avg** | SUIM |
|---|---|---|---|---|---|---|
| Ours (no-pp, chưa hậu xử lý) | 42.3 / 47.1 | 41.8 / 57.2 | 80.0 / 86.2 | 69.3 / 79.3 | 58.3 / 67.5 | 35.0 / 54.2 |
| Ours (đầy đủ) | 42.6 / 47.7 | 45.7 / 60.3 | 79.8 / 86.1 | 74.6 / 82.7 | **60.7 / 69.2** | 35.1 / 53.5 |

**5-shot**

| Cấu hình | Deepglobe | ISIC | Chest X-ray | FSS-1000 | **CD-FSS Avg** | SUIM |
|---|---|---|---|---|---|---|
| Ours (no-pp) | 48.2 / 53.4 | 50.8 / 63.9 | 81.6 / 87.4 | 73.1 / 82.3 | 63.4 / 71.8 | 41.1 / 58.3 |
| Ours (đầy đủ) | 49.0 / 54.6 | 53.3 / 66.1 | 81.4 / 87.3 | 76.2 / 84.2 | **65.0 / 73.1** | 41.3 / 58.2 |

> Khi tự chạy lại baseline, kết quả có thể lệch nhẹ so với bài báo (adapter khởi tạo ngẫu nhiên, SGD, augmentation ngẫu nhiên). **Dùng số liệu tự chạy làm baseline chính thức để so sánh**, bảng trên chỉ để đối chiếu.

### 3.2 Vì sao phải báo cáo cả FB-IoU và % foreground
Bài báo (Phụ lục A) chỉ ra mIoU trong FSS 1-way **tăng khi dự đoán nhiều foreground hơn**, còn FB-IoU phạt việc dự đoán thừa. Ví dụ Deepglobe 1-shot: predictor ngây thơ (toàn foreground) đạt mIoU 43.0 nhưng FB-IoU chỉ 21.5, trong khi ABCDFSS đạt 42.6 / 47.7 với 48.6% foreground (thực tế là 43.5%).
**Hệ quả cho thí nghiệm:** một cấu hình chỉ được coi là cải thiện nếu **FB-IoU không giảm** và tỉ lệ foreground dự đoán không lệch xa tỉ lệ thực. Cải thiện mIoU đơn thuần có thể chỉ là hiệu ứng dự đoán thừa.

### 3.3 Ghi chú về dữ liệu đánh giá
- Benchmark gồm Deepglobe, ISIC, Chest X-ray, FSS-1000 (+ SUIM). **Không dùng PASCAL/COCO**, vì không có giai đoạn huấn luyện nguồn.
- Bài báo chỉ ra Deepglobe có nhãn không chính xác do cắt ảnh nhỏ, nên **chênh lệch nhỏ trên Deepglobe không nên được diễn giải quá mức**.
- FSS-1000 thuộc "cross-dataset" hơn là "cross-domain"; SUIM (dưới nước) và các tập y tế phản ánh CD-FSS thực sự hơn.

---

## 4. Dữ liệu & thiết lập đánh giá

| Mục | Nội dung |
|---|---|
| Tập đánh giá | Deepglobe, ISIC, Chest X-ray, FSS-1000 (CD-FSS benchmark), SUIM |
| Thiết lập | 1-shot và 5-shot, 1-way; mỗi episode 1 query |
| Backbone | ResNet50 ImageNet, đóng băng, 13 tầng |
| Adapter | 2 lớp Conv 1×1 (+BN, ReLU), 64 kênh ra, SGD 25 epoch, lr 0.01, 2 augmentation shear tối đa 20° |
| Chỉ số | mIoU (tích lũy I, U theo lớp rồi lấy trung bình), FB-IoU, **% foreground dự đoán** |
| Tính lặp lại | Cố định seed; **chạy ≥ 3 seed** và báo cáo trung bình ± độ lệch chuẩn |

**Lưu ý thiết kế quan trọng:** adapter mất ≈ 3.7 s/episode (1-shot) nhưng giai đoạn đầu ra chỉ cần các bản đồ `q̂_pred_l` đã tính xong. Vì mọi thí nghiệm trong tài liệu này nằm **sau** bước adapter, nên nên:

1. Chạy adapter **một lần** cho mỗi (tập, seed, shot).
2. **Lưu lại** 13 bản đồ `q̂_pred_l` (trước upsample, dạng `.npy`/`.pt`), ảnh query RGB, mặt nạ GT và mặt nạ support.
3. Chạy tất cả biến thể gộp/ngưỡng/hậu xử lý **offline** trên dữ liệu đã lưu.

Cách này vừa nhanh hơn rất nhiều, vừa đảm bảo mọi biến thể được so sánh trên **cùng** đặc trưng đã thích ứng, loại bỏ nhiễu do khởi tạo ngẫu nhiên.

---

## 5. Giả thuyết & kế hoạch thí nghiệm

### 5.1 Nhóm A — Gộp tầng (Layer Fusion)

Ký hiệu nhóm: **L** = tầng 1–4, **M** = tầng 5–10, **H** = tầng 11–13.

| Mã | Biến thể | Giả thuyết |
|---|---|---|
| A0 | Trung bình đều 13 tầng *(baseline)* | — |
| A1 | Chỉ L | Tốt hơn trên Deepglobe/ISIC, kém trên FSS/SUIM |
| A2 | Chỉ M | Ổn định nhất khi dùng một nhóm riêng lẻ (Hình 10) |
| A3 | Chỉ H | Tốt trên FSS-1000/SUIM, nhưng mặt nạ rất thô (upsample ×32) |
| A4 | L+M, M+H, L+H | Kiểm tra nhóm nào đóng góp nhiều nhất (kiểu ablation) |
| A5 | Trọng số theo nhóm `w_L, w_M, w_H` (lưới, ví dụ w ∈ {0.5, 1, 2}) | Đặt trọng số cao cho nhóm phù hợp miền có thể cải thiện, nhưng trọng số cố định khó tổng quát giữa các miền |
| A6 | Trọng số **thích ứng theo từng episode**, ví dụ theo độ tách của histogram (Otsu between-class variance), độ tương phản hoặc entropy của từng bản đồ | Tự chọn tầng tốt cho từng miền mà không cần nhãn query; là hướng có tiềm năng nhất vì miền đích không biết trước |
| A7 | Chuẩn hóa từng bản đồ (min–max hoặc z-score) trước khi gộp | Cân bằng thang điểm giữa các tầng, nhưng có thể **làm mất** cơ chế "tầng tự tin đóng góp nhiều hơn" mà bài báo nêu |
| A8 | Phép gộp khác: max, trung bình hình học, trung vị | Thử độ bền đối với tầng nhiễu |

**Quy tắc đạo đức thí nghiệm:** trọng số ở A5 **không được** chọn bằng cách dò trên chính tập query của benchmark rồi báo cáo kết quả trên tập đó (rò rỉ nhãn). Hoặc (i) chọn trọng số bằng pseudo-episode dựng từ support như Thuật toán 1, hoặc (ii) nếu dò lưới thì phải nói rõ là "oracle", chỉ dùng để tham chiếu cận trên.

### 5.2 Nhóm B — Cắt ngưỡng

| Mã | Biến thể | Giả thuyết |
|---|---|---|
| B0 | `max(mean, Otsu)` *(baseline)* | — |
| B1 | Chỉ mean | Dự đoán thừa foreground → mIoU có thể giữ/tăng nhẹ nhưng FB-IoU giảm (Hình 9) |
| B2 | Chỉ Otsu | Tốt khi hai cụm tách rõ; dễ dự đoán thừa ở trường hợp kiểu Hình 9c |
| B3 | Otsu đa mức (multi-Otsu, 3 lớp) rồi lấy lớp cao nhất | Xử lý được lớp "không rõ" ở giữa |
| B4 | Ngưỡng theo phân vị, tỉ lệ foreground ước lượng từ `mean(Mˢ)` của support | Dùng tiên nghiệm tỉ lệ foreground, nhưng tỉ lệ foreground query có thể khác support |
| B5 | Ngưỡng cục bộ/thích ứng (adaptive threshold) | Có thể tốt cho ảnh có độ sáng/độ tương phản không đều, nhưng dễ nhiễu |
| B6 | `otsu_mean`: `λ·otsu + (1−λ)·mean`, λ ∈ [0,1] (`--threshold-otsu-weight`, mặc định 0.5) | Đã triển khai; nội suy giữa hai thái cực |

### 5.3 Nhóm C — Hậu xử lý

| Mã | Biến thể | Giả thuyết |
|---|---|---|
| C0 | DenseCRF + quyết định bật/tắt động *(baseline)* | — |
| C1 | Tắt hậu xử lý (no-pp) | Mốc dưới, đã có trong bài báo |
| C2 | DenseCRF luôn bật | Giúp FSS-1000/ISIC, hại Chest X-ray (bài báo nói CRF hầu như không có lợi ở X-ray) |
| C3 | Guided Filter (ảnh RGB query làm guide) làm mượt `q̂_fused` **trước** khi cắt ngưỡng | Làm biên bám cạnh ảnh, nhanh hơn CRF; có thể thay thế CRF |
| C4 | Phép toán hình thái: opening (khử nhiễu hạt), closing (lấp lỗ), loại bỏ thành phần liên thông nhỏ, lấp lỗ (hole filling) | Khử vùng dương tính giả lẻ tẻ (như Hình 11, cột w/o TA), nhưng có thể xóa mất đối tượng nhỏ (như cá nhỏ ở Hình 9c) |
| C5 | Guided Filter → ngưỡng → hình thái (`guided-morph-open`, `guided-morph-close`, `guided-morph-open-close`) | Đã triển khai; kết hợp lọc theo ảnh với tinh chỉnh mask |
| C6 | Cơ chế bật/tắt động (Thuật toán 1) áp dụng cho **mỗi** phương pháp C3–C5 | Quyết định bằng pseudo-episode có thể giúp mỗi phương pháp chỉ hoạt động khi có lợi |

**Triển khai gợi ý:** Guided Filter có sẵn trong `cv2.ximgproc.guidedFilter` (cần `opencv-contrib-python`); hình thái học dùng `cv2.morphologyEx`/`scipy.ndimage`/`skimage.morphology`. Kích thước kernel và bán kính cần được xem là siêu tham số phụ thuộc độ phân giải ảnh.

### 5.4 Thứ tự ưu tiên
1. **B** (rẻ nhất, tác động trực tiếp lên tỉ lệ foreground) → 2. **A5–A6** (tiềm năng cao nhất) → 3. **C3–C5** (thay CRF) → 4. các biến thể còn lại.
Sau khi có kết quả từng nhóm riêng lẻ, mới thử **kết hợp** tốt nhất của A, B, C.

---

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

---

## 7. Phương pháp phân tích kết quả

- **So sánh có ghép cặp:** mọi biến thể chạy trên cùng episode và cùng đặc trưng đã lưu, nên báo cáo hiệu số (Δ) so với baseline thay vì chỉ giá trị tuyệt đối.
- **Độ bền:** báo cáo trung bình ± độ lệch chuẩn qua ≥ 3 seed; chênh lệch nhỏ hơn độ lệch chuẩn không kết luận là cải thiện.
- **Phân tích theo miền:** đặc biệt lưu ý hiệu ứng trái chiều giữa các miền (ví dụ nhóm L tốt cho Deepglobe/ISIC nhưng kém cho FSS/SUIM). Một cấu hình "tốt trung bình" nhưng hại một miền cần được ghi rõ.
- **Ablation:** với cấu hình cuối cùng, tách riêng đóng góp của A, B, C.
- **Phân tích lỗi định tính:** chọn vài episode (thành công, thất bại, trường hợp tương tự Hình 9c) và vẽ lại `q̂_pred_l`, `q̂_fused`, mặt nạ trước/sau từng bước.
- **Chi phí tính toán:** đo thời gian bước đầu ra (hậu xử lý CPU như CRF/Guided Filter có thể chiếm phần đáng kể so với 27/18 fps của chế độ quick-infer ở bài báo), vì giảm thời gian cũng là một lợi ích thực tế.

---

## 8. Hạn chế & hướng phát triển

- Tất cả biến thể đều là **heuristic**, không học scene-level context. Bài báo cũng thừa nhận hạn chế này và gợi ý huấn luyện lại trên tác vụ phân đoạn nguồn có thể cải thiện thêm.
- Các siêu tham số hậu xử lý (kernel, bán kính, ngưỡng) dễ **quá khớp benchmark**; cần chọn theo pseudo-episode hoặc theo quy tắc cố định thống nhất giữa các miền.
- Hiệu năng Deepglobe bị giới hạn bởi nhãn nhiễu, nên khó rút kết luận chắc chắn từ tập này.
- Hướng mở rộng (nếu còn thời gian): gộp có trọng số học nhẹ từ support (ví dụ meta-tham số rất ít), tinh chỉnh ngưỡng bằng ước lượng tỉ lệ foreground, hoặc kết hợp CRF với Guided Filter.

---

## 9. Tài liệu tham khảo chính

- Herzog, J. *Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation.* arXiv:2402.17614v2, 2024.
- Shi et al. *DCAMA: Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation.* ECCV 2022. (cơ chế tương quan chéo)
- Min, Kang, Cho. *Hypercorrelation Squeeze for Few-Shot Segmentation (HSNet).* ICCV 2021. (chia nhóm tầng L/M/H)
- Otsu, N. *A threshold selection method from gray-level histograms.* IEEE TSMC, 1979.
- Krähenbühl, Koltun. *Efficient inference in fully connected CRFs with Gaussian edge potentials.* NeurIPS 2011.
- Lei et al. *Cross-domain few-shot semantic segmentation (PATNet).* ECCV 2022. (CD-FSS benchmark)
