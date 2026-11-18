# Người 3: Cải tiến khâu đầu ra

Phạm vi: từ đặc trưng query/support đã thích ứng đến mặt nạ phân đoạn nhị phân cuối cùng $\hat{M}^q$, tương ứng mục 3.3, 3.4 và Phụ lục D. Mục tiêu là đo riêng ảnh hưởng của gộp đa tầng, cắt ngưỡng và hậu xử lý, sau đó chọn cấu hình có kết quả tốt hơn mà vẫn kiểm soát được chi phí suy luận.

## 1. Pipeline và nguyên lý

1. **Tương quan chéo (cross-correlation):** chuẩn hóa đặc trưng query và support theo dạng vector tại từng pixel, tính tích vô hướng giữa từng cặp pixel để tạo ma trận affinity. Softmax theo các pixel support biến điểm tương quan thành trọng số; nhân ma trận này với support mask để ước lượng xác suất foreground cho mỗi pixel query.
2. **Dự đoán đa tầng:** ResNet50 trích xuất 16 feature map từ các bottleneck. Pipeline hiện bỏ qua 3 map đầu (`l0=3`), tính dự đoán trên 13 map còn lại, nội suy từng dự đoán về cùng kích thước rồi lấy trung bình đều. Kết quả được đưa về kích thước mask support trước khi cắt ngưỡng.
3. **Cắt ngưỡng:** ngưỡng Otsu chọn ngưỡng từ phân bố điểm dự đoán. Cấu hình hiện tại `pred_mean` không phải Mean thuần: nó lấy `max(Otsu, mean(prediction))`. Thí nghiệm cần phân biệt rõ Otsu, Mean thuần và `pred_mean` hiện hành.
4. **Hậu xử lý:** mặc định hiện tại là `off`; `always` áp dụng DenseCRF cho mọi dự đoán, còn `dynamic` chỉ áp dụng khi kiểm tra pseudo-prediction trên support cho thấy có lợi. Hậu xử lý nhận ảnh query và bản đồ xác suất trước cắt ngưỡng.

## 2. Kế hoạch thực nghiệm

### 2.1. Cải tiến gộp tầng (layer fusion)

Sau khi tính affinity và nội suy, hệ thống gộp 13 bản đồ dự đoán của ResNet50. Các map được chia thành ba nhóm: **Low** (4 map), **Mid** (6 map) và **High** (3 map). Baseline `mean` lấy trung bình đều; các chế độ mới cho phép chọn nhóm, đặt trọng số theo stage hoặc giảm ảnh hưởng của map không đồng thuận.

- `low`, `mid`, `high`: chỉ dùng trung bình của nhóm tương ứng; `stage_uniform`, `low_biased`, `high_biased`, `high_suppressed`: gộp trung bình theo stage với trọng số khác nhau.
- `weighted_stage` và `weighted_stage_geomean`: gộp stage theo trọng số tùy chỉnh `--stage-weights WL WM WH`.
- `geomean`: trung bình hình học 13 map; `trimmed_mean`: loại các giá trị thấp/cao nhất theo từng pixel để giảm ảnh hưởng ngoại lệ.
- `agreement`: tự giảm trọng số stage bất đồng với đồng thuận giữa các stage. `pseudo_gate` chấm các phương pháp trên pseudo-episode từ support rồi chọn phương pháp có IoU cao nhất cho episode, không dùng nhãn query.

Ví dụ: `python main.py --benchmark fss --datapath ./datasets/fss/ --fusion-method pseudo_gate`. Có thể chỉ định ứng viên bằng `--fusion-candidates`; mặc định là `mean,high_suppressed,geomean,trimmed_mean,agreement`. Phép gộp không thay đổi backbone hay bước tính affinity.

### 2.2. Cắt ngưỡng và hậu xử lý

Ngưỡng được tính riêng cho mỗi bản đồ dự đoán; pixel được gán foreground khi `prediction > threshold`:

- `mean`: trung bình số học của bản đồ xác suất.
- `otsus`: Otsu trên bản đồ được chuẩn hóa về `[0, 255]`; các pixel dưới 5% dải cường độ bị bỏ khỏi bước ước lượng ngưỡng.
- `pred_mean` (mặc định): `max(Otsu, mean)`, giúp so sánh trực tiếp với hai thành phần riêng lẻ.
- `otsu_mean` (B6): `λ × Otsu + (1 − λ) × mean`; cấu hình `λ` bằng `--threshold-otsu-weight` (mặc định `0.5`, miền `[0, 1]`). `λ=1` tương đương Otsu, `λ=0` tương đương Mean.

Đánh giá các lựa chọn trên cùng cấu hình fusion và episode; giữ `pred_mean` làm baseline, không chọn `λ` dựa trên nhãn query.

Với mỗi ngưỡng, so sánh các nhánh hậu xử lý sau trên cùng dự đoán xác suất và ảnh query:

- **P0 — Tắt:** không tinh chỉnh sau khi cắt ngưỡng.
- **P1 — DenseCRF:** chạy luôn (`always`); chạy `dynamic` như một chế độ riêng để đánh giá cơ chế quyết định có tinh chỉnh hay không. Giữ nguyên cấu hình CRF hiện có.
- **P2 — Guided Filter:** lọc bản đồ xác suất bằng ảnh query RGB đã chuẩn hóa, bán kính 8 pixel và `eps=1e-3`, sau đó tính lại ngưỡng theo phương pháp đã chọn.
- **P3 — Morphology:** thực hiện opening 3×3, closing 3×3 hoặc opening→closing trên mask nhị phân. Có thể ghép Guided Filter → cắt ngưỡng → morphology bằng các chế độ `guided-morph-*`.

Chạy ablation từng yếu tố trước (fusion, ngưỡng, hậu xử lý), rồi mới chạy các tổ hợp của cấu hình fusion tốt nhất với ba ngưỡng và các nhánh hậu xử lý. Không chọn cấu hình bằng cách nhìn vào ground-truth của tập đang báo cáo.

## 3. Quy trình chạy và ghi nhận

- Dùng cùng benchmark, episode/query-support, số shot, seed, trọng số backbone và thiết lập thích ứng cho mọi cấu hình trong một phép so sánh.
- Lưu mã cấu hình, tham số, mIoU, FB-IoU, thời gian suy luận trung bình mỗi episode và tỷ lệ foreground dự đoán. Ghi rõ lỗi hoặc trường hợp không chạy được do phụ thuộc hậu xử lý.
- Báo cáo từng benchmark sẵn có (`fss`, `deepglobe`, `lung`, `isic`, `suim`) và kết quả tổng hợp; nếu benchmark không thể chạy do thiếu dữ liệu hoặc phụ thuộc, ghi rõ thay vì bỏ qua không giải thích.
- Chọn cấu hình trên validation nếu có. Nếu không có validation tách biệt, giữ F0/T-PredMean/P0 làm cấu hình mặc định, trình bày đầy đủ kết quả ablation và không tinh chỉnh tham số dựa trên nhãn test.
- Đưa bảng kết quả cạnh baseline để kết luận cấu hình thắng về chất lượng, chi phí thời gian và độ ổn định; không mặc định thay DenseCRF chỉ vì một benchmark tăng điểm.

## 4. Kiểm thử và tiêu chí hoàn thành

- Mỗi cấu hình fusion tạo bản đồ có cùng batch/kích thước không gian, giá trị hữu hạn và không làm sai lệch thứ tự tương ứng giữa query/support.
- Mỗi phương pháp ngưỡng trả về ngưỡng hữu hạn và mask nhị phân; T-PredMean khớp hành vi hiện hành `max(Otsu, mean)`.
- P0 giữ nguyên mask sau cắt ngưỡng; các nhánh hậu xử lý trả mask đúng kích thước, không có giá trị ngoài 0/1 và xử lý được ảnh/mask biên nhỏ.
- Tài liệu kết quả nêu rõ baseline, biến số được kiểm soát, metric, thời gian chạy, cấu hình tốt nhất và giới hạn/phụ thuộc của các phương pháp.

## Phạm vi và giả định

- Các chế độ fusion, threshold, Guided Filter và morphology đã được nối vào pipeline; mặc định vẫn là F0/T-PredMean/P0 để giữ hành vi cũ.
- Fusion chỉ thay đổi cách tổng hợp 13 map hiện được dùng; không thay đổi backbone, adapter, phép tính affinity hay kích thước đầu ra.
- Guided Filter và morphology là nhánh so sánh thử nghiệm; DenseCRF vẫn là phương án tham chiếu. `pydensecrf` phải được cài riêng nếu chọn `always` hoặc `dynamic`; chương trình không cài package tự động.

## 5. Chạy cấu hình

Các tùy chọn mới có thể kết hợp độc lập trong cùng lệnh:

```bash
python main.py --benchmark fss --datapath ./datasets/fss/ --nshot 1 \
  --fusion-method high_biased --threshold-method otsus --postprocessing guided
```

- `--fusion-method`: `mean`, `geomean`, `trimmed_mean`, `low`, `mid`, `high`, `stage_uniform`, `low_biased`, `high_biased`, `high_suppressed`, `weighted_stage`, `weighted_stage_geomean`, `agreement`, `pseudo_gate`.
- `--threshold-method`: `otsus`, `mean`, `pred_mean`, `otsu_mean`; `--threshold-otsu-weight` cấu hình λ cho `otsu_mean`.
- `--postprocessing`: `off`, `always`, `dynamic`, `guided`, `morph-open`, `morph-close`, `morph-open-close`, `guided-morph-open`, `guided-morph-close`, `guided-morph-open-close`.
- Dùng `--help` để xem mô tả tham số; chạy từng cấu hình trên cùng benchmark/episode/seed để lập bảng ablation. Kết quả benchmark đầy đủ cần được chạy trong môi trường đã cài dependencies và dữ liệu tương ứng.

## 6. Cấu hình tốt hơn baseline

Kết quả 1-shot, seed 0, 600 episode/tập (nhóm C: 200 episode Deepglobe). Baseline
tự chạy: A0+B0+C1 = **54.42 / 64.87**; A0+B0+C0 = **42.63 / 47.57** (Deepglobe).

| Nhóm | Biến thể | ΔmIoU | ΔFB-IoU | Cách chạy |
|---|---|---|---|---|
| A | `A8_geomean` — trung bình hình học 13 tầng | +0.19 | −0.23 | `--fusion-method geomean` |
| B | `B2_otsus` — Otsu thuần | +0.36 | −0.23 | `--threshold-method otsus` |
| C | `C2_crf_always` — DenseCRF luôn bật | +0.23 | +0.73 | `--postprocessing always` |

`A8_geomean` và `B2_otsus` tăng mIoU nhưng giảm nhẹ FB-IoU (trong ngưỡng ±0.3);
`C2_crf_always` tăng cả hai chỉ số nhưng mới đo trên Deepglobe. Mặc định vẫn giữ
F0/T-PredMean/P0 theo quy tắc không tinh chỉnh trên nhãn test; các biến thể trên
chỉ dùng khi chạy ablation có kiểm soát. Chi tiết đầy đủ ở `docs/exp.md`.
