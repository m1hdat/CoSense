# Report tuần 3: Khang · hướng B3 (thêm vector cả câu vào phần dự đoán)

**Kết luận (1 seed, chỉ dev): không giữ B3.** So với B2 cùng seed 42, B3 trượt cả ba điều kiện "Giữ nếu" (Delta MAE, Var. corr., Spearman). Điểm sáng duy nhất là RMSE giảm (ModAvg 0,974 → 0,811), chưa rõ nguyên nhân và không nằm trong điều kiện giữ. Giả thuyết "thêm vector cả câu thì điểm đổi theo câu hợp lý hơn" **chưa được ủng hộ** (không phải đã bị bác bỏ: một seed, chênh lệch nhỏ).

## 1. Đã chạy gì

| Run | Seed | GPU | Thời gian | Saved version Kaggle |
| --- | --- | --- | --- | --- |
| NN B3 FULL, best epoch 4/8, early stop ở epoch 6 | 42 | GPU Kaggle | ≈ 8,3 phút (train 498 s) | https://www.kaggle.com/code/thaonguyencute/b3-kaggle |

Log báo `OUTPUTS_READY` (5 file: results ZIP, model ZIP, 2 CSV, `ARTIFACTS.json`). Không chạy PV. Không đánh giá test (`EVALUATE_LOCAL_TEST=False`).

## 2. Kết quả so với mốc (NN, dev 504 dòng)

Mốc = B2 gốc, cùng seed 42, cùng split và manifest. Δ = B3 − mốc.

| Output | Chỉ số | Mốc (B2) | B3 | Δ | B3 tốt hơn? |
| --- | --- | ---: | ---: | ---: | --- |
| ModAvg | Spearman ↑ | 0,503 | 0,476 | −0,027 | không |
| ModAvg | RMSE ↓ | 0,974 | 0,811 | −0,163 | có |
| ModAvg | MAE ↓ | 0,721 | chưa tính | | ? |
| ModAvg | Delta MAE ↓ | 0,602 | 0,608 | +0,006 | không |
| ModAvg | Var. corr. ↑ | 0,225 | 0,072 | −0,153 | không |
| HeadAvg | Spearman ↑ | 0,552 | 0,550 | −0,002 | không (coi như bằng) |
| HeadAvg | RMSE ↓ | 0,808 | 0,780 | −0,028 | có |
| HeadAvg | MAE ↓ | 0,602 | chưa tính | | ? |
| HeadAvg | Delta MAE ↓ | 0,601 | 0,603 | +0,002 | không |
| HeadAvg | Var. corr. ↑ | 0,181 | 0,176 | −0,005 | không (coi như bằng) |

Đối chiếu luật "Giữ nếu":

| Điều kiện giữ B3 | B3 đạt được | Đạt? |
| --- | --- | --- |
| Delta MAE thấp hơn mốc ở ≥ 1 output | 0/2 output (+0,006; +0,002) | không |
| Var. corr. cao hơn mốc | 0/2 output (−0,153; −0,005) | không |
| Spearman không giảm | ModAvg giảm 0,027; HeadAvg giảm 0,002 | không |

MAE tổng: notebook không in. Tính từ cột `absolute_error_*` trong `predictions.tsv` (results ZIP); hai ô MAE không đổi kết luận vì luật không dùng MAE.

## 3. Đã đổi đúng một thứ

Chọn biến thể B3 thay B2. Phần dự đoán nhận **3 vector thay vì 2** (rộng 768×3 thay vì 768×2):

```python
pooled = [pool(hidden, part1_mask), pool(hidden, part2_mask)]
if variant == 'B3': pooled.append(pool(hidden, context_mask))   # trung bình các token còn lại của câu
```

`context_mask` = token của câu **trừ** hai thành phần của cụm và token đặc biệt. Dòng bị fallback về cụm từ (không có câu để pool) thì vector này bằng 0. Nhánh B3 đã có sẵn trong code dùng chung; so với notebook B2 (diff đã kiểm), notebook B3 chỉ đổi nhãn `B2 → B3` và thêm `'B3'` vào danh sách `['B1','B2']` của `publish_run_artifacts`. Mọi thứ khác giữ nguyên B2: backbone `bert-base-uncased` (pin revision), `lr=2e-5`, batch 8×2, 8 epoch, patience 2, split và manifest đã khóa.

## 4. Hiểu thêm được gì

- **Delta MAE không đổi**: chênh +0,006 / +0,002, cùng cỡ nhiễu của một seed. Không thấy bằng chứng vector cả câu giúp nhận ra điểm đổi theo câu.
- **RMSE giảm rõ nhưng Spearman (ModAvg) giảm**: hai chỉ số đi ngược chiều. RMSE ModAvg 0,811 còn thấp hơn B1 (0,899) và đoán trung bình B0 (0,938). Giả thuyết chưa kiểm: prediction của B3 co về gần giá trị trung bình (ít phân tán hơn) nên sai số lớn ít đi nhưng thứ hạng kém đi. Kiểm bằng độ lệch chuẩn của prediction trong `predictions.tsv`, B3 so B2. Một seed nên chưa quy cho B3.
- **Var. corr. nhiễu**: theo epoch ModAvg dao động 0,03–0,15 (0,10 / 0,03 / 0,06 / 0,07 / 0,15 / 0,10). Dev chỉ có 67 expression, nên Spearman trên ≤ 67 điểm có sai số chuẩn cỡ 0,12 (ước lượng thô). Chênh 0,153 so với B2 chỉ khoảng 1,2 lần sai số đó, chưa tách được khỏi nhiễu.
- **Overfit**: train loss giảm đến 0,11, dev tốt nhất ở epoch 4 (Spearman trung bình 0,513). Early stopping đã xử lý; chưa có số B2 để so nên không kết luận B3 overfit nhanh hơn.

## 5. Bug và vướng mắc

- Bug: không. Run FULL chạy hết. Cảnh báo `UNEXPECTED cls.*` khi nạp BERT là các head pretraining không dùng, bỏ qua được.
- Đang bị chặn: không.

## 6. Tuần sau đề xuất

1. Ghi B3 vào sheet `experiments` như run âm tính; không đưa vào bảng so B1/B2.
2. Từ `predictions.tsv` của B2 và B3 (không cần GPU): tính MAE tổng (điền hai ô còn thiếu) và độ lệch chuẩn prediction để kiểm giả thuyết "co về trung bình" ở mục 4.
3. Chỉ chạy thêm seed nếu nhóm cần chốt chắc: B2 và B3 cùng seed 43, 44 (4 run, mỗi run ≈ 8 phút). Rẻ hơn: dùng dao động giữa seed Đạt đã đo để xem chênh +0,006 của Delta MAE có vượt nhiễu không.

