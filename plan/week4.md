# Tuần 4 (11/11–17/11): đo cho chắc, chuẩn bị G2

**Hạn việc và report: 20:00 Thứ Ba 17/11 · Họp: 21:00 Thứ Ba 17/11**

Bảng lộ trình trong `KeHoach_SemEval2027_Task5.md` ghi tuần 4 là 04/11–10/11 vì G1 đã lùi. Từ nay theo chu kỳ Thứ Tư–Thứ Ba. **G2 vẫn 24/11.**
Kết quả họp G1 10/11 chưa nằm trong repo. Nếu G1 giao việc khác với mục của bạn dưới đây, theo G1 và báo Khang.
Căn cứ: `main` ngày 12/11 (report tuần 3 của Đạt và Khang, `docs/RESULTS.md`). Chưa thấy report của Cương và Phát.

## 1. Mục tiêu chung

Kết quả tuần 3 phần lớn mới 1 seed, và seed 42 chạy ở môi trường cũ hơn seed 43/44. Tuần 4 làm cho so sánh công bằng và kết luận tin được, rồi chuẩn bị hai thứ G2 cần: điểm trần, và đường từ checkpoint ra file dự đoán.

| Người | Sản phẩm cuối tuần |
|---|---|
| Đạt | Mốc B1/B2 seed 42 cùng môi trường mới; `PREDICT_kaggle.ipynb` v0 |
| Cương | Hướng "giảm lệch điểm" kiểm lại bằng seed 43/44 và trên PV |
| Khang | `results/analysis/ceiling.md`; số "dự đoán co về trung bình"; Sheet `experiments` đầy đủ |
| Phát | Danh sách lỗi PV (ưu tiên dòng fallback); ≥3 giả thuyết cho G2; RoBERTa thêm seed nếu còn đáng |

## 2. Quy tắc chung

- Chạy trên **bản copy Kaggle của mình**; không sửa notebook trong repo; không sửa `align`, `IRREGULAR`, `overrides`.
- Mỗi run: đợi `OUTPUTS_READY`, mở Output của **đúng saved version**, tải results ZIP + model ZIP về `results/{task}/{Model}_seed{N}/` ngay, ghi một dòng vào Sheet `experiments` kèm **link saved version**.
- **So cùng seed, cùng môi trường.** Run mới của bạn chạy image mới, nên so với B2 seed 43/44 của Đạt (cùng image). Seed 42 chỉ so với `B2_seed42_envmatch` khi Đạt xong (mục 4.1), không so với pilot cũ.
- **Quy tắc kết luận:** từ 3 seed, so trung bình ± SD (n−1). Chỉ ghi "tốt hơn" khi chênh trung bình **lớn hơn SD lớn nhất của hai bên**; không thì ghi "chưa phân biệt được". Dưới 3 seed luôn ghi "tạm thời". (Chưa có khoảng tin cậy; công cụ đó làm ở tuần 5.)
- **Hướng tuần 3 không tốt hơn mốc:** đừng chạy thêm seed. Chọn đúng một thay đổi mới theo mẫu *vấn đề → giả thuyết → một thay đổi → metric → giữ/bỏ* và nhắn Khang **trước 20:00 Thứ Sáu 13/11**.
- Làm theo thứ tự trong mục của mình; hết quota GPU thì dừng và ghi lại đã làm tới đâu.
- Report và Git chỉ ghi số và ID dòng, **không chép câu trong dữ liệu**.

## 3. Mốc để so

Mốc = **B2 gốc** (BERT, không đổi gì), dev, trung bình 3 seed 42/43/44 (từ `docs/RESULTS.md`). SD ở đây là dao động giữa các run, không phải nhiễu seed thuần.

| Output | Spearman ↑ | RMSE ↓ | MAE ↓ | Delta MAE ↓ |
|---|---:|---:|---:|---:|
| NN ModAvg | 0,4676 ± 0,0364 | 0,9285 | 0,7008 | 0,6182 |
| NN HeadAvg | 0,5816 ± 0,0256 | 0,7564 | 0,5765 | 0,5833 |
| PV Avg | 0,4972 ± 0,0229 | 0,9156 | 0,7426 | 0,7541 |

B0 (đoán trung bình), seed 42: NN ModAvg RMSE 0,938; PV RMSE 1,0849, MAE 0,9039.
Dev nhỏ: NN 67 expression, PV 24 expression, nên Spearman nhiễu; đừng đọc chênh vài phần trăm như kết quả thật.
Seed 42 (pilot 27/10, NN): B2 ModAvg 0,503 / RMSE 0,974; HeadAvg 0,552 / 0,808. Chỉ dùng để đối chiếu với run seed 42 của tuần 3.

## 4. Việc từng người

### 4.1 Đạt

**Mục tiêu:** có mốc seed 42 cùng môi trường với seed 43/44 để mọi hướng so công bằng, và có đường từ model ZIP ra file dự đoán **không cần gold**.

**Cần làm (theo thứ tự):**
1. **Hạn 20:00 Thứ Sáu 13/11:** chạy `handoff/week3_dat/followup/NN_B1_B2_seed42_envmatch_kaggle.ipynb` (2 run). Guard báo khác môi trường thì dừng và xử lý, không bỏ qua. Lưu `B1_seed42_envmatch`, `B2_seed42_envmatch`, giữ pilot cũ.
2. Điền hai link saved version còn thiếu vào `reports/week3/Dat.md` và Sheet.
3. Viết `PREDICT_kaggle.ipynb` v0 trên Kaggle: nạp model ZIP (`best.pt`, tokenizer, config), đọc TSV **không có gold** (`load_records(labeled=False)`), dùng `predict_checkpoint`, xuất NN hai số / PV một số. Chạy trên **toàn bộ dev** của 1 model NN và 1 model PV. (Khác tuần 3, chỉ kiểm 16 dòng.) Đưa vào repo bằng PR sau họp 17/11.

**Dùng gì:** `handoff/week3_dat/` (README, `verify_pv_checkpoint.py`), `docs/RUN_GUIDE.md`, `notebooks/NN_B*_kaggle.ipynb`, `predict_checkpoint` trong notebook, model ZIP đã nhận (`results/<task>/models_seeds43_44.zip`, `complete.zip` seed 42).

**Đánh giá:**
- Hai run envmatch đạt guard môi trường, đủ 5 file trong Output của đúng saved version; có bảng seed 42 envmatch vs pilot cũ (Spearman, RMSE từng output) để biết môi trường ảnh hưởng bao nhiêu.
- PREDICT: dự đoán lại toàn dev **khớp `predictions.tsv` đến 4 chữ số thập phân**, cả NN và PV; ghi số giây chạy.
- `reports/week3/Dat.md` có đủ link saved version.

### 4.2 Cương

**Mục tiêu:** biết hướng "giảm lệch điểm" của tuần 3 có còn đúng khi đổi seed và sang PV không.

**Cần làm:**
1. Giữ đúng thay đổi tuần 3 (khởi đầu `head[1].bias` bằng trung bình train; hoặc `lr=5e-5` nếu chẩn đoán đã loại giả thuyết bias). Chạy **B2 only** (đỡ quota): NN `SEED=43`, `44` (2 run).
2. PV B2 `SEED=43` với cùng thay đổi (1 run). Còn quota thì thêm seed 44.
3. So với B2 gốc **cùng seed, cùng task** (mục 3 và Sheet). Kết quả seed 42 của tuần 3 so với `B2_seed42_envmatch`.

**Dùng gì:** bản copy B2 đã sửa ở tuần 3, `predictions.tsv` (cột `absolute_error_*` để tính MAE), Sheet `experiments`.

**Đánh giá:**
- NN: **giữ** nếu RMSE trung bình giảm ở cả ModAvg và HeadAvg so với B2 gốc cùng seed (theo quy tắc mục 2), RMSE ModAvg ≤ 0,938, và Spearman không giảm quá 0,02.
- PV: báo 4 chỉ số so với B2 gốc PV cùng seed; dưới 3 seed thì ghi "tạm thời".

### 4.3 Khang

**Mục tiêu:** biết điểm trần của type-level (expression nói chung ghép nghĩa đến đâu) để G2 chọn đầu tư đọc ngữ cảnh hay kiến thức cấp expression; và kiểm giả thuyết "dự đoán co về trung bình" mà B3 gợi ra.

B3 tuần 3 **không giữ** (1 seed, trượt cả ba điều kiện). Không chạy thêm seed B3.

**Cần làm:**
1. Viết `results/analysis/ceiling.md` (≤1 trang, không trích câu):
   - Kiểm tra dữ liệu có điểm từng người chấm không. Schema chỉ có `Avg`, `Std` nên split-half **không tính được**; ghi rõ.
   - **Type-level oracle:** dự đoán mỗi dòng dev bằng gold trung bình của expression chứa nó (chỉ để tính trần, không phải hệ thống). Chấm Spearman/MAE/RMSE, đặt cạnh B0, B1, B2.
   - Phân rã phương sai (ANOVA một chiều hoặc ICC): tỉ lệ phương sai giữa expression / trong expression, riêng `ModAvg`, `HeadAvg`, `Avg`.
   - 3–5 câu kết luận.
2. Không cần GPU: từ `predictions.tsv` tính (a) MAE tổng B3 (hai ô còn thiếu trong report tuần 3); (b) `SD(dự đoán) / SD(gold)` cho B1, B2, B3, từng output, seed 42 (B1/B2 thêm seed 43/44 của Đạt). Kiểm B3 có co về trung bình không; số này cũng là bằng chứng cho hướng của Cương.
3. Sheet `experiments` đủ mọi run tuần 3–4 (mã run, người chạy, task, model, seed, cấu hình khác mặc định, metrics, best epoch, link saved version, ngày); ghi B3 là run âm tính.
4. Nếu chưa xong từ tuần 3: email ban tổ chức (5 câu hỏi ở `week3.md` mục 4.3), thử lại CodaBench 17971.

**Dùng gì:** results ZIP dev của B0/B1/B2/B3, `evaluate` trong `src/compartment/metrics.py` (chấm oracle), pandas/numpy trong `.venv`.

**Đánh giá:**
- `ceiling.md` có đủ: số oracle cạnh B0/B1/B2, tỉ lệ phương sai cho 3 output, kết luận; hoặc ghi rõ vì sao không tính được.
- Có bảng `SD(dự đoán) / SD(gold)` và kết luận 1–2 câu (có co / không co).
- Sheet đủ run, không thiếu cột.

### 4.4 Phát

**Mục tiêu:** có lỗi cụ thể của PV và ≥3 giả thuyết "một thay đổi" cho G2; kiểm RoBERTa thêm seed nếu hướng còn đáng.

**Cần làm:**
1. **Đọc lỗi PV:** từ results ZIP PV B1/B2 seed 42 của Đạt, chọn 10–20 dòng dev gồm B2 tốt hơn B1, B2 tệ hơn B1, và dòng bị fallback. **Ưu tiên fallback:** PV B2 fallback 43/231 dòng dev (18,6%), cao hơn xa ngưỡng 5% của plan. Ghi vào `results/analysis/pv_errors.tsv` (ngoài Git): ID dòng, gold, dự đoán B1/B2, spans, lý do nghi ngờ. Không sửa gold.
2. **Giả thuyết cho G2:** viết ≥3 giả thuyết theo mẫu *vấn đề → giả thuyết → một thay đổi → metric/slice kiểm tra → điều kiện giữ/bỏ*; mỗi cái dựa trên ≥3 dòng lỗi (PV bước 1 hoặc NN tuần 3). Không sửa `align`; ca khó xử lý ở bước tokenize trong notebook (phương án P1, `week2.md` mục 7).
3. **RoBERTa:** nếu tuần 3 là "tốt hơn" hoặc "chưa kết luận": B2 RoBERTa NN `SEED=43`, `44` (2 run). Nếu "không tốt hơn": bỏ bước này.

**Dùng gì:** results ZIP PV của Đạt; web (`streamlit run app/streamlit_app.py`, Results explorer) để xem từng dòng; `results/pv/comparison_seed42/paired_error_analysis.tsv` nếu có; `alignment_failures.tsv`; bản copy B2 RoBERTa (tuần 3).

**Đánh giá:**
- `pv_errors.tsv` đủ trường cho 10–20 dòng, có số dòng fallback theo nhãn lỗi.
- Mỗi giả thuyết có đủ 5 phần, và kiểm tra được bằng một chỉ số đã có trong `evaluate` / `error_groups`.
- RoBERTa: **giữ** nếu Spearman trung bình 3 seed cao hơn B2 gốc và RMSE không cao hơn (theo quy tắc mục 2).

## 5. Report cuối tuần: mẫu để điền

Mỗi người nộp **một file `reports/week4/<Tên>.md`, tối đa 1 trang** (thư mục đã có trong repo, như `reports/week3/`) trước 20:00 Thứ Ba 17/11. Khang đọc dòng "Kết luận" và mục 2.

````markdown
# Report tuần 4: <Tên> · việc: <tên việc>

**Kết luận 1 dòng:** [tốt hơn mốc / không tốt hơn mốc / chưa phân biệt được] vì ...
(Dưới 3 seed thì ghi "tạm thời".)

## 1. Đã chạy gì
| Run | Task | Seed | GPU | Số phút | Link saved version Kaggle |
|---|---|---|---|---|---|
| | | | | | |

## 2. Kết quả so với mốc (dev, trung bình ± SD qua seed)
| Output · chỉ số | Mốc (B2 gốc) | Của mình | Chênh | Hơn / kém / chưa phân biệt |
|---|---:|---:|---:|---|
| NN ModAvg · Spearman ↑ | | | | |
| NN ModAvg · RMSE ↓ | | | | |
| NN ModAvg · MAE ↓ | | | | |
| NN ModAvg · Delta MAE ↓ | | | | |
| NN HeadAvg · Spearman ↑ | | | | |
| NN HeadAvg · RMSE ↓ | | | | |
| NN HeadAvg · MAE ↓ | | | | |
| NN HeadAvg · Delta MAE ↓ | | | | |
| PV Avg · Spearman ↑ | | | | |
| PV Avg · RMSE ↓ | | | | |
| PV Avg · MAE ↓ | | | | |
| PV Avg · Delta MAE ↓ | | | | |
(Dòng nào không chạy thì bỏ trống.)

## 3. Phần riêng của việc mình
- Đạt: bảng seed 42 envmatch vs pilot cũ; PREDICT khớp / không khớp `predictions.tsv` (lệch bao nhiêu).
- Cương: nếu đổi hướng, kết quả chẩn đoán và thay đổi mới.
- Khang: số oracle, tỉ lệ phương sai, bảng `SD(dự đoán) / SD(gold)`, kết luận.
- Phát: tên file `pv_errors.tsv` (chỉ ID dòng), ≥3 giả thuyết đủ 5 phần.

## 4. Hiểu thêm được gì, có gì bất ngờ
- 

## 5. Bug và vướng mắc
- Bug đã gặp, đã đi tiếp: 
- Đang bị chặn, cần giúp: 

## 6. Đề xuất cho G2 (24/11)
- Giữ / bỏ hướng của mình, vì sao (1–2 dòng):

Giờ làm tay: ... · Giờ GPU đã dùng: ...
````
