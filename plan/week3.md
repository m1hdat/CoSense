# Tuần 3 (02/11–10/11): mỗi người tự train một hướng

## 1. Tuần này làm gì

Dữ liệu đã xong. Mỗi người **tự train một mô hình trên Kaggle của mình**, đổi đúng **một thứ**, so kết quả với mốc ở mục 3, rồi viết report theo mẫu ở mục 6.

- Chỉnh notebook trên **bản copy ở Kaggle**, không sửa notebook trong repo. Hướng nào tốt thì sau họp G1 mới đưa vào repo.
- Gặp bug: nếu run vẫn chạy được thì **ghi lại một dòng rồi đi tiếp, sửa sau**. Chỉ dừng khi run không chạy nổi, và báo nhóm ngay.
- Mới chạy một seed thì kết luận chỉ là **tạm thời**.
- Không sửa `align`, `IRREGULAR`, `overrides`: sửa sẽ làm lệch dữ liệu đã chia sẵn, các run không còn so được với nhau.

## 2. Các từ cần biết

| Từ | Nghĩa |
|---|---|
| **NN** (noun compound) | Danh từ ghép, ví dụ *flea market*. Mỗi câu có **2 điểm** cần đoán: `ModAvg` (từ đứng trước, *flea*) và `HeadAvg` (từ chính, *market*) |
| **PV** (particle verb) | Động từ + tiểu từ, ví dụ *take off*. Mỗi câu có **1 điểm** cần đoán: `Avg` |
| Điểm | Thang 0–5. Cao = thành phần đó giữ nghĩa gốc rõ trong cụm; thấp = ít giữ nghĩa |
| **B0** | Mốc ngốc nhất: luôn đoán điểm trung bình của tập train |
| **B1** | Mô hình **chỉ đọc cụm từ** (ví dụ `flea market`), không đọc câu |
| **B2** | Mô hình **đọc cả câu** chứa cụm từ |
| **B3** | Như B2, nhưng thêm thông tin của cả câu khi đoán (hướng của Khang) |
| **BERT, RoBERTa** | Mô hình ngôn ngữ có sẵn, mình chỉ tinh chỉnh cho bài này. Hiện đang dùng BERT |
| **dev** | Tập dùng để so các mô hình với nhau (không phải tập thi thật) |
| **seed** | Số quyết định phần ngẫu nhiên của một lần train. Đổi seed thì kết quả dao động nhẹ ("nhiễu seed") |
| **run, smoke** | Run = một lần chạy notebook trên Kaggle. Smoke = chạy thử nhanh (16 dòng, 1 epoch), chỉ để chắc không lỗi |
| **Spearman** ↑ | Xếp thứ tự đúng không (−1 đến 1, cao = tốt) |
| **RMSE, MAE** ↓ | Đoán lệch trung bình bao nhiêu điểm (thấp = tốt; RMSE phạt lệch lớn nặng hơn) |
| **Delta MAE** ↓ | Với các câu cùng một cụm từ: đoán đúng mức điểm *thay đổi* giữa các câu không (thấp = tốt) |
| **Var. corr.** ↑ | Cụm nào có điểm dao động nhiều giữa các câu thì model có nhận ra không (cao = tốt) |

## 3. Mốc cần vượt

Kết quả đã có: NN, tập dev, seed 42, BERT. **"B2 gốc" trong tài liệu này là dòng B2 ở bảng này.**

| Output | Model | Spearman ↑ | MAE ↓ | RMSE ↓ | Delta MAE ↓ | Var. corr. ↑ |
|---|---|---:|---:|---:|---:|---:|
| ModAvg | B0 (đoán trung bình) | không tính | | 0,938 | | |
| ModAvg | B1 | 0,446 | 0,676 | 0,899 | 0,585 | không tính |
| ModAvg | **B2 gốc** | 0,503 | 0,721 | 0,974 | 0,602 | 0,225 |
| HeadAvg | B1 | 0,549 | 0,582 | 0,794 | 0,546 | không tính |
| HeadAvg | **B2 gốc** | 0,552 | 0,602 | 0,808 | 0,601 | 0,181 |

Ba điều rút ra, cũng là chỗ cần cải thiện:

- B2 **xếp thứ tự tốt hơn B1 nhưng lệch điểm nhiều hơn**: RMSE của ModAvg (0,974) còn tệ hơn đoán trung bình (0,938).
- B2 **chưa đoán tốt sự thay đổi điểm giữa các câu** của cùng một cụm từ (Delta MAE không thấp hơn B1).
- Với HeadAvg, đọc thêm câu gần như không giúp gì (0,549 → 0,552).

PV (cụm động từ) chưa có kết quả nào, và mới chạy một seed nên chưa biết dao động giữa các lần train lớn cỡ nào.

## 4. Việc của từng người

Bốn hướng độc lập nhau. Chỉ Đạt làm phần PV và đo dao động seed; ba người còn lại làm trên NN (danh từ ghép).

### 4.1 Đạt: chạy PV và đo dao động giữa các lần train

**Mục tiêu:** có kết quả B1 và B2 cho **PV** (phần chưa chạy lần nào) và biết kết quả NN dao động bao nhiêu giữa các seed. Không có con số dao động này thì không ai kết luận được hướng của mình.

1. **03/11:** mở lại 2 lần chạy NN cũ trên Kaggle, xem còn file model (`best.pt`, `tokenizer/`, `encoder_config/`) không. Còn thì tải về; mất thì chạy lại NN seed 42 ở bước 4.
2. **03/11, smoke PV B1:** ở cell 4 đổi `TASK = 'pv'`, `SMOKE = True`, chạy lại từ cell cấu hình trở xuống. Thấy `OUTPUTS_READY` thì mở **Output của đúng saved version**, kiểm tra đủ 5 file (results ZIP, model ZIP, 2 CSV, `ARTIFACTS.json`).
3. **PV B1 xong 05/11, PV B2 xong 06/11:** `SEED=42`, `SMOKE=False`, `EVALUATE_LOCAL_TEST=False`. Tải về ngay; ghi số phút chạy và loại GPU.
4. **08/11:** NN B1 và B2 với `SEED=43` và `SEED=44` (4 run).
5. **09/11:** kiểm tra model đã lưu dùng được: nạp model ZIP của PV B1, dự đoán 16 dòng dev bằng `predict_checkpoint` (cell 8), phải khớp `predictions.tsv` trong results ZIP.

**Trả lời hai câu:** (a) PV có giống NN không, tức B2 xếp thứ tự tốt hơn B1 nhưng lệch điểm không giảm? (b) Giữa các seed, Spearman dao động bao nhiêu (trung bình ± độ lệch chuẩn) so với chênh B2 − B1 ở ModAvg (0,057)?

**Dễ sai:** quên đổi `TASK` (mặc định là `'nn'`); tin thông báo trong session mà không mở Output của saved version.

### 4.2 Cương: giảm lệch điểm của B2

**Vấn đề:** B2 lệch điểm nhiều (RMSE ModAvg 0,974, tệ hơn đoán trung bình 0,938) dù xếp thứ tự tốt.
**Giả thuyết:** lớp cuối của mô hình bắt đầu từ 0, trong khi điểm thật có trung bình khác 0. Với tốc độ học `2e-5`, lớp này khó tự lên tới mức trung bình, nên mô hình bù bằng cách kéo lệch cả phần còn lại.
**Thay đổi duy nhất:** đặt giá trị khởi đầu của `head[1].bias` bằng điểm trung bình của tập train. Áp dụng được cho cả B1 và B2 nên **chạy cả hai**.

1. **03/11, chẩn đoán (không cần GPU, 20 phút):** từ `predictions.tsv` của 2 results ZIP NN cũ, mỗi output và mỗi model tính ba số: (a) trung bình của `dự đoán − điểm thật`; (b) `SD(dự đoán) / SD(điểm thật)`; (c) RMSE sau khi chỉnh `a·dự đoán + b` trên chính dev (chỉ để biết còn gỡ được bao nhiêu, không báo như kết quả). **Nếu (c) gần như không giảm so với RMSE gốc** (tạm: giảm dưới 0,02) thì giả thuyết sai: ghi lại, rồi đổi sang thử `lr = 5e-5` (chỉ sửa thông số).
2. Sửa bản copy: trong `train()` (cell 8), ngay sau khi tạo `Regressor`, đặt `head[1].bias` bằng trung bình nhãn train của từng output (bỏ qua khi `resume`). Khoảng 4 dòng.
3. Chạy NN B1 và B2 seed 42 (smoke trước). Tải về thư mục riêng, ví dụ `results/nn/B2bias_seed42/`.

**Giữ nếu:** RMSE của B2 giảm ở cả hai output (dưới 0,974 và 0,808), RMSE ModAvg ≤ 0,938, Spearman không giảm đáng kể (tạm: dưới 0,02).
**Lưu ý:** notebook chưa in MAE tổng; lấy từ cột `absolute_error_*` trong `predictions.tsv`.

### 4.3 Khang: B3, cho mô hình biết thêm thông tin cả câu

**Vấn đề:** B2 chưa đoán tốt sự thay đổi điểm giữa các câu của cùng một cụm từ (Delta MAE ≈ 0,60 không thấp hơn B1; Var. corr. chỉ ≈ 0,2).
**Giả thuyết:** B2 chỉ cho phần dự đoán nhìn 2 thành phần của cụm. Cho thêm thông tin trung bình của cả câu thì điểm sẽ đổi theo câu hợp lý hơn.
**Thay đổi duy nhất:** dùng B3. Code B3 **đã có sẵn trong notebook**, chưa ai train.

1. Copy `B2_kaggle` thành `B3_...`, đổi `'B2'` thành `'B3'` ở cell 4, 12, 14, 16 (Ctrl+F tìm `B2`), và ở **cell 7**: danh sách `['B1','B2']` trong `publish_run_artifacts` phải thêm `'B3'`. Sót chỗ nào thì run báo lỗi: ghi lại rồi sửa tiếp.
2. Smoke, rồi chạy NN B3 seed 42. Tải về `results/nn/B3_seed42/`.
3. B3 là biến thể riêng, không đưa vào bảng so B1 với B2.

**Giữ nếu:** Delta MAE thấp hơn B2 gốc (0,602 / 0,601) ở ít nhất 1 output, Var. corr. cao hơn (0,225 / 0,181), Spearman không giảm.

**Việc khác của Khang (không phải train):**
- Tạo Google Sheet `experiments`, mỗi run một dòng.
- Gửi email tới ban tổ chức (`semeval-compartment@ims.uni-stuttgart.de`) hỏi 5 điều: thước đo chính thức; nộp riêng English có xếp hạng không; được dùng LLM, mô hình có sẵn, dữ liệu ngoài không; tên file và cột file nộp; số lần nộp và ngày phát hành test.
- Thử lại CodaBench 17971.

### 4.4 Phát: thử RoBERTa thay BERT

**Vấn đề:** mới dùng BERT, chưa biết hiện tượng của B2 ("xếp thứ tự tốt nhưng lệch điểm") là của bài toán hay của BERT.
**Giả thuyết:** RoBERTa (mô hình mạnh hơn) hiểu nghĩa từng thành phần trong câu tốt hơn, nên B2 tốt hơn cả về thứ tự lẫn độ lệch.
**Thay đổi duy nhất:** đổi `backbone='roberta-base'` và `revision` ở cell 4, cho cả B1 và B2.

1. Lấy mã commit của `roberta-base` trên trang Hugging Face (mục Files and versions) để điền `revision`; không để nguyên mã của BERT.
2. Smoke B1 rồi B2. Xem `run_metadata.json` mục `preprocessing` (số dòng bị cắt, số dòng fallback): RoBERTa tách từ khác BERT. Nếu báo lỗi `Incomplete constituent tokens` hàng loạt thì ghi lại và báo nhóm.
3. Chạy NN B1 và B2 seed 42. Tải về `results/nn/B2roberta_seed42/`, `results/nn/B1roberta_seed42/`.
4. **Đọc lỗi (việc của Phát):** chọn 10–20 dòng dev gồm B2 RoBERTa tốt hơn B2 BERT, tệ hơn, và dòng bị fallback. Ghi điểm thật, dự đoán, vì sao nghi ngờ; không tự sửa điểm thật.

**Giữ nếu:** Spearman của B2 RoBERTa cao hơn B2 gốc (0,503 / 0,552) và RMSE không cao hơn (0,974 / 0,808). Chỉ tốt một mặt thì ghi "chưa kết luận" và chạy thêm seed 43.

## 6. Report cuối tuần: mẫu để điền

Mỗi người nộp **một file `week3_<tên>.md`, tối đa 1 trang**, vào thư mục Drive `CompartMent/reports/` (hoặc dán vào chat nhóm). Chỉ ghi số và ID dòng, **không chép nguyên câu trong dữ liệu** (quyền dữ liệu chưa rõ). Từ tuần 4 dùng lại mẫu này, chỉ đổi số tuần. Khang đọc dòng "Kết luận" và mục 2 là đủ.

````markdown
# Report tuần 3: <Tên> · hướng: <tên hướng>

**Kết luận 1 dòng:** hướng này [tốt hơn mốc / không tốt hơn mốc / chưa kết luận] vì ...
(Mới chạy 1 seed thì ghi "tạm thời".)

## 1. Đã chạy gì
| Run | Seed | GPU | Số phút | Link saved version Kaggle |
|---|---|---|---|---|
| | | | | |

## 2. Kết quả so với mốc (NN, tập dev)
| Output | Chỉ số | Mốc (B2 gốc) | Của mình | Tốt hơn mốc? |
|---|---|---:|---:|---|
| ModAvg | Spearman ↑ | 0,503 | | |
| ModAvg | RMSE ↓ | 0,974 | | |
| ModAvg | MAE ↓ | 0,721 | | |
| ModAvg | Delta MAE ↓ | 0,602 | | |
| HeadAvg | Spearman ↑ | 0,552 | | |
| HeadAvg | RMSE ↓ | 0,808 | | |
| HeadAvg | MAE ↓ | 0,602 | | |
| HeadAvg | Delta MAE ↓ | 0,601 | | |
(Đạt: thay bằng bảng PV B1 so với PV B2, và bảng NN B1/B2 trung bình ± độ lệch chuẩn qua 3 seed.)

## 3. Mình đã đổi đúng một thứ gì
(1–3 dòng; dán dòng code hoặc thông số đã sửa)

## 4. Hiểu thêm được gì, có gì bất ngờ
- 

## 5. Bug và vướng mắc
- Bug đã gặp, đã đi tiếp: 
- Đang bị chặn, cần giúp: 

## 6. Tuần sau mình đề xuất làm gì
- 

Giờ làm tay: ... · Giờ GPU đã dùng: ...
````
