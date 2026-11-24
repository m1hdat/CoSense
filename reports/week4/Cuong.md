# Report tuần 4: Cương — đối chứng cách chọn checkpoint

**Kết luận tạm thời: không thay selector Spearman bằng mean RMSE trong recipe hiện tại.** Trên cả hai seed NN43/44, selector RMSE giảm lỗi modifier nhưng tăng lỗi head; NN44 còn giảm Spearman quá ngưỡng 0,02. PV43 cùng chọn epoch1, predictions giống hệt. Đây là kết quả dev với hai seed NN và một seed PV, chưa phải kết luận trên test.

Cập nhật 24/11/2026, phục vụ G2. Phần tổng hợp hoàn thành sau hạn tuần 4 (17/11).

## 1. Đã thực hiện

Tuần 3 thử khởi tạo bias theo trung bình train trên NN seed42 nhưng không đạt điều kiện giữ; [report tuần 3](../week3/Cuong.md) và số liệu gốc được giữ nguyên. Theo quy tắc đổi hướng khi không tốt hơn mốc, tuần 4 không mở rộng bias; chuyển sang đối chứng **selector checkpoint**. Đã hoàn thành rà soát bốn results ZIP cũ và kiểm chứng artifacts của đủ ba run mới:

| Run B2 | Seed | GPU | Phút theo metadata | Epochs chạy | Epoch Spearman / RMSE | Saved version |
|---|---:|---|---:|---:|---|---|
| NN | 43 | Tesla T4 | 4.09 | 3 | 1 / 3 | [373190543](https://www.kaggle.com/code/azoncx/nn-b2-seed43/output?scriptVersionId=373190543) |
| NN | 44 | Tesla T4 | 7.07 | 5 | 3 / 1 | [373192597](https://www.kaggle.com/code/azoncx/nn-b2-seed44/notebook?scriptVersionId=373192597) |
| PV | 43 | Tesla T4 | 2.01 | 3 | 1 / 1 | [373194424](https://www.kaggle.com/code/azoncx/pv-b2-seed43/notebook?scriptVersionId=373194424) |

Ba lần train, mỗi lần xuất hai checkpoint; không tính thành sáu run hoặc ba seed của cùng task. NN dùng 2.479 train rows và 504 dev rows/67 expressions; PV dùng 1.100 train rows và 231 dev rows/24 expressions. Tất cả giữ frozen split seed42, chỉ đánh giá dev.

## 2. Kết quả so với đối chứng

Đối chứng chính là checkpoint Spearman **của cùng lần train**, không dùng điểm từ run khác thay vào. `S` = chọn Spearman trung bình cao nhất; `R` = chọn RMSE trung bình thấp nhất. NN dùng một checkpoint chung cho hai output.

| Task/seed | Selector | Output | Spearman ↑ | RMSE ↓ | MAE ↓ | Delta MAE ↓ | Var. corr. ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| NN43 | S | ModAvg | 0.42988 | 0.93384 | 0.71799 | 0.62287 | 0.13285 |
| NN43 | S | HeadAvg | 0.59602 | 0.72922 | 0.56539 | 0.56420 | 0.30062 |
| NN43 | R | ModAvg | 0.44350 | 0.89451 | 0.67654 | 0.61981 | 0.16969 |
| NN43 | R | HeadAvg | 0.57772 | 0.74342 | 0.56652 | 0.61260 | 0.25704 |
| NN44 | S | ModAvg | 0.47036 | 0.87731 | 0.66339 | 0.63016 | 0.22564 |
| NN44 | S | HeadAvg | 0.59670 | 0.73198 | 0.56224 | 0.58511 | 0.37158 |
| NN44 | R | ModAvg | 0.43844 | 0.84520 | 0.64412 | 0.61383 | 0.26483 |
| NN44 | R | HeadAvg | 0.55113 | 0.76143 | 0.60492 | 0.58135 | 0.41599 |
| PV43 | S | Avg | 0.52367 | 0.91181 | 0.74283 | 0.76953 | 0.35217 |
| PV43 | R | Avg | 0.52367 | 0.91181 | 0.74283 | 0.76953 | 0.35217 |

Điều kiện giữ NN đã đặt trước run: RMSE giảm ở **cả** modifier/head, modifier RMSE ≤0,938, Spearman mỗi output không giảm quá 0,02. Chênh dưới đây là `R − S`; error âm tốt hơn, Spearman dương tốt hơn.

| Run | Δ RMSE modifier | Δ RMSE head | Δ Spearman modifier | Δ Spearman head | Kết luận |
|---|---:|---:|---:|---:|---|
| NN43 | -0.03933 | +0.01420 | +0.01362 | -0.01830 | Không đạt RMSE head |
| NN44 | -0.03210 | +0.02945 | -0.03192 | -0.04557 | Không đạt RMSE head và Spearman |
| PV43 | Không áp dụng | Không áp dụng | Không đổi | Không áp dụng | Cùng epoch1; bốn metric chính đều không đổi |

NN43 đạt ngưỡng Spearman nhưng RMSE head tăng; MAE và Delta MAE head cũng tăng. NN44 chỉ giảm mean RMSE rất ít (0,80464 →0,80332), trong khi Spearman modifier/head giảm lần lượt 0,03192/0,04557. Delta MAE NN44 giảm ở cả hai output là điểm thuận lợi riêng, không thay tiêu chí giữ. PV43 không cho bằng chứng cải thiện; cùng checkpoint ở một run không chứng minh hai selector tương đương nói chung.

## 3. Thiết kế và căn cứ đổi hướng

**Vấn đề → giả thuyết:** checkpoint tối ưu thứ hạng không nhất thiết tối ưu sai số điểm. Kiểm tra liệu chọn mean RMSE có giảm sai số mà vẫn giữ Spearman trong ngưỡng chấp nhận.

**Đúng một thay đổi:** so hai selector trên cùng quá trình train. Giữ BERT/revision, khởi tạo baseline, masked MSE, lr=2e-5, optimizer, batch, dữ liệu, seed và **early stopping theo Spearman**. Không kết hợp bias hoặc đổi learning rate. Hai selector chỉ được chọn trong các epoch đã chạy; đây không phải thí nghiệm early stopping theo RMSE. Mean RMSE của NN là trung bình số học RMSE modifier/head, không phải RMSE gộp.

Rà soát lịch sử tuần 3 cho thấy:

| Run seed42 cũ | Epoch chọn theo Spearman | Mean RMSE tại đó | Epoch mean RMSE thấp nhất | Mean RMSE thấp nhất |
|---|---:|---:|---:|---:|
| B1 baseline | 7 | 0,84682 | 4 | 0,80317 |
| B2 baseline | 3 | 0,89111 | 2 | 0,81284 |
| B1 bias | 8 | 0,82238 | 4 | 0,80641 |
| B2 bias | 3 | 0,83047 | 4 | 0,81014 |

Đây là audit run cũ, không tính thành run tuần 4. History cũ chỉ lưu mean, không đủ suy ra metric riêng từng output ở epoch không export. Đối chứng mới đã bổ sung metrics từng output theo epoch và predictions của cả hai selector.

RMSE tính độc lập từ predictions khớp số lưu; qua rà mã chưa thấy sai công thức masked MSE hoặc đảo thứ tự output. Chưa có bằng chứng quy nguyên nhân cho BERT hoặc learning rate. Phân rã `MSE = mean(error)² + mean((error−mean(error))²)` cho thấy B2 bias giảm lệch trung bình nhưng tăng phần sai số còn lại ở cả hai output. Centered RMSE chỉ là chẩn đoán dùng gold dev, không phải kết quả mô hình mới.

Đính chính giả thuyết ban đầu: `nn.Linear` mặc định khởi tạo bias ngẫu nhiên gần 0, không đúng bằng 0; affine fit trên dev không xác định nguyên nhân lỗi. [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Linear.html)

## 4. Kiểm chứng và bàn giao

- Hash/size khớp ARTIFACTS và SELECTION_ARTIFACTS; CRC toàn bộ ba ZIP của mỗi run đạt. Results nhúng trong complete ZIP trùng byte với bản riêng; có checkpoint/config/tokenizer cho cả hai selector.
- Tính lại Spearman, RMSE, MAE, Delta MAE và variance correlation khớp bản lưu với sai số ≤1e-12. ID/gold/context/spans, manifests và train/dev IDs khớp trong từng run; không có test exports.
- Epoch lựa chọn đúng cực trị trong `epoch_metrics.json`; metrics của epoch chọn khớp predictions được export. NN43/44 khớp dữ liệu, split, recipe trừ training seed và metadata môi trường. So sánh trong cùng run dùng chung trajectory theo metadata.
- Runtime: Tesla T4, Python3.13.16, torch2.11.0+cu130, transformers5.20.2, tokenizers0.23.3, numpy2.1.3, pandas2.3.3, scipy1.16.3. NN fallback train68/2.479, dev14/504; PV train161/1.100, dev43/231; không truncation.
- Tests local: **27 passed, 2 skipped**; hai test checkpoint bỏ qua vì thiếu torch/transformers. Tests xác nhận bộ tạo notebook giữ nguyên loss/optimizer/early stopping và export đúng metadata; không thay kiểm chứng training bằng weights thật.

| Bằng chứng local | Vị trí |
|---|---|
| Audit seed42 cũ | `results/nn/week4_cuong_audit/` |
| Metrics, chênh lệch và verification NN43/44 | `results/nn/week4_cuong_seed43/`, `results/nn/week4_cuong_seed44/` |
| Metrics và verification PV43 | `results/pv/week4_cuong_seed43/` |
| Bảng gộp và đối chiếu giữa run NN | `results/week4_cuong/combined_metrics.csv`, `verification_summary.json` cùng thư mục |
| Scripts tái lập và registry Kaggle | [handoff/week4_cuong](../../handoff/week4_cuong/README.md) |

CSV/JSON chi tiết và model ZIP giữ ngoài Git; report không chép câu nguồn. Ba URL saved version đã đủ, do người dùng cung cấp; chưa xác minh từ xa vì công cụ không truy cập được Kaggle.

## 5. Giới hạn và việc còn lại

NN có hai seed, PV một seed; không báo kết quả ba seed, ý nghĩa thống kê hoặc gộp hai task thành một điểm. Dev đã được dùng chẩn đoán và chọn thiết kế, nên kết quả là exploratory; chưa kiểm tra internal/official test. Quy tắc chênh mean >SD trong kế hoạch nhóm không phải kiểm định thống kê và chưa dùng để kết luận ở đây.

Chưa chạy inference/resume từ checkpoint và chưa có notebook đã execute để đối chiếu toàn bộ code; hash/CRC không thay các kiểm tra này. Đối chứng Spearman mới chưa đối chiếu với artifacts baseline seed43/44 của Đạt, nên chỉ diễn giải chênh giữa hai selector trong cùng run. Không đánh dấu thí nghiệm bias nhiều seed/PV đã hoàn thành: phần này được thay bằng hướng selector sau kết quả âm tuần 3. Việc nhóm chấp thuận đổi hướng chưa có biên bản trong artifacts được cung cấp.

## 6. Đề xuất cho G2

Giữ kết quả tuần 3 và audit tuần 4 như bằng chứng tái lập; **chưa áp dụng khởi tạo bias hoặc selector mean RMSE cho recipe chung**. Trên NN, giảm mean RMSE có thể che đánh đổi giữa hai output; báo riêng modifier/head trước mọi quyết định. Trên PV, không có khác biệt trong run đã thử.

Chốt hướng này ở phạm vi hiện có, trình bày kết quả âm và chưa tự mở thêm seed hoặc thay ngưỡng giữ/bỏ. Nếu nhóm cần kiểm tra tiếp, phải đặt câu hỏi/đối chứng mới trước và giữ test ngoài quyết định recipe.

GPU training theo metadata: **13.17 phút (0.220 giờ)** cho ba lần train; không tính mỗi selector thành một lần train, chưa gồm toàn bộ setup/export. Giờ làm tay: chưa ghi nhận.
