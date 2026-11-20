# Report tuần 3: Cương — khởi tạo bias theo trung bình train

**Kết luận tạm thời (seed42, chỉ dev): không giữ biến thể bias theo tiêu chí đã đặt.** B2 giảm RMSE ở cả hai output nhưng Spearman head giảm 0,06624, vượt mức giảm cho phép 0,02. Không kết luận bias luôn có hại từ một seed.

## 1. Đã chạy gì

NN B1/B2 bias, train 2.479 dòng; dev 504 dòng thuộc 67 expressions, frozen split42. Baseline là hai ZIP gốc, đã đối chiếu SHA256 với hồ sơ chẩn đoán. Chưa đánh giá test.

| Run | Seed | GPU theo metadata | Phút training | Best epoch / epochs chạy | Saved version |
|---|---|---|---:|---|---|
| B1 bias | 42 | Tesla T4 | 10.85 | 8 / 8 | [368869513](https://www.kaggle.com/code/azoncx/nn-b1-bias/output?scriptVersionId=368869513) |
| B2 bias | 42 | Tesla T4 | 7.30 | 3 / 5 | [368869805](https://www.kaggle.com/code/azoncx/nn-b2-bias/notebook?scriptVersionId=368869805) |

## 2. Kết quả so với baseline

| Model | Biến thể | Output | Spearman ↑ | MAE ↓ | RMSE ↓ | Delta MAE ↓ | Var. corr. ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| B1 | baseline | ModAvg | 0.44639 | 0.67594 | 0.89930 | 0.58523 | NA |
| B1 | baseline | HeadAvg | 0.54878 | 0.58231 | 0.79434 | 0.54612 | NA |
| B1 | bias | ModAvg | 0.46652 | 0.64185 | 0.86021 | 0.58523 | NA |
| B1 | bias | HeadAvg | 0.52783 | 0.59079 | 0.78456 | 0.54612 | NA |
| B2 | baseline | ModAvg | 0.50253 | 0.72106 | 0.97434 | 0.60155 | 0.22504 |
| B2 | baseline | HeadAvg | 0.55198 | 0.60198 | 0.80789 | 0.60063 | 0.18082 |
| B2 | bias | ModAvg | 0.48470 | 0.65832 | 0.86153 | 0.60513 | 0.17950 |
| B2 | bias | HeadAvg | 0.48575 | 0.60756 | 0.79942 | 0.59262 | 0.08939 |

B1 có prediction hằng theo expression nên variance correlation không xác định (NA), không phải 0.

| Điều kiện giữ B2 bias | Quan sát | Đạt? |
|---|---|---|
| RMSE giảm ở cả hai output | Modifier −0,11281; head −0,00847 | Có |
| RMSE modifier ≤0,938 | 0,86153 | Có |
| Spearman mỗi output giảm không quá 0,02 | Modifier −0,01783; head −0,06624 | **Không** |

## 3. Thiết kế và kiểm chứng

Chẩn đoán trước training: affine fit trên dev giảm RMSE B2 modifier 0,18273 và head 0,04189, đều >0,02; vì vậy chọn thử bias. Đây chỉ là chẩn đoán trên chính dev, không phải kết quả model mới.

Khởi tạo `head[1].bias` bằng trung bình nhãn train hữu hạn của từng output; bỏ qua khi resume. Giữ lr=2e-5, backbone/revision, split và training budget. Theo metadata, cấu hình chỉ thêm `init_train_mean_bias` và `cuong_decision`.

- Hash/size và CRC của các ZIP đạt; results và predictions nhúng trong complete ZIP trùng bản riêng. Có `best.pt`, `resume.pt`, tokenizer và encoder config.
- Metrics tính lại khớp bản lưu với sai số <1e-12; manifests, train/dev IDs và gold/context khớp cả bốn run. Không có test exports.
- Bốn bundle ghi cùng môi trường: Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, tokenizers 0.23.3, numpy 2.1.3, pandas 2.3.3, scipy 1.16.3; Tesla T4.
- B2 fallback 68/2.479 train, 14/504 dev; không truncation theo metadata.

## 4. Giới hạn và bàn giao

Một seed trên dev: RMSE giảm chưa đi kèm thứ hạng tốt hơn. Giữ baseline cho so sánh chính. Chưa nạp checkpoint để inference/resume; chưa có notebook đã execute để đối chiếu code. Hai link Kaggle do người dùng cung cấp, chưa xác minh trực tiếp với ZIP local.

Chi tiết kiểm chứng: `results/nn/cuong_bias_comparison/`. Hướng dẫn tái lập ở [README](README.md); URL phiên bản ở [SAVED_VERSIONS.json](SAVED_VERSIONS.json).

Cập nhật 20/11/2026.
