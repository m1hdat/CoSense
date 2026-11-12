# Report tuần 3: Đạt · PV và dao động giữa các run

**Kết luận:** PV B2 tốt hơn B1 cả thứ hạng lẫn sai số ở ba seed; NN ModAvg chưa cho thấy lợi ích B2 ổn định. Thống kê ba seed là mô tả vì môi trường seed42 khác seed43/44.

## 1. Đã chạy gì
Đủ 12 run NN/PV × B1/B2 × seeds42/43/44, Tesla T4. Tám run mới hoàn tất 11/11, không lỗi; seed42 dùng lại pilot 27/10. GPU/phút từng run: `results/week3_dat/runs.csv`; thiếu link saved version.

## 2. Kết quả dev (42/43/44)
Spearman: mean ± sample SD (`ddof=1`); sai số: mean ba run. NN: 504 dòng/67 cụm; PV: 231 dòng/24 cụm. Cùng raw data, frozen split, recipe.

| Task/output | Model | Spearman ↑ | RMSE ↓ | MAE ↓ | Delta MAE ↓ |
|---|---|---:|---:|---:|---:|
| NN ModAvg | B1 | 0.4460 ± 0.0146 | 0.9251 | 0.6936 | 0.5852 |
| NN ModAvg | B2 | 0.4676 ± 0.0364 | 0.9285 | 0.7008 | 0.6182 |
| NN HeadAvg | B1 | 0.5571 ± 0.0132 | 0.7831 | 0.5708 | 0.5461 |
| NN HeadAvg | B2 | 0.5816 ± 0.0256 | 0.7564 | 0.5765 | 0.5833 |
| PV Avg | B1 | 0.3632 ± 0.0212 | 1.0299 | 0.8348 | 0.7878 |
| PV Avg | B2 | 0.4972 ± 0.0229 | 0.9156 | 0.7426 | 0.7541 |

**PV:** B2 tăng Spearman cả ba seed (+0,0971/+0,1774/+0,1275), giảm RMSE/MAE/Delta MAE; khác NN seed42. Chưa đánh giá test.

**NN:** B2−B1 ModAvg: +0,0561/−0,0305/+0,0392; mean +0,0216, SD chênh lệch 0,0459. SD Spearman B1/B2: 0,0146/0,0364. Mức tăng seed42 ≈0,057 chưa ổn định, có lần đổi dấu. HeadAvg tăng Spearman cả ba run; Delta MAE B2 tệ hơn B1 ở cả hai output, cả ba seed.

## 3. Đã đổi gì
Giữ split seed42, alignment, BERT revision, hyperparameters; đổi training seed và mount. Image thực tế đổi Transformers 5.16.1→5.20.2, tokenizers 0.23.1→0.23.3, torch cu128→cu130, Python 3.13.15→3.13.16. SD trên không đo riêng nhiễu seed.

## 4. Kiểm chứng
Tám results ZIP đạt CRC; ID/gold/context khớp raw data/manifest; metrics tính lại khớp ≤1e-12. Đã nhận model của cả tám run mới trong hai ZIP gộp: CRC/metadata/predictions khớp; nạp từng model thật, dự đoán lại 16 dòng dev đạt ngưỡng 1e-5. PV B1 seed42 cũng đã đạt kiểm tra này. B2 fallback NN 14/504, PV 43/231; không truncation.

## 5. Còn thiếu khi bàn giao
Cần bổ sung hai link saved version; bốn manifest NN đã khớp hash. Model mới đủ cho inference/báo cáo; đã chủ động bỏ resume.pt nên không có optimizer/RNG để tiếp tục train. Smoke PV cũ chưa có bằng chứng riêng.

## 6. Tiếp theo
Nộp report với giới hạn môi trường đã ghi; không cần train thêm để báo cáo kết quả hiện có. Nếu cần đo riêng nhiễu seed, bổ sung NN seed42 dưới đúng môi trường seed43/44, giữ pilot cũ.

Giờ làm tay: chưa ghi nhận · GPU training: 1.048 giờ (tám run mới: 0.684 giờ), không gồm setup/export.
