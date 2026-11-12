# Kết quả hiện tại — cập nhật 12/11/2026

[Trang chính](../README.md) · [Lộ trình](README.md) · [Cách chạy](RUN_GUIDE.md)

## Kết quả ba seed 42/43/44 — kiểm chứng 12/11

**Đã đủ 12 run NN/PV × B1/B2 × 3 seed.** Tám results ZIP mới đạt CRC, metadata/ID/gold/context đối chiếu đúng raw data và frozen manifest, metrics tính lại khớp trong 1e-12. Hai notebook đã chạy kết thúc không lỗi ngày 11/11 trên Tesla T4, code chỉ đổi đường dẫn Input so với bản gộp đã chuẩn bị. Raw data, alignment, split seed42, BERT revision và hyperparameters được giữ nguyên.

Spearman là mean ± sample SD (`ddof=1`), các sai số là mean qua ba run, trên **dev**, không phải điểm test.

| Task/output | Model | Spearman ↑ | RMSE ↓ | MAE ↓ | Delta MAE ↓ |
|---|---|---:|---:|---:|---:|
| NN ModAvg | B1 | 0,4460 ± 0,0146 | 0,9251 | 0,6936 | 0,5852 |
| NN ModAvg | B2 | 0,4676 ± 0,0364 | 0,9285 | 0,7008 | 0,6182 |
| NN HeadAvg | B1 | 0,5571 ± 0,0132 | 0,7831 | 0,5708 | 0,5461 |
| NN HeadAvg | B2 | 0,5816 ± 0,0256 | 0,7564 | 0,5765 | 0,5833 |
| PV Avg | B1 | 0,3632 ± 0,0212 | 1,0299 | 0,8348 | 0,7878 |
| PV Avg | B2 | 0,4972 ± 0,0229 | 0,9156 | 0,7426 | 0,7541 |

- **PV:** B2 tăng Spearman, giảm RMSE/MAE/Delta MAE ở cả ba seed; bằng chứng dev thuận lợi hơn NN, chưa chứng minh chất lượng trên test.
- **NN ModAvg:** chênh Spearman B2−B1 ở 42/43/44 là +0,0561/−0,0305/+0,0392; mean +0,0216, sample SD chênh lệch 0,0459. Có lần đổi dấu nên chưa xem mức tăng seed42 là lợi ích ổn định.
- **NN HeadAvg:** Spearman tốt hơn ở cả ba run; nhưng Delta MAE của B2 vẫn tệ hơn B1 ở cả hai output, cả ba seed. Không kết luận B2 tốt hơn toàn diện.

**Giới hạn môi trường:** seed42 dùng Transformers 5.16.1, tokenizers 0.23.1, torch 2.11.0+cu128, Python 3.13.15; seed43/44 dùng 5.20.2, 0.23.3, 2.11.0+cu130, Python 3.13.16. B1/B2 trong cùng seed khớp môi trường, nhưng SD ba seed là **dao động giữa các run**, không đo riêng nhiễu seed. Muốn tách yếu tố này cần thêm NN seed42 trong môi trường mới đã khóa; giữ nguyên pilot cũ.

**Checkpoint:** bốn complete ZIP seed42 còn đủ và CRC đạt; PV B1 seed42 đã đối chiếu 16 dev predictions trên CPU. Tám run mới đã được bàn giao qua hai ZIP gộp `NN_all_models_results.zip` và `PV_all_models_results.zip`, giữ best.pt/tokenizer/config và dev results, chủ động bỏ resume.pt. Bản lưu local: `results/<task>/models_seeds43_44.zip`, mỗi run có `MODEL_ARCHIVE.json` chỉ đúng member và hash. Kết quả kiểm tra CRC/metadata/predictions và inference 16 dòng của từng model được ghi tại `results/week3_dat/model_bundle_verification.json`. Không cần bản resume để viết report hay dự đoán; không tuyên bố có thể tiếp tục train từ optimizer/RNG của tám run mới. Còn thiếu link saved version; đã nhận bốn manifest NN, size/hash results khớp.

Đọc [report của Đạt](../reports/week3/Dat.md). Bảng chi tiết ở `results/week3_dat/metrics.csv`, `seed_summary.csv`, `paired_seed_changes.csv`, `paired_seed_summary.csv`, `runs.csv`, `audit.json`. Notebook có outputs được giữ tại `results/nn/executed_seeds43_44.ipynb` và bản PV tương ứng. Predictions mới nằm đúng `results/<task>/<model>_seed<seed>/dev/`, results ZIP local mở được trong Results explorer. Phần dưới là hồ sơ pilot **seed42 ngày 27/10**, không thay thế bảng ba seed bên trên.

## Hồ sơ pilot seed42 — 27/10/2026

**Đã đủ NN/PV × B1/B2 seed42 cùng bản lưu đầy đủ checkpoint.** Mở trang này để đọc kết quả; chỉ vào `results/` khi cần predictions, ZIP hoặc model. Baseline B0 cũng có cho cả NN/PV. Tiếp theo là phân tích lỗi và chốt recipe.

## NN — hai đầu ra modifier/head

B1 đọc expression; B2 đọc thêm context. Cả hai dùng cùng frozen split, config, backbone revision và 504 dev rows thuộc 67 expressions. Checkpoint chọn bằng dev Spearman; bảng này là dev, chưa phải internal test hay điểm cuộc thi.

| Đầu ra | B1 Spearman ↑ | B2 Spearman ↑ | B1 RMSE ↓ | B2 RMSE ↓ |
|---|---:|---:|---:|---:|
| Modifier (`ModAvg`) | 0,4464 | 0,5025 | 0,8993 | 0,9743 |
| Head (`HeadAvg`) | 0,5488 | 0,5520 | 0,7943 | 0,8079 |

B2 xếp thứ tự tốt hơn nhưng sai số điểm lớn hơn. Một seed trên dev chưa đủ kết luận context luôn có ích. B2 fallback về expression-only ở 14/504 rows (2,78%); không truncation. Epoch tốt nhất: B1 là 7/8, B2 là 3/5 epochs đã chạy.

Predictions và metrics trong gói tải ngày 27/10 **trùng từng byte với bộ NN đã lưu trước đó**. Metadata/history có thay đổi; không tính đây là một seed hay một kết quả độc lập mới. Điểm mới là đã giữ được `complete.zip` của cả hai model, có weights, tokenizer và trạng thái resume.

Paired bootstrap theo expression, 2.000 lần, seed42: CI95 của macro MAE gain `MAE(B1) − MAE(B2)` chứa 0 ở cả hai đầu ra. Đây là chẩn đoán dev; không phải CI cho chênh lệch Spearman. Xem định nghĩa tại [Metrics](METRICS.md).

## PV — một đầu ra Avg

PV B1/B2 dùng cùng raw data và frozen manifest khớp bản local: train/dev/test **1.100/231/236 rows**. Dev có **24 expressions**; cùng config, packages, backbone revision và ID/gold/context. Cả hai notebook đã chạy có code khớp bản PV trong repo. Chỉ có dev predictions; test chưa đánh giá.

| Model | Spearman ↑ | MAE ↓ | RMSE ↓ | Delta MAE macro ↓ | Variance corr. ↑ |
|---|---:|---:|---:|---:|---:|
| B0 | NA | 0,9039 | 1,0849 | 0,7878 | NA |
| B1 | 0,3870 | 0,8264 | 1,0234 | 0,7878 | NA |
| B2 | 0,4842 | 0,7458 | 0,9182 | 0,7343 | 0,4087 |

B0 có prediction hằng nên correlation undefined, không phải 0. B2 cải thiện Spearman **+0,0971**, giảm MAE **0,0807** và RMSE **0,1052** so với B1 trên dev. B1 chọn epoch 1/3 epochs đã chạy; B2 chọn epoch 2/4. Checkpoint chọn theo Spearman; dừng sớm phù hợp patience2, không phải lỗi training.

B2 fallback về expression-only ở **43/231 dev rows (18,61%)**, train là 161/1.100 (14,64%); không truncation. Chỉ xét cùng 188 dev rows không fallback cho cả hai model vẫn có B2 Spearman cao hơn (0,4960 so với 0,3281) và RMSE thấp hơn (0,9090 so với 1,0457). Đây là phân tích một subset, không thay kết quả toàn dev hoặc chứng minh alignment đúng ngữ nghĩa.

Paired target-cluster bootstrap (2.000 replicates, seed42), macro target MAE gain `MAE(B1) − MAE(B2)` = **0,0736**, CI95 **[−0,0466; 0,1910]**. CI chứa 0; chưa đủ bằng chứng kết luận cải thiện ổn định. Macro target MAE gain cân đều theo expression, khác mức giảm overall row MAE 0,0807 ở bảng trên. CI này không dùng cho Spearman. Không gộp điểm NN/PV thành một điểm chung.

## Tìm artifacts

```text
results/
  README.md                         Chỉ mục local
  metrics_summary.csv               B0/B1/B2 × NN/PV, từng output riêng
  nn/
    B0_seed42/dev/                  Baseline trung bình nhãn train
    B1_seed42/                      Run B1
      results.zip                   File nhỏ để mở trên web
      complete.zip                  Bản đầy đủ có best.pt, resume.pt, tokenizer
      executed.ipynb                Notebook đã chạy, giữ nguyên outputs
      dev/                          Metrics, predictions và errors đã giải nén
      IMPORT.json                   Hash và kiểm tra gói tải
      ...                           Config, provenance, history, inventory
    B2_seed42/                      Giống cấu trúc B1
    comparison_seed42/              Bảng so sánh và kiểm tra ghép cặp
  pv/
    B0_seed42/dev/                  Baseline PV
    B1_seed42/                      Cấu trúc giống NN B1
    B2_seed42/                      Cấu trúc giống NN B2
    comparison_seed42/              Metrics, paired errors và verification PV
```

| Bạn muốn làm gì? | Mở file nào? |
|---|---|
| Xem nhanh kết quả | Trang này |
| Xem toàn bộ B0/B1/B2 NN/PV | `results/metrics_summary.csv` (ô trống là metric undefined) |
| Xem bảng so sánh B1/B2 | `results/<task>/comparison_seed42/metrics_comparison.csv` |
| Đọc lỗi từng context | `results/<task>/comparison_seed42/paired_error_analysis.tsv` |
| Xem trên web | Chọn local `results/<task>/B1_seed42/results.zip` và bản B2 cùng task |
| Giữ model để inference/resume sau này | `results/<task>/{B1,B2}_seed42/complete.zip` |
| Kiểm tra nguồn, hash và phép so sánh | `results/<task>/comparison_seed42/verification.json` |

Weights chỉ giữ trong `complete.zip` để tránh thêm bản giải nén lớn. Khi cần inference/resume, giải nén **toàn bộ** ZIP vào thư mục riêng. Đã kiểm tra CRC toàn bộ ZIP, file bắt buộc, SHA256 bản sao, results nhúng và tính lại metrics khớp trong sai số 1e-12. Chưa chạy nạp weights/inference/resume; chưa có URL Kaggle saved version để đối chiếu.

## Quy tắc lưu từ lần sau

- Lưu theo `results/<task>/<model>_seed<seed>/`; cặp B1/B2 dùng `comparison_seed<seed>/`.
- Nếu chạy lại cùng seed với recipe khác, thêm hậu tố có ngày hoặc tên biến thể; giữ run cũ.
- `notebooks/` chứa code để sửa/chạy lần sau. `executed.ipynb` trong results là bằng chứng của run, không thay notebook nguồn.
- Results cũ trước khi sắp xếp nằm tại `archive/results-before-update-20261027/`. B0 NN/PV được chuyển sang cấu trúc mới, giữ nguyên files.
- Downloads được giữ nguyên. Raw data, outputs notebook, predictions và checkpoint tiếp tục nằm ngoài Git.

Việc tiếp theo của pilot seed42 được mở rộng ở phần cập nhật ba seed đầu trang. Xem [bàn giao tuần 3](../handoff/week3_dat/README.md) để đối chiếu trạng thái hiện tại.
