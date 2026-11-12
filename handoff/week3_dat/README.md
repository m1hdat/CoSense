# Phần việc tuần 3 của Đạt

Kế hoạch nguồn: branch `plans`, commit `36c7fb0`, mục 4.1 của `plan/week3.md`.

**Cập nhật 12/11: đã đủ kết quả 12 run NN/PV × B1/B2 × seeds42/43/44.** Đã nhập tám results ZIP mới và giữ nguyên hai notebook đã chạy trong `results/<task>/executed_seeds43_44.ipynb`. Không cần chạy lại hai notebook gộp để phân tích kết quả hiện có. [Report của Đạt](../../reports/week3/Dat.md) và `results/week3_dat/` đã có bảng ba seed.

Seed43/44 dùng môi trường mới hơn seed42; SD hiện tại mô tả dao động giữa run, không tách riêng ảnh hưởng seed. **Đã nhận đủ tám model mới** qua NN/PV_all_models_results.zip, giữ bản local ở `results/<task>/models_seeds43_44.zip`; không cần tải lại full complete ZIP để viết report/inference. Các gói mới chủ động bỏ resume.pt. Còn cần ghi hai link saved version. Nếu muốn đo riêng nhiễu seed, cần thêm NN seed42 trong đúng môi trường mới đã khóa và giữ kết quả pilot cũ riêng.

Đã nhận thêm bốn manifest NN qua chat, lưu tại `results/nn/<model>_seed<seed>/ARTIFACTS.json`; size/SHA256 results khớp cả bốn. Bản ZIP gộp có inventory riêng thay cho yêu cầu tải thêm full complete ZIP/PV manifest để làm báo cáo. Chi tiết kiểm tra model ở `results/week3_dat/model_bundle_verification.json`. Giữ nhận định rõ: đủ để inference, không có trạng thái optimizer/RNG để tiếp tục train.

Đã chuẩn bị tùy chọn [NN B1/B2 seed42 đối chiếu môi trường](followup/NN_B1_B2_seed42_envmatch_kaggle.ipynb): **hai run**, kiểm tra versions/Python/GPU khớp seed43/44 trước khi train. Import thành notebook Kaggle mới; nếu guard báo khác môi trường thì dừng để xử lý, không bỏ qua guard. Lưu artifacts vào `B1_seed42_envmatch`/`B2_seed42_envmatch`, giữ pilot cũ. Bản này mới kiểm tra local, chưa có kết quả GPU; không cần chạy lại tám run 43/44.

## Hai notebook để chạy trên Kaggle

| Notebook | Nội dung |
|---|---|
| [NN_B1_B2_seeds43_44_kaggle.ipynb](NN_B1_B2_seeds43_44_kaggle.ipynb) | Bốn run NN bắt buộc: B1/B2 × seed 43/44 |
| [PV_B1_B2_seeds43_44_kaggle.ipynb](PV_B1_B2_seeds43_44_kaggle.ipynb) | Bốn run PV bổ sung theo yêu cầu: B1/B2 × seed 43/44 |

Import **`NN_B1_B2_seeds43_44_kaggle.ipynb`** lên Kaggle, Add Input `cosense-kaggle-data`, bật GPU + Internet rồi Save Version → Save & Run All. Bản gộp chạy lần lượt B1 seed43 → B2 seed43 → B1 seed44 → B2 seed44. Mỗi lần train khởi tạo model mới, không resume; giữ nguyên helper training/data/metrics từ notebook cũ và split seed42. Có giải phóng GPU giữa các run. Mỗi run xuất bộ complete/results ZIP và ARTIFACTS riêng; không gộp weights hay predictions giữa các model/seed.

Hai bản gộp đã có bằng chứng chạy hết trên Kaggle/Tesla T4 ngày 11/11, cả hai tới `ALL_FOUR_RUNS_READY`, code chỉ khác đường dẫn DATA_ROOT so với bản chuẩn bị. Bảng ba seed đã ghép với seed42. Bốn notebook seed riêng và ZIP cũ đã được bỏ để tránh chạy trùng. Các notebook nguồn trong `notebooks/` vẫn phục vụ tái tạo bản gộp; notebook có outputs trong `results/` là bằng chứng chạy.

## Cách chạy và lưu kết quả

Bổ sung theo yêu cầu: **`PV_B1_B2_seeds43_44_kaggle.ipynb`** chạy PV B1/B2 × seed 43/44, dùng cùng Input và cách chạy như bản NN. Kết quả PV có một output `Avg`, prefix file `PV_` riêng. Đây là bốn run bổ sung để đo độ ổn định PV; seed42 đã có và không train lại. Chờ `ALL_FOUR_RUNS_READY`, tải đủ bốn bộ ZIP/ARTIFACTS; tổng hợp với PV seed42 riêng, không gộp điểm NN/PV.

1. Import notebook gộp NN hoặc PV từ bảng trên vào Kaggle. Mỗi notebook chạy bốn run của một task.
2. Add Input dataset `cosense-kaggle-data` đang dùng cho seed 42; không thêm model ZIP hay results ZIP. Dataset phải chứa frozen manifests NN/PV hoặc ZIP dữ liệu gốc.
3. Bật GPU và Internet. Run All hoặc Save Version → Save & Run All. Các notebook đã chọn sẵn task NN/PV tương ứng, full B1/B2, training seed 43/44, **split seed 42**, `EVALUATE_LOCAL_TEST=False`, không resume.
4. Kiểm tra `QUEUE` và các dòng `TRAIN: NN_...` hoặc `TRAIN: PV_...` đúng task/model/seed. Chỉ training seed thay đổi; giữ nguyên recipe, alignment và dữ liệu. Nếu package versions khác seed 42, ghi lại trong report, chưa quy mọi thay đổi cho seed.
5. Sau `OUTPUTS_READY`, mở **Output của đúng saved version**, tải `<RUN>_complete.zip`, `<RUN>_results.zip`, các CSV và `<RUN>_ARTIFACTS.json` được notebook xuất ra. Giữ tất cả file được inventory xác nhận; ưu tiên complete ZIP có weights thật.
6. Ghi URL saved version, GPU, thời gian. Giữ nguyên ZIP tải về. Lưu riêng vào `results/<nn hoặc pv>/<B1 hoặc B2>_seed<43 hoặc 44>/`; đặt bản sao hai ZIP thành `complete.zip` và `results.zip`. Giải nén **complete ZIP** ngay vào thư mục run tương ứng để có `dev/predictions.tsv`, config/metadata và manifest. Results ZIP nhỏ chứa `predictions.tsv` ở gốc, không cần giải nén thêm. Không ghi đè seed 42.

## Kiểm tra và tổng hợp local (PowerShell tại thư mục repo)

```powershell
.\.venv\Scripts\python.exe handoff/week3_dat/verify_pv_checkpoint.py
.\.venv\Scripts\python.exe handoff/week3_dat/import_results.py
.\.venv\Scripts\python.exe handoff/week3_dat/summarize.py
```

Lệnh đầu nạp **model PV B1 seed42 thật** và đối chiếu 16 dev predictions, không train. Lệnh import chỉ đọc tám results ZIP và hai notebook từ Downloads, kiểm tra code/CRC/provenance, giữ nguyên nguồn; chạy lại không ghi đè file khác nội dung. Lệnh cuối kiểm tra 12 run, tính lại metrics và bảng ba seed, phân biệt results với checkpoint. SD dùng sample SD (`ddof=1`), chỉ báo khi đủ ba seed. Bảng chi tiết có cờ môi trường; không coi SD hiện tại là ảnh hưởng seed thuần.

Report theo cấu trúc nhóm ở [reports/week3/Dat.md](../../reports/week3/Dat.md); script cũng giữ bản local `week3_Dat.md` để tiện đối chiếu. Report ghi rõ còn thiếu link saved version và giới hạn môi trường; không đưa nguyên câu dữ liệu vào report. Kết quả chi tiết và kiểm chứng nằm local ở `results/week3_dat/`.
