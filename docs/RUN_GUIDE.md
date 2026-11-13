# Cách chạy: Kaggle → kết quả → web → bài nộp

[Lộ trình](README.md) · [Metrics và lý do chọn](METRICS.md)

## 1. Chỉ cần nhớ các file này

| File/thư mục | Dùng lúc nào? |
|---|---|
| `notebooks/B1_kaggle.ipynb`, `B2_kaggle.ipynb` | Sửa code training trực tiếp, import lên Kaggle |
| `data/cosense-kaggle-data.zip` | Bundle English train + frozen manifests để đưa lên Kaggle |
| `data/bundle/split_manifest_{nn,pv}.json` | Danh sách ID train/dev/internal-test cố định |
| `data/trial/` | Trial English chỉ dùng kiểm tra parser/inference/submission, không train |
| `scripts/prepare_data_bundle.py` | Audit train/trial và tái tạo bundle theo seed 42 |
| `scripts/audit_pv_preprocessing.py` | Audit alignment/fallback/truncation PV trong WSL |
| `scripts/run_pv_smoke_wsl.py` | CPU smoke train PV B1/B2, 16 rows mỗi split |
| `results/nn_B1_B2_seed42/` | Hai results ZIP NN pilot và phân tích đã có |
| `results/baselines/` | Baseline B0 NN/PV đã có |
| `docs/audit/` | Audit cũ; chỉ mở khi cần tìm vấn đề dữ liệu |

Local không chạy full training; WSL chỉ dùng cho audit và smoke test nhỏ. Không cần script build notebook. Khi sửa preprocessing/model/metrics chung, sửa cả hai notebook tương ứng. Không đưa raw data/results chứa context lên Git; quyền phân phối dữ liệu chưa được xác nhận.

Khi dữ liệu nguồn thay đổi hoặc cần kiểm tra lại bundle, chạy:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data_bundle.py
```

Script giữ nguyên bytes của TSV train, kiểm tra số hàng/target/schema/ID, xác nhận train và trial không trùng ID/target, rồi sinh manifests, `dataset_info.json` và ZIP sáu file. Không sửa manifest bằng tay.

Trong WSL, sau khi kích hoạt environment có PyTorch, Transformers, NumPy, pandas và SciPy:

```bash
python scripts/audit_pv_preprocessing.py
python scripts/run_pv_smoke_wsl.py B1
python scripts/run_pv_smoke_wsl.py B2
```

Audit chạy toàn bộ frozen split nhưng không đọc gold để chọn model. Hai smoke run chỉ dùng 16 train + 16 dev rows, một epoch và CPU; kết quả PASS xác minh pipeline chứ không phải score nghiên cứu. Full PV vẫn chạy trên Kaggle với `SMOKE=False` và cấu hình khóa bên dưới.

## 2. Train trên Kaggle

1. Import notebook B1; Add Input Dataset từ data ZIP đã có. ZIP có raw TSV và frozen manifests, không cần Dataset source code hay clone repo.
2. Bật GPU CUDA và Internet để tải pretrained BERT; giữ PyTorch CUDA của image. Notebook tự xử lý dependencies thiếu, không dùng `requirement.txt` local để thay môi trường Kaggle.
3. Chọn cấu hình dưới đây rồi chạy từ trên xuống. B2 dùng session riêng, cùng Dataset/config/seed; không đọc checkpoint B1.

```python
TASK = 'pv'                 # 'nn' cho noun compounds
SEED = 42                   # Ngẫu nhiên của lần train
SPLIT_SEED = 42             # Giữ split đã khóa
SMOKE = False               # True chỉ kiểm tra 16 rows/1 epoch
EVALUATE_LOCAL_TEST = False # Giữ False lúc thử recipe trên dev
```

Đổi `TASK` không đổi `MODEL`: chọn B1/B2 bằng đúng notebook. Khi đổi settings phải chạy lại cell cấu hình và các cell sau đó. Thêm seeds 43/44 thì chỉ đổi training seed, không chia lại dữ liệu. `42` không phải số giúp model tốt hơn; nó giúp kiểm soát ngẫu nhiên.

Default hiện tại: BERT-base-uncased pin revision `86b5e0934494bd15c9632b12f734a8a67f723594`, LR 2e-5, batch 8 × accumulation 2, tối đa 8 epochs, patience 2, max length 128. Checkpoint chọn bằng dev Spearman; early stop không phải crash. Train loss giảm không bảo đảm dev/test tốt hơn.

## 3. Chạy xong phải giữ gì?

Notebook hiện trong repo xuất các file ở gốc `/kaggle/working`, ví dụ NN B1 seed42:

```text
NN_B1_seed42_full_results.zip     Metrics/predictions/history → mở bằng web
NN_B1_seed42_full_model.zip       Best model/tokenizer/config → inference
NN_B1_seed42_full_metrics.csv     Bảng điểm dễ đọc
NN_B1_seed42_full_history.csv     Diễn biến theo epoch
NN_B1_seed42_full_ARTIFACTS.json  Danh sách file và kích thước
```

Đợi `OUTPUTS_READY`, xem bảng metrics/output inventory rồi lưu version kèm files. Mở Output của **đúng saved version** để kiểm tra files thực sự tồn tại; thông báo trong session chưa chứng minh saved version đã giữ files.

`resume.pt` còn ở `cosense_outputs/runs/{task}/{model}_seed{seed}/`, không nằm trong model ZIP; muốn resume phải giữ cả resume checkpoint và siblings best/tokenizer/encoder config, cùng config/schedule. Tăng epochs không phải resume nguyên schedule cũ.

**Hai run NN cũ của bạn dùng exporter trước đó:** results ZIP chỉ khoảng 139/158 KB, không chứa weights. Không cần retrain để xem/đóng gói metrics. Nếu checkpoint còn trên Kaggle, tải về; chỉ cần train lại cho inference khi không còn checkpoint ở đâu. Bản exporter mới trong repo chưa được xác minh trên một saved version GPU của người dùng.

Tải kết quả về các folder riêng để không ghi đè, ví dụ:

```text
results/pv/B1_seed42/   results ZIP + model ZIP + CSVs + ARTIFACTS
results/pv/B2_seed42/   tương tự
```

Ghi cùng mỗi run: task/model/seed, Dataset version/source hash/manifest hash, config, notebook version, best epoch, dev metrics và đường dẫn checkpoint. Kiểm tra identity/config/coverage trước khi so B1/B2.

## 4. Mở web local

PowerShell tại repo, lần đầu:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
```

Những lần sau chỉ cần:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Mở http://localhost:8501. Data explorer đọc TSV; Results explorer upload hai **results ZIP**, không upload model ZIP. Web hiện xem metrics và so predictions theo expression; chưa load checkpoint để chấm câu mới. Chạy checks local bằng `.\.venv\Scripts\python.exe -m pytest -q`.

## 5. Chuẩn bị official submission — còn phải làm

| Bước | Điều kiện hoàn thành |
|---|---|
| Xác nhận rules | Đã xác nhận metric, filenames, không header, số cột và ZIP layout; submission quota, Terms và quyền dùng model/data ngoài vẫn cần xác nhận |
| Inference từ checkpoint | Đọc official input không có gold, tokenizer/preprocessing giống train, tạo NN hai điểm/PV một điểm |
| Kiểm tra predictions | Đủ ID, không trùng, đúng thứ tự/cột, toàn số hữu hạn; dùng official sample/checker khi có |
| Nộp CodaBench | Nền tảng chấp nhận đúng version; giữ receipt/score và bản ZIP chính xác |

`src/compartment/artifacts.py` đã được đối chiếu với format CodaBench ngày 12/11/2026: `en-nn-pred.tsv` gồm ID + modifier + head, `en-pv-pred.tsv` gồm ID + một prediction, không header, nằm ở root của một ZIP. Helper vẫn chưa phải pipeline inference hoàn chỉnh. Không nộp `results_bundle.zip` của web thay official submission.

Ngày 12/11/2026, [CodaBench 17971](https://www.codabench.org/competitions/17971/) xác nhận metric chính là Spearman, kèm context-variance Spearman và RMSE; Practice kết thúc 09/01/2027, Evaluation diễn ra 10–31/01/2027 và Post-evaluation bắt đầu 01/02/2027. Leaderboard Practice chưa hoạt động nhưng scorer vẫn tạo `scores.txt`. Terms đang được cập nhật và trang chưa công bố submission quota; không suy đoán các phần này từ task khác. Giữ internal test để đánh giá nghiên cứu; official test không dùng tune.

## 6. Tra cứu dữ liệu khi cần

| Task | Rows / expressions | Train / dev / internal-test rows | Đầu ra |
|---|---|---|---|
| NN train | 3.480 / 444 | 2.483 / 505 / 492 | `ModAvg`, `HeadAvg` |
| PV train | 1.557 / 158 | 1.093 / 236 / 228 | `Avg` |
| NN trial | 10 / 2 | Không chia; không train | `ModAvg`, `HeadAvg` |
| PV trial | 10 / 2 | Không chia; không train | `Avg` |

Split theo expression, gần 70/15/15 số targets, seed42; cùng expression không xuất hiện ở cả train và dev/test. Exact duplicate contexts được giữ cùng partition. Near-duplicate audit không chứng minh không còn paraphrases. Không đổi membership vì đổi training seed.

Data ZIP có sáu file: `en-nn-train.tsv`, `en-pv-train.tsv`, `split_manifest_nn.json`, `split_manifest_pv.json`, `dataset_info.json`, `README.md`. Trial không nằm trong ZIP. Notebook kiểm tra SHA256/ID coverage và dùng nguyên frozen IDs. Raw TSV trong ZIP giữ nguyên bytes so với `data/train/`.

Input features chỉ từ expression/context; không đưa gold, annotation Std hay ContextID vào encoder. ID đọc dạng string. Với PV, verb/particle có thể biến hình hoặc đứng cách nhau. Alignment hiện là heuristic; ambiguous/missing dùng expression fallback và giữ rows, không tự xem đó là context prediction thành công. Báo coverage/fallback/truncation theo split.

Results ZIP chứa `run_metadata.json`, `metrics.json`, `predictions.tsv`, `target_statistics.tsv`, `error_analysis.tsv`, `error_groups.tsv`, `training_history.csv`, `split_manifest.json`. Đó là kết quả đánh giá có gold; model ZIP phục vụ inference có vai trò khác.
