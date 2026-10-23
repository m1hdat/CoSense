# Cách chạy: Kaggle → kết quả → web → bài nộp

[Lộ trình](README.md) · [Metrics và lý do chọn](METRICS.md)

## 1. Chỉ cần nhớ các file này

| File/thư mục | Dùng lúc nào? |
|---|---|
| `notebooks/B1_kaggle.ipynb`, `B2_kaggle.ipynb` | Sửa code training trực tiếp, import lên Kaggle |
| `data/cosense-kaggle-data.zip` | Dataset dữ liệu đã chuẩn bị, không cần tạo lại |
| `data/splits/{nn,pv}/split_manifest.json` | Danh sách ID train/dev/test cố định |
| `results/nn_B1_B2_seed42/` | Hai results ZIP NN pilot và phân tích đã có |
| `results/baselines/` | Baseline B0 NN/PV đã có |
| `docs/audit/` | Audit cũ; chỉ mở khi cần tìm vấn đề dữ liệu |

Local không train; không cần script build notebook. Khi sửa preprocessing/model/metrics chung, sửa cả hai notebook tương ứng. Không đưa raw data/results chứa context lên Git; quyền phân phối dữ liệu chưa được xác nhận.

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
| Xác nhận rules | Đúng track/language, metric/scorer, filenames, header, ID/order, ZIP layout, submission limits và quyền dùng data/model |
| Inference từ checkpoint | Đọc official input không có gold, tokenizer/preprocessing giống train, tạo NN hai điểm/PV một điểm |
| Kiểm tra predictions | Đủ ID, không trùng, đúng thứ tự/cột, toàn số hữu hạn; dùng official sample/checker khi có |
| Nộp CodaBench | Nền tảng chấp nhận đúng version; giữ receipt/score và bản ZIP chính xác |

`src/compartment/artifacts.py` có helper kiểm tra/xuất TSV theo giả định nội bộ: NN/PV filename và ID + 2/1 số. Helper này **chưa được đối chiếu official format** và chưa phải pipeline inference hoàn chỉnh. Không nộp `results_bundle.zip` của web thay official submission.

Ngày 23/10/2026: [trang Task 5](https://sites.google.com/view/semeval-compartment) dẫn format/rules sang [CodaBench 17971](https://www.codabench.org/competitions/17971/); lần kiểm tra này CodaBench timeout. Các chi tiết chưa xác minh phải được kiểm tra trước khi export/nộp; không suy đoán số lượt nộp hoặc áp dụng rules task khác. Giữ internal test để đánh giá nghiên cứu; official test không dùng tune.

## 6. Tra cứu dữ liệu khi cần

| Task | Rows / expressions | Train / dev / internal-test rows | Đầu ra |
|---|---|---|---|
| NN | 3.490 / 446 | 2.479 / 504 / 507 | `ModAvg`, `HeadAvg` |
| PV | 1.567 / 160 | 1.100 / 231 / 236 | `Avg` |

Split theo expression, gần 70/15/15 số targets, seed42; cùng expression không xuất hiện ở cả train và dev/test. Exact duplicate contexts được giữ cùng partition. Near-duplicate audit không chứng minh không còn paraphrases. Không đổi membership vì đổi training seed.

Data ZIP có sáu file: `en-nn-train.tsv`, `en-pv-train.tsv`, `split_manifest_nn.json`, `split_manifest_pv.json`, `dataset_info.json`, `README.md`. Notebook kiểm tra SHA256/ID coverage và dùng nguyên frozen IDs. Không cần tokenize/đóng gói lại local. Raw TSV giữ nguyên bytes.

Input features chỉ từ expression/context; không đưa gold, annotation Std hay ContextID vào encoder. ID đọc dạng string. Với PV, verb/particle có thể biến hình hoặc đứng cách nhau. Alignment hiện là heuristic; ambiguous/missing dùng expression fallback và giữ rows, không tự xem đó là context prediction thành công. Báo coverage/fallback/truncation theo split.

Results ZIP chứa `run_metadata.json`, `metrics.json`, `predictions.tsv`, `target_statistics.tsv`, `error_analysis.tsv`, `error_groups.tsv`, `training_history.csv`, `split_manifest.json`. Đó là kết quả đánh giá có gold; model ZIP phục vụ inference có vai trò khác.
