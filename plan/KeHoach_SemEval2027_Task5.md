# Kế hoạch nhóm SemEval-2027 Task 5 (ComPartMent)

Bản cập nhật 24/10/2026 · Gốc: `KeHoach_SemEval2027_Task5.docx` (14/10/2026, @Khang Huân) · Đã sửa cho khớp repo `CoSense` ở commit `90061d4`

## Đã đổi gì so với bản Word (đọc trước)

Bản Word viết ngày 14/10, trước khi repo được dọn lại ngày 23/10. Phần lớn nội dung giữ nguyên (13 tuần, 4 vai trò, 4 mốc, họp thứ Ba). Các chỗ dưới đây đã sửa cho khớp thực tế:

| Mục | Bản Word 14/10 | Bản này | Căn cứ trong repo |
|---|---|---|---|
| Phạm vi | EN + DE, noun compound + particle verb | **Chỉ English NN và PV.** Việc riêng cho tiếng Đức (tách từ ghép, chính tả lịch sử, XLM-R, gbert, prompt tiếng Đức) bỏ. Chỉ thêm lại nếu `task_facts.md` cho thấy bắt buộc | `README.md`, `docs/README.md` |
| Chia dữ liệu | 5 fold out-of-fold, `data/folds.csv` | **Split cố định** train/dev/internal-test, chia theo expression (≈70/15/15, seed 42), không chia lại. Tính ổn định bằng nhiều training seed 42/43/44 | `src/compartment/splits.py`, `docs/RUN_GUIDE.md` |
| Hệ thống chính | Embedding / fine-tune / LLM, rồi ensemble | **B0, B1, B2** là trục chính. Embedding (E) và LLM (L) là hướng phụ, giữ hay bỏ ở mốc G2. Ensemble chỉ làm nếu có ≥2 hệ thống đủ tốt | `docs/README.md` |
| Cách chạy | Script `systems/*/train.py`, `predict.py` | Notebook Kaggle độc lập `B1_kaggle.ipynb`, `B2_kaggle.ipynb`. Local không train, không có bước build notebook | `docs/RUN_GUIDE.md` |
| "Hợp đồng" dự đoán | CSV `id,pred` | Results ZIP (`predictions.tsv`, `metrics.json`...) để phân tích; TSV nộp bài (ID + 2 số cho NN, 1 số cho PV) qua `artifacts.py` | `src/compartment/artifacts.py` |
| Thước đo | Spearman, chọn phạm vi tính (`scope`) | Spearman (chọn checkpoint), MAE, RMSE, delta MAE, variance correlation. Official scorer **chưa xác nhận** | `docs/METRICS.md`, `src/compartment/metrics.py` |
| Repo | `khanghb2006/SemEval-2027` | `m1hdat/CoSense` (remote hiện tại của repo này) | `git remote -v` |
| Thư mục | `src/`, `systems/`, `preds/`, `reports/`, `paper/` | `notebooks/`, `app/`, `src/compartment/`, `tests/`, `docs/`, `literature-review/`, `plan/`; `data/`, `results/` nằm ngoài Git | `.gitignore`, `README.md` |
| Câu hỏi nghiên cứu | Chưa có | Thêm RQ1–RQ3 (context có giúp không, expression chưa gặp, context giúp/gây lỗi ở đâu) để mọi thí nghiệm bám theo | `docs/README.md` |

Con số nào là đề xuất của bản này (không có trong repo) đều ghi "(đề xuất)".

## Tổng quan

Nhóm làm trong 13 tuần, từ 14/10/2026 đến 15/01/2027. Mục tiêu là có hệ thống hoàn chỉnh, chạy lại được từ checkpoint, và nộp kết quả trên tập test trong khoảng 10–15/01/2027, sớm hơn hạn chót 31/01 hơn hai tuần. Lịch evaluation 10–31/01/2027 lấy từ trang Task 5 (kiểm tra 23/10); ngày phát hành test, số lần nộp và format còn phải xác nhận ở tuần 1.

Mỗi người giữ một vai trò cố định suốt dự án. Các hướng chạy độc lập, một người chậm không làm cả nhóm dừng.

| Thành viên | Vai trò | Sản phẩm chính |
|---|---|---|
| Khang | Trưởng nhóm: luật thi, Codabench, bảng thí nghiệm, công cụ phân tích, ensemble, nộp bài | `task_facts.md`, bảng thí nghiệm, công cụ so sánh hệ thống, pipeline nộp bài, file nộp cuối |
| Cương | Scorer nội bộ; hướng embedding (E0 không huấn luyện, E1 đặc trưng + mô hình nhỏ) | `metrics.py` khớp scorer chính thức, notebook E0/E1, bảng ablation |
| Đạt | Fine-tune B1/B2: chạy Kaggle, seeds, checkpoint, inference | Bốn pilot run, mười hai run cuối, notebook dự đoán |
| Phát | Ngôn ngữ học: định vị cụm từ, đọc lỗi, nhãn hiện tượng; LLM (nếu luật cho phép) | Báo cáo alignment, phân tích lỗi RQ3, nhãn hiện tượng, hướng L |

Giả định: cả nhóm biết Python cơ bản, dùng GPU Kaggle (hoặc Colab), mỗi người 6–8 giờ/tuần. Tuần 10–12 làm nhẹ hơn vì có thể trùng thi cuối kỳ và kỳ nghỉ lễ; nếu lịch thi khác, đổi chỗ các tuần.

### Lộ trình và mốc

Mỗi mốc kiểm tra trong buổi họp thứ Ba cuối giai đoạn. Mốc nào chưa đạt thì tuần sau ưu tiên sửa cho đạt trước khi làm việc mới.

| Giai đoạn | Tuần · ngày | Nội dung | Mốc |
|---|---|---|---|
| Khởi động và nền tảng | 1–3 · 14/10–03/11 | Luật thi, dữ liệu, scorer, split, alignment, bốn pilot run | **G1 · 03/11** — task_facts đủ, 4 pilot run NN/PV × B1/B2 mở lại được, thử format nộp bài |
| Baseline và cải tiến | 4–6 · 04/11–24/11 | Trần điểm, nhiễu seed, backbone, hướng E/L, công cụ so sánh, inference v0 | **G2 · 24/11** — quyết định giữ/bỏ hướng E/L, danh sách thay đổi thiết kế |
| Khóa recipe và thí nghiệm cuối | 7–8 · 25/11–08/12 | Thử tối đa 4 thay đổi, khóa recipe 01/12, chạy 12 run cuối, kiểm tra rò rỉ | Khóa recipe · 01/12 |
| Sẵn sàng cho test | 9–10 · 09/12–22/12 | Chọn ứng viên nộp, script dự đoán, tổng duyệt 1, bắt đầu viết bài | **G3 · 22/12** — tổng duyệt 1 chạy hết, chọn bài nộp chính |
| Sửa lỗi, khóa code | 11–12 · 23/12–05/01 | Dự phòng, tổng duyệt 2, bản nháp bài báo | **G4 · 05/01** — khóa code, tag `v1.0-final` |
| Test và nộp bài | 13 · 06/01–15/01 | Test dự kiến 10/01 · nộp chính trước 12/01 · hạn BTC 31/01 | — |

Đối chiếu với lộ trình trong `docs/README.md`: Mốc A = G1; Mốc B = G2 + khóa recipe 01/12; Mốc C = tuần 8; Mốc D = tuần 9–13.

## Hiện đang ở đâu (24/10/2026, tuần 2)

Kiểm tra theo nội dung repo và `docs/`. Việc nằm ngoài repo (Drive, Codabench, họp nhóm) chưa kiểm được; ghi "chưa thấy bằng chứng" không có nghĩa chưa làm.

| Hạng mục | Trạng thái |
|---|---|
| Dữ liệu English NN/PV, frozen splits (NN 3.490 dòng/446 expression; PV 1.567/160) | Đã có |
| Code dùng chung: đọc dữ liệu, alignment, split, metrics, results ZIP, web xem kết quả, tests | Đã có |
| Baseline B0 NN/PV (dev) | Đã có |
| NN B1/B2 seed 42 (dev). Modifier Spearman 0,446 → 0,503; Head 0,549 → 0,552; RMSE chưa tốt hơn | Đã có |
| PV B1/B2 | **Chưa chạy** |
| Checkpoint NN từ run cũ | Hai results ZIP không chứa weights; chưa thấy checkpoint |
| Luật thi, official metric, format nộp, `task_facts.md` | **Chưa xác nhận** (CodaBench timeout lúc kiểm tra 23/10) |
| Inference từ checkpoint, nộp bài thật | Chưa làm |
| Nhiều seed, internal test, paper | Chưa làm |

Checkout hiện tại không có `data/`, `results/`, `.venv/` (bị `.gitignore`); copy từ máy cũ hoặc Drive khi cần chạy web/test.

## Quy ước làm việc chung

Ai làm khác quy ước thì phải báo cả nhóm trước.

**Lịch họp và hạn nộp.** Mỗi tuần chạy từ thứ Tư đến thứ Ba. Hạn của việc trong tuần là 20:00 thứ Ba. Họp 45 phút lúc 21:00 thứ Ba: mỗi người 5 phút nói kết quả và vướng mắc, Khang ghi biên bản vào `plan/meetings.md`. Ngoại lệ tuần 2: hạn 20:00 Chủ nhật 01/11, họp 21:00 Thứ Hai 02/11.

**Repo và thư mục.** Repo `m1hdat/CoSense`. Mọi thay đổi vào `main` qua pull request có ít nhất một người khác review. Cấu trúc:

```text
plan/               kế hoạch, task_facts.md, meetings.md, candidates.md (không chứa dữ liệu)
notebooks/          B1/B2 (+ notebook E/L nếu giữ ở G2, + notebook dự đoán)
src/compartment/    logic dùng chung; app/ web xem kết quả; tests/
docs/               ba trang hướng dẫn hiện có; không thêm trang mới
literature-review/  research notes, ghi chú đọc
data/               dữ liệu, split, data ZIP — ngoài Git
results/            results Kaggle, baseline, phân tích — ngoài Git
  {task}/{Model}_seed{seed}/   mỗi run một thư mục
  analysis/                    báo cáo có trích câu context (eda, lỗi, hiện tượng)
```

Dữ liệu, results và mọi báo cáo có trích câu context **không lên Git** (quyền phân phối dữ liệu chưa xác nhận). Chia sẻ trong nhóm qua thư mục Drive chung `CompartMent/` (data ZIP, results, model ZIP).

**Một run là gì.** Một lần chạy notebook cho ra results ZIP, model ZIP, CSV metrics/history và `ARTIFACTS.json`. Đợi `OUTPUTS_READY`, lưu version kèm files, mở Output của *đúng saved version* để chắc files còn. Tải về `results/{task}/{Model}_seed{seed}/` ngay, không chờ.

**Bảng thí nghiệm.** Google Sheet `experiments`, mỗi run một dòng: mã run, người chạy, task, model, seed, cấu hình khác mặc định, dev Spearman/MAE/RMSE từng output, best epoch, hash dữ liệu/manifest, version notebook, ngày, commit.

**Split và seed.** `SPLIT_SEED=42` không đổi. Đổi `SEED` chỉ đổi ngẫu nhiên khi train, không đổi membership. Dev để chọn; internal test chỉ mở (`EVALUATE_LOCAL_TEST=True`) sau khi khóa recipe.

**Làm từ nhỏ đến lớn.** `SMOKE=True` (16 dòng, 1 epoch) → một run dev → nhiều seed.

**Một thay đổi một lần.** Mỗi lần đổi thiết kế ghi trước: vấn đề quan sát → giả thuyết → đúng một thay đổi → metric/slice sẽ kiểm tra → điều kiện giữ/bỏ. Chỉ giữ thay đổi hơn nhiễu seed (đo ở tuần 4) và có CI của hiệu không chứa 0.

**So sánh công bằng.** B0/B1/B2 cùng dữ liệu, backbone, head capacity, cấu hình. Biến thể chỉ áp dụng được cho B2 (ví dụ loss xếp hạng giữa các context) ghi là biến thể riêng, không đưa vào bảng B1 vs B2. NN báo modifier và head riêng; PV riêng; không gộp.

**Đổi logic chung thì sửa cả B1 và B2** (và notebook E/L nếu có), rồi chạy `.\.venv\Scripts\python.exe -m pytest -q`.

**Một việc được tính là xong khi:** code đã merge qua pull request có người khác review, kết quả đã vào bảng thí nghiệm, file nằm đúng thư mục, và mọi ô trong "Hoàn thành khi" đã tick.

**Điều chưa xác nhận không ghi như sự thật.** Official scorer, format nộp, số lượt nộp, lịch paper: ghi "chưa xác nhận" cho đến khi `task_facts.md` có nguồn.

**Viết tắt.** NN: noun compound. PV: particle verb. B0: luôn đoán trung bình nhãn train. B1: chỉ đọc expression. B2: đọc thêm câu context. E0/E1: hướng embedding (không huấn luyện / đặc trưng + mô hình nhỏ). L: hướng LLM.

## Tuần 1 (14/10–20/10): Khởi động

Cuối tuần này cả nhóm đã đăng ký cuộc thi, có dữ liệu, có môi trường chạy chung và biết bài nộp được chấm thế nào. **Trạng thái 24/10: dữ liệu và công cụ đã có; phần luật thi vẫn mở.**

### Khang: Đăng ký cuộc thi và chốt các thông tin còn thiếu

**Mục tiêu:** trả lời chắc chắn những điều trang nhiệm vụ chưa nói, để mọi việc sau dựa trên thông tin thật.

**Việc cần làm:**

- Đảm bảo Cương, Đạt, Phát có quyền ghi vào `m1hdat/CoSense`; bật bảo vệ `main` (cần review) nếu chủ repo đồng ý.
- Tạo tài khoản Codabench, mở cuộc thi 17971 (lần kiểm tra 23/10 bị timeout, thử lại), đọc các tab Overview, Evaluation, Data, Phases, Terms, Forum rồi đăng ký. Tạo đội nếu nền tảng hỗ trợ.
- Ghi vào `plan/task_facts.md` câu trả lời cho từng câu, kèm tên tab nguồn:
  1. Thước đo chính thức là gì, tính trong phạm vi nào: trong từng expression, toàn bộ, hay theo từng nhóm (EN-NN, EN-PV)?
  2. Có leaderboard riêng cho từng nhóm không? Điểm tổng gộp thế nào? **Có nộp riêng English được không, hay phải có ngôn ngữ khác mới xếp hạng?**
  3. Thang điểm và chiều (cao = ghép nghĩa?). Nhãn là trung bình? NN có hai đầu ra `ModAvg`/`HeadAvg`, PV có `Avg` — format nộp xử lý ra sao?
  4. Kích thước train/dev/test. Test có chứa expression chưa gặp trong train không?
  5. Định dạng file dữ liệu và file nộp (tên file, cột, header, thứ tự ID, nén). Helper nội bộ đang giả định `en-nn-pred.tsv`, `en-pv-pred.tsv`, ID + 2/1 số.
  6. Có baseline chính thức, sample submission, scorer chính thức không? Điểm bao nhiêu?
  7. Mỗi pha nộp được bao nhiêu lần? Tính bài cuối hay bài tốt nhất?
  8. Có được dùng dữ liệu ngoài, mô hình tiền huấn luyện, LLM/API trả phí, chú thích thủ công không? Điều khoản chia sẻ dữ liệu?
  9. Mỗi đội tối đa mấy người?
  10. Hạn chót theo múi giờ nào?
- Câu nào không tìm được: ghi "chưa rõ" và gửi một email tới `semeval-compartment@ims.uni-stuttgart.de` (giới thiệu đội, các câu hỏi, xin vào mailing list).
- Ghi tên file và số dòng dữ liệu vào `task_facts.md` (đã có số liệu NN/PV ở bảng trên).

**Đầu ra:** `plan/task_facts.md`.

**Hoàn thành khi:**

- [ ] Cả nhóm có quyền ghi vào repo và `main` đã được bảo vệ
- [ ] Đội đã đăng ký thành công trên Codabench
- [ ] `task_facts.md` trả lời đủ 10 câu; câu chưa rõ đã được hỏi qua email
- [x] Dữ liệu English NN/PV đã có (data ZIP, raw TSV, manifests)

### Cương: Môi trường chạy chung

**Mục tiêu:** cả 4 người chạy được cùng một code, không ai phải tự mày mò cài đặt.

**Đã có:** `pyproject.toml`, `requirement.txt`, README hướng dẫn tạo `.venv` và mở web; notebook B1/B2 tự xử lý dependencies trên Kaggle (thay cho `00_setup.ipynb` của bản Word).

**Việc còn lại:**

- Mỗi người tự: tạo `.venv`, `pytest -q` qua, mở web local tại http://localhost:8501, chạy `SMOKE=True` notebook B1 trên Kaggle (Add Input Dataset từ data ZIP, bật GPU và Internet).
- Ghi mọi chỗ bị vướng vào README (sửa README hiện có, không tạo tài liệu mới).
- Thống nhất quy ước đặt tên nhánh `<tên>/<việc>` (ví dụ `cuong/scorer`) và ghi vào README.

**Hoàn thành khi:**

- [x] Repo có cấu trúc, `requirement.txt`, README
- [ ] Cả 4 người đã chạy được pytest, web và smoke B1
- [ ] Ít nhất một pull request đã merge đúng quy trình review

### Đạt: Nắm pipeline fine-tune

**Mục tiêu:** hiểu cách B1/B2 fine-tune và cách các đội trước làm, để chạy và sửa notebook không bị mù.

**Việc cần làm:**

- Đọc `literature-review/RN002.md` (thiết kế nghiên cứu), `RN006.md` (Garcia et al. 2021, điểm type-level và token-level) và phần mô tả hệ thống tốt nhất trong bài tổng quan SemEval-2022 Task 2. Ghi chú ≤1 trang vào `literature-review/`: mỗi bài 5–7 gạch đầu dòng (input định dạng ra sao, mô hình, loss, kết quả, điều nhóm áp dụng được).
- Đọc code B1 và B2: pooling, loss, cách chọn checkpoint (dev Spearman), early stopping, fallback khi alignment lỗi. Trình bày 5 phút về khác biệt input B1 vs B2.

*(Bản Word có bài STS-B demo; bỏ vì notebook B1/B2 đã là pipeline hoàn chỉnh và có test.)*

**Đầu ra:** ghi chú đọc trong `literature-review/`.

**Hoàn thành khi:**

- [ ] Ghi chú đọc đã nộp, mỗi bài có đủ 5 ý
- [ ] Đã trình bày 5 phút trong buổi họp

### Phát: Đọc bài của ban tổ chức và liệt kê bẫy ngôn ngữ học

**Mục tiêu:** hiểu cách ban tổ chức đo compositionality, và biết trước những hiện tượng sẽ làm hệ thống sai.

**Việc cần làm:**

- Đọc Miletić & Schulte im Walde 2023, Jenkins et al. 2023, Miletić & Schulte im Walde 2024, và "Losing My Composure" (Jenkins et al. 2026, arXiv 2607.11667) nếu truy cập được. Garcia et al. 2021 đã có trong `RN006.md`.
- Viết ghi chú vào `literature-review/`: phát hiện kỹ thuật dùng được ngay (lớp BERT nào hữu ích, vai trò của head và modifier), mỗi ý ghi nguồn.
- Lập danh sách bẫy, mỗi loại 2 ví dụ **tự viết** (không trích câu từ dữ liệu):
  - PV bị tách rời: "take your coat off"
  - biến thể hình thái: took off, taken off, taking off, flea markets
  - NN viết rời / gạch nối / đứng cạnh từ khác gây ghép nhầm
  - cụm đa nghĩa (nghĩa đen vs nghĩa bóng), câu quá ngắn thiếu ngữ cảnh

**Đầu ra:** ghi chú đọc và danh sách bẫy trong `literature-review/`.

**Hoàn thành khi:**

- [ ] Mọi phát hiện đều có nguồn
- [ ] Danh sách bẫy có ít nhất 4 loại, mỗi loại 2 ví dụ
- [ ] Đã trình bày 5 phút trong buổi họp

## Tuần 2 (26/10–01/11): Dữ liệu, scorer, split, alignment

**Hạn việc 20:00 Chủ nhật 01/11, họp 21:00 Thứ Hai 02/11.** Tuần 2 kéo dài hơn một tuần quy ước nên chồng với tuần 3; thứ tự việc và các hạn nhỏ nằm ở `plan/week2.md`.

Cuối tuần này có bốn công cụ dùng chung đã được kiểm tra: bộ đọc dữ liệu, scorer nội bộ, split cố định, và hàm tìm cụm từ trong câu. **Trạng thái 24/10: cả bốn đã có trong `src/compartment/`; còn phần đo đạc và báo cáo.**

### Khang: Bộ đọc dữ liệu và phân tích khám phá

**Đã có:** `load_records`, `audit`, `duplicates` trong `data.py` (kiểm tra schema, ID trùng, Std âm, canonical constituents); Data explorer trong web.

**Việc còn lại:** viết `results/analysis/eda.md` (≤1 trang, kèm 3 hình quan trọng nhất) trả lời:

- Số expression và số câu mỗi task; mỗi expression có bao nhiêu câu (nhỏ nhất, trung vị, lớn nhất).
- Histogram gold: `ModAvg`, `HeadAvg`, `Avg`.
- Chiều thang: in 10 câu điểm cao nhất và 10 thấp nhất mỗi output, đọc và xác nhận điểm cao nghĩa là gì (đối chiếu `task_facts.md` câu 3).
- SD của gold bên trong từng expression (expression gần như không đổi giữa các câu rất khó xếp hạng nội bộ); phân bố annotation Std; độ dài câu.

**Hoàn thành khi:**

- [x] Bộ đọc dữ liệu dùng chung có test
- [ ] `eda.md` có đủ các mục và đã trình bày trong họp

### Cương: Scorer nội bộ

**Đã có:** `evaluate` (Spearman, RMSE, delta MAE, variance correlation, coverage và lý do NA), `error_groups`, `compare_predictions` (bootstrap ghép cặp theo expression cho gain MAE), test trong `tests/test_pipeline.py`.

**Việc còn lại:**

- `evaluate` chưa trả MAE tổng (MAE chỉ nằm trong `error_groups`): thêm field MAE hoặc ghi rõ cách tính.
- Thêm test ví dụ tính tay: gold [4,8; 4,5; 1,0; 1,3], pred [0,91; 0,62; 0,40; 0,35] phải ra ρ = 0,8; đảo dấu dự đoán ra −0,8.
- Khi `task_facts.md` cho biết official scorer: nếu phạm vi tính khác (trong từng expression, hoặc theo nhóm) thì thêm tham số `scope` vào `evaluate`. Nếu có scorer chính thức, chạy cả hai trên cùng một file dự đoán ngẫu nhiên và so khớp.
- Thêm ví dụ gọi `evaluate` vào README.

**Hoàn thành khi:**

- [x] Test cho Spearman NA, delta, bootstrap, ID ghép cặp đã qua
- [ ] Có MAE tổng và test ví dụ tính tay
- [ ] Khớp scorer chính thức đến 4 chữ số thập phân (nếu có scorer chính thức)

### Đạt: Split cố định và đối chứng seen/unseen

**Đã có:** split cố định theo expression (≈70/15/15, seed 42) cho NN (2.479/504/507 dòng) và PV (1.100/231/236), exact-duplicate context giữ cùng partition, SHA256 và manifest trong data ZIP; `overlap_report`; thiết kế seen/unseen cho RQ2 (`seen_unseen_design`). Thay cho 5-fold của bản Word.

**Việc còn lại:**

- Chạy `pytest` phần split và `overlap_report`; ghi lại thành phần (constituent) dùng chung giữa train/dev/test, vì có thể bị hỏi trong paper (expression chưa gặp khi train không có nghĩa thành phần chưa gặp).
- Quyết định có chạy đối chứng seen/unseen (RQ2) hay không, và thiết kế (support count, ngân sách dòng) **trước khi xem kết quả**; ghi vào `plan/meetings.md`. Split hiện tại không tự chứng minh unseen khó hơn seen.
- Nếu `task_facts.md` cho biết test official chứa nhiều expression đã có trong train, báo nhóm: thiết kế target-disjoint chỉ mô phỏng một tình huống.

**Hoàn thành khi:**

- [x] Split cố định, manifest và test đã có
- [ ] Quyết định RQ2 đã ghi biên bản

### Phát: Định vị cụm từ trong câu

**Đã có:** `align` bảo thủ trong `data.py`: dạng biến hình, bảng động từ bất quy tắc, PV tách rời (động từ đứng trước tiểu từ), NN chỉ chấp nhận khoảng trắng/gạch nối giữa hai phần; ambiguous/missing dùng fallback expression và giữ dòng; có `overrides` cho span thủ công.

**Việc còn lại:**

- Đo coverage theo task × split: exact / inflection / ambiguous / missing. Xuất các dòng thất bại ra `results/analysis/alignment_failures.tsv`.
- Mục tiêu (đề xuất): ambiguous + missing ≤ 5% mỗi task. Nếu vượt, bổ sung bảng bất quy tắc hoặc dùng `overrides` và ghi rõ.
- Thêm test cho từng loại bẫy ở tuần 1.
- **Sửa `align` thì phải sửa ở cả hai notebook** (và notebook E/L nếu có).

**Hoàn thành khi:**

- [x] Alignment bảo thủ có test (biến hình, tách rời, mơ hồ)
- [ ] Có số coverage theo task × split và danh sách thất bại
- [ ] Mỗi loại bẫy ở tuần 1 có ít nhất một test

## Tuần 3 (28/10–03/11): Mốc A và nộp thử (G1)

Cuối tuần này có bốn pilot run hợp lệ NN/PV × B1/B2 (Mốc A), `task_facts.md` đã trả lời, và format nộp bài đã được thử. Đây là mốc G1.

### Khang: Pipeline nộp bài, bảng thí nghiệm, họp G1

**Mục tiêu:** biến một file dự đoán thành bài nộp hợp lệ chỉ bằng một lệnh, và chứng minh nó chạy được trên Codabench.

**Việc cần làm:**

- Đối chiếu `write_submission`, `validate_submission`, `submission_zip` trong `artifacts.py` với format chính thức ở `task_facts.md` (tên file, header, thứ tự ID, số cột, nén). Sửa nếu khác giả định.
- Nộp thử lên pha phát triển (nếu có) một file dự đoán B0 hoặc ngẫu nhiên trên input chính thức, để chắc định dạng được chấp nhận. Gặp lỗi thì ghi thông báo lỗi và sửa.
- Tạo Google Sheet `experiments`, điền các run đã có (B0 NN/PV, NN B1/B2 seed 42).
- Chủ trì G1: kiểm tra công cụ tuần 2, `task_facts.md`, bốn pilot run, kết quả nộp thử.

**Đầu ra:** `artifacts.py` đã đối chiếu format, Google Sheet, ít nhất một bài nộp thử (nếu pha dev có).

**Hoàn thành khi:**

- [ ] Ít nhất một bài nộp được Codabench chấp nhận (hoặc ghi rõ pha phát triển không cho nộp)
- [ ] Format trong `artifacts.py` khớp `task_facts.md`
- [ ] Bảng experiments có các run của tuần này

### Cương: Hướng E0 v1 — embedding không huấn luyện

**Điều kiện:** hướng phụ, giữ hay bỏ ở G2. Luật phải cho dùng mô hình tiền huấn luyện (gần như chắc chắn, vì B1/B2 cũng dùng BERT).

**Mục tiêu:** một hệ thống không huấn luyện cho dự đoán hợp lệ, làm mốc so sánh. **Ý tưởng:** nếu vector của cụm trong câu nằm gần vector của thành phần, cụm được dùng theo nghĩa ghép.

**Việc cần làm:**

- Làm trước trên đúng 1 expression (~10 câu) để xem từng bước, rồi mới chạy toàn bộ.
- Dùng cùng backbone với B1/B2 (BERT-base-uncased, pin revision), `output_hidden_states=True`. Vector cụm = trung bình subword nằm trong span từ `align`; dòng fallback dùng vector expression.
- Vector thành phần: từ đứng một mình. NN: cosine(cụm, modifier) → proxy `ModAvg`, cosine(cụm, head) → proxy `HeadAvg`. PV: trung bình cosine với động từ gốc và tiểu từ. Thử lớp 1, 4, 8, 12.
- Cosine không cùng thang 0–5, nên chỉ Spearman có nghĩa; muốn có MAE/RMSE thì hiệu chỉnh tuyến tính bằng split train, ghi thành biến thể riêng. Đổi dấu nếu chiều ngược gold.
- Chạy trên dev, xuất results ZIP cùng định dạng (`export_bundle`, `model='E0'`) để web đọc được.

**Đầu ra:** `notebooks/E0_kaggle.ipynb`, results ZIP.

**Hoàn thành khi:**

- [ ] Có results ZIP dev cho NN và PV, mở được trong web
- [ ] Điểm theo từng lớp đã ghi vào bảng thí nghiệm
- [ ] Notebook có phần minh họa trên 1 expression: cosine từng câu cạnh điểm gold

### Đạt: Mốc A — bốn pilot run

**Mục tiêu:** có bốn pilot run hợp lệ NN/PV × B1/B2 và chắc checkpoint lưu được.

**Việc cần làm:**

- Kiểm tra saved version Kaggle của hai run NN cũ: còn `best.pt`, tokenizer, config không? Tải về nếu còn. Hai results ZIP cũ chỉ khoảng 139/158 KB, không chứa weights.
- Chạy PV B1 và PV B2 (mỗi cái một session): `TASK='pv'`, `SEED=42`, `SPLIT_SEED=42`, `SMOKE=False`, `EVALUATE_LOCAL_TEST=False`.
- Xác minh exporter mới trên một saved version GPU (chưa ai xác minh): `OUTPUTS_READY`, đủ results ZIP, model ZIP, CSV, `ARTIFACTS.json` trong Output của đúng saved version.
- Tải về `results/pv/B1_seed42/`, `results/pv/B2_seed42/`; mở results ZIP bằng web; ghi data/config/version vào bảng.
- Nếu checkpoint NN cũ không còn: chạy lại NN B1/B2 seed 42 bằng exporter mới.

**Đầu ra:** `results/pv/...`, model ZIP.

**Hoàn thành khi:**

- [ ] Bốn pilot run có results ZIP mở được trong web
- [ ] Mỗi run có model ZIP đã kiểm tra tồn tại trên saved version
- [ ] Điểm từng output đã vào bảng thí nghiệm; so với B0 trên cùng split

### Phát: Đọc lỗi pilot NN, và khảo sát LLM (nếu luật cho phép)

**Mục tiêu:** có các ví dụ cụ thể cho RQ3 và biết LLM mở chấm compositionality đến đâu.

**Việc cần làm:**

- **Đọc lỗi pilot NN** (bắt buộc): chọn 10–20 dòng dev gồm B2 giúp, B2 làm tệ hơn, fallback/ambiguous. Ghi gold, dự đoán hai model, spans, lý do nghi ngờ vào `results/analysis/`. Không tự sửa gold theo dự đoán.
- **Khảo sát LLM** (chỉ nếu `task_facts.md` cho phép LLM; giới hạn 3 giờ): LLM mở cỡ 7–8B chạy 4-bit trên T4; tập cố định 100 dòng **chỉ lấy từ split train** (chia NN/PV, 25 expression × 4 câu); prompt zero-shot giải thích compositionality, đánh dấu cụm, yêu cầu trả số trên đúng thang (NN hai số, PV một số). Báo tỉ lệ tách được số và Spearman.

**Đầu ra:** ghi chú lỗi; tập khảo sát và prompt v1 (nếu làm).

**Hoàn thành khi:**

- [ ] 10–20 ví dụ lỗi đã ghi đủ trường
- [ ] (LLM) tỉ lệ tách được số ≥ 95%, Spearman và thời gian mỗi câu đã vào bảng

## Tuần 4 (04/11–10/11): Trần điểm, nhiễu seed, baseline đầy đủ

Cuối tuần này mọi hướng đều có điểm dev, và nhóm biết điểm trần thực tế cùng mức nhiễu do seed.

### Khang: Điểm trần và vai trò của type-level

**Mục tiêu:** biết bao nhiêu phần điểm đến từ việc chỉ cần biết "expression này nói chung ghép nghĩa đến đâu".

**Việc cần làm:**

- Kiểm tra dữ liệu có điểm từng người chấm không. Schema hiện chỉ có `Avg` và `Std`, nên split-half reliability (điểm "giỏi như con người") **không tính được**; ghi rõ.
- "Type-level oracle": dự đoán mỗi dòng bằng gold trung bình của expression chứa nó (chỉ để tính trần, không phải hệ thống thật). Chấm Spearman/MAE/RMSE trên dev, so với B0, B1, B2.
- Phân rã phương sai: tỉ lệ phương sai giữa expression so với bên trong từng expression (ANOVA một chiều hoặc ICC), tính riêng `ModAvg`, `HeadAvg`, `Avg`.
- Viết `results/analysis/ceiling.md`: ba con số và 3–5 câu kết luận: ưu tiên khả năng đọc ngữ cảnh hay kiến thức type-level.

**Hoàn thành khi:**

- [ ] `ceiling.md` có đủ các con số, hoặc ghi rõ vì sao không tính được
- [ ] Kết luận chiến lược đã thảo luận trong họp

### Cương: E0 v2 — quét cấu hình có hệ thống

**Mục tiêu:** tìm cấu hình embedding tốt nhất cho mỗi task bằng thí nghiệm, không bằng đoán.

**Việc cần làm:**

- Quét: backbone (bert-base-uncased, bert-base-cased, roberta-base), lớp 0–12, cách so sánh (cosine với từng thành phần, trung bình, cosine với tổng hai vector), vector thành phần lấy khi đứng một mình hoặc ngay tại vị trí trong câu. Lưu vector trung gian để không tính lại.
- **Chọn cấu hình trên split train** (E0 không huấn luyện nên train là tập chọn sạch), chấm trên dev. Không chọn bằng dev rồi báo dev.
- Bảng: mỗi dòng một cấu hình, cột là điểm từng output; vẽ đường điểm theo lớp.

**Đầu ra:** bảng quét trong `results/analysis/`, results ZIP E0 v2.

**Hoàn thành khi:**

- [ ] Bảng quét và biểu đồ theo lớp đã có
- [ ] Cấu hình được chọn trên train, chấm trên dev

### Đạt: Nhiễu seed

**Mục tiêu:** biết dao động do seed lớn cỡ nào so với chênh lệch giữa B1 và B2.

**Việc cần làm:**

- Chạy NN B1 và B2 với `SEED=43` và `44` (cùng recipe, cùng `SPLIT_SEED=42`); PV cùng làm nếu quota cho phép.
- Tính mean ± sample SD của dev Spearman/RMSE từng output. Nếu SD lớn hơn chênh B2−B1, mọi so sánh từ đây phải dùng nhiều seed.
- Lưu results và model ZIP như tuần 3.

**Hoàn thành khi:**

- [ ] Có ít nhất 3 seed cho NN B1/B2, kết quả vào bảng thí nghiệm
- [ ] Mức nhiễu seed đã báo trong họp

### Phát: Prompt LLM (nếu luật cho phép) hoặc đọc lỗi PV

**Việc cần làm:**

- **Nếu LLM được phép:** thử ≥4 biến thể prompt trên tập khảo sát tuần 3, giữ nguyên mô hình: zero-shot (mốc), few-shot 4 ví dụ **lấy từ split train**, khác expression với dòng đang chấm; yêu cầu giải thích rồi mới cho điểm; thêm nghĩa đen của thành phần. Ghi Spearman, tỉ lệ tách được số, thời gian; so phân bố điểm LLM với gold (có nghiêng về thành ngữ không). Ước lượng thời gian chạy trên toàn dev và test.
- **Nếu không:** đọc lỗi pilot PV (10–20 dòng, như tuần 3) và lập danh sách giả thuyết "một thay đổi" cho G2 theo mẫu *vấn đề → giả thuyết → thay đổi → metric/slice → giữ/bỏ*.

**Hoàn thành khi:**

- [ ] (LLM) bốn biến thể có bảng so sánh và ước lượng thời gian
- [ ] (không LLM) 10–20 ví dụ PV và danh sách giả thuyết đã trình bày

## Tuần 5 (11/11–17/11): Cải tiến từng hướng

Cuối tuần này mỗi hướng có một bản cải tiến, và nhóm có công cụ so sánh hai hệ thống bất kỳ có cơ sở thống kê.

### Khang: Công cụ so sánh nhiều hệ thống

**Mục tiêu:** một công cụ ai cũng chạy được để biết hệ thống sai ở đâu, kèm khoảng tin cậy.

**Đã có:** `compare_predictions` chỉ so B1 với B2, chỉ cho gain MAE; Results explorer trong web.

**Việc cần làm:**

- Mở rộng thành hai model bất kỳ cùng task/split/seed (B0, B1, B2, E0/E1, L). Thêm gain Spearman, bootstrap theo expression, báo CI của hiệu và tỉ lệ A hơn B. Quy tắc nhóm: chỉ kết luận A tốt hơn B khi CI của hiệu không chứa 0. CI của MAE gain không phải CI của Spearman gain.
- Với mỗi hệ thống: 20 expression sai nhất, biểu đồ phân tán gold vs dự đoán; ma trận tương quan giữa dự đoán các hệ thống (càng thấp thì ghép càng có lợi).
- Thêm vào Results explorer và test trong `tests/test_pipeline.py`.

**Hoàn thành khi:**

- [ ] So sánh được hai hệ thống bất kỳ chỉ bằng cách chọn hai results ZIP
- [ ] Cả nhóm thống nhất quy tắc so sánh theo CI

### Cương: E1 — đặc trưng và mô hình nhỏ

**Mục tiêu:** biến kết quả quét thành đặc trưng số, để mô hình đơn giản học cách kết hợp.

**Việc cần làm:**

- Khoảng 15–25 đặc trưng mỗi dòng: cosine của 3 cấu hình tốt nhất ở tuần 4; khoảng cách giữa vector cụm trong câu và vector trung bình của cùng expression qua mọi câu của nó (chỉ dùng câu đầu vào, không dùng gold, nên hợp lệ trên test); tần suất cụm và thành phần (`wordfreq`, tiếng Anh); độ dài câu; vị trí cụm; `alignment_status`; task.
- Ridge và LightGBM huấn luyện trên train, chọn siêu tham số bằng GroupKFold theo expression **bên trong train**, báo dev. NN hai model (modifier, head), PV một model.
- Xuất results ZIP (`model='E1'`), so với E0 bằng công cụ của Khang.
- Ghi mô tả từng cột đặc trưng (một dòng mỗi cột) trong notebook.

**Đầu ra:** `notebooks/E1_kaggle.ipynb`, results ZIP.

**Hoàn thành khi:**

- [ ] Mỗi đặc trưng có một dòng mô tả
- [ ] E1 đã được chấm và so với E0 có khoảng tin cậy

### Đạt: So sánh backbone

**Mục tiêu:** chọn 1–2 backbone cho B1/B2.

**Việc cần làm:**

- Giữ nguyên mọi thứ, chỉ đổi backbone: bert-base-uncased (mốc hiện tại, revision đã pin), roberta-base, deberta-v3-base. Đổi cả B1 và B2, NN và PV nếu quota đủ, seed 42, dev. Pin revision cho từng backbone.
- Ghi điểm từng output, thời gian, loại GPU; so từng backbone với mốc bằng công cụ của Khang.
- Chọn backbone đi tiếp và ghi lý do: điểm, thời gian, độ ổn định giữa seed.

**Hoàn thành khi:**

- [ ] Đủ results cho mọi backbone đã thử
- [ ] Quyết định chọn backbone ghi vào biên bản họp

### Phát: Cải tiến alignment PV

**Mục tiêu:** kiểm tra heuristic alignment ở PV *trước khi* đổi encoder.

**Việc cần làm:**

- Từ danh sách thất bại tuần 2–4, chọn **đúng một** cải tiến (ví dụ thêm dạng bất quy tắc, hoặc chọn ứng viên gần nhất thay vì fallback khi có nhiều ứng viên). Ghi giả thuyết và điều kiện giữ/bỏ trước khi làm.
- Đo: coverage trước/sau; điểm B2 trên các dòng bị ảnh hưởng (phối hợp Đạt chạy lại B2 PV với `align` mới).
- Cập nhật `align` trong cả hai notebook và `data.py`, kèm test.

**Hoàn thành khi:**

- [ ] Có số đo trước/sau cho coverage và cho các dòng bị ảnh hưởng
- [ ] Quyết định giữ hay bỏ ghi rõ kèm số liệu

## Tuần 6 (18/11–24/11): Đánh giá giữa kỳ (G2)

Cuối tuần này nhóm chốt hướng đầu tư cho 4 tuần sau. Đây là mốc G2.

### Khang: Báo cáo giữa kỳ và họp G2

**Việc cần làm:**

- Cập nhật bảng tổng hợp mọi hệ thống: điểm dev từng output, CI, so với B0, trần (tuần 4) và baseline chính thức (nếu có).
- Viết `plan/midterm.md` (1–2 trang, không trích câu context): hệ thống tốt nhất ở từng task; 3 điểm yếu lớn nhất; đề xuất dừng hướng nào, đầu tư hướng nào; rủi ro tiến độ.
- Xem lại Forum, thông báo Codabench, email ban tổ chức; cập nhật `task_facts.md`.
- Họp G2 60 phút, quyết định và ghi biên bản:
  1. **Giữ hay bỏ hướng E/L.** Giữ nếu luật cho phép, và hoặc Spearman dev ngang/cao hơn B1/B2, hoặc tương quan thấp với B1/B2 (ích cho ensemble). Mỗi notebook thêm là một bản code chung phải đồng bộ; bỏ hướng nào thì người phụ trách chuyển sang việc ở tuần 8–10.
  2. Danh sách tối đa 4 thay đổi thiết kế được thử ở tuần 7 (mỗi thay đổi theo mẫu *vấn đề → giả thuyết → thay đổi → metric → giữ/bỏ*).
  3. English-only có đủ không (theo `task_facts.md`).
  4. Chính sách refit sau khi khóa recipe (mặc định không refit).

**Đầu ra:** `plan/midterm.md`, biên bản G2.

**Hoàn thành khi:**

- [ ] `midterm.md` đã gửi cả nhóm muộn nhất tối thứ Hai 23/11
- [ ] Biên bản G2 ghi rõ bốn quyết định và thay đổi so với kế hoạch

### Cương: Đặc trưng cấp expression và độ lệch trong expression (nếu giữ E)

**Mục tiêu:** tách điểm thành hai phần: expression nói chung ghép nghĩa đến đâu (type-level) và câu này lệch khỏi mức chung bao nhiêu (token-level), theo hướng của `RN006`.

**Việc cần làm:**

- Với mỗi đặc trưng số, thêm: trung bình và SD của đặc trưng qua mọi câu cùng expression; z-score của câu trong expression; hạng của câu. Chỉ dùng câu đầu vào, không dùng gold.
- Mô hình hai tầng: tầng 1 dự đoán điểm trung bình của expression từ đặc trưng cấp expression (huấn luyện trên expression của train, phải khái quát sang expression mới); tầng 2 dự đoán độ lệch của câu. Dự đoán cuối = tổng.
- So ba cách: một tầng đặc trưng cũ, một tầng đặc trưng mới, hai tầng. Báo Spearman, MAE/RMSE và delta MAE, từng output.

**Hoàn thành khi:**

- [ ] Bảng so sánh ba cách có khoảng tin cậy
- [ ] Cách tốt nhất đã xuất results ZIP

### Đạt: Notebook dự đoán v0

**Mục tiêu:** có đường từ checkpoint đến dự đoán sớm, vì đây là phần còn thiếu lớn nhất để nộp bài.

**Việc cần làm:**

- Viết `notebooks/PREDICT_kaggle.ipynb`: nạp model ZIP (best, tokenizer, encoder config), đọc TSV **không có gold** (`load_records(labeled=False)`), alignment và tokenization giống lúc train, xuất NN hai số / PV một số.
- Kiểm tra tính đúng: chạy trên dev của một pilot run phải trùng `predictions.tsv` trong results ZIP đến 4 chữ số thập phân.
- Đo thời gian, ghi vào `docs/RUN_GUIDE.md` (mục 5).

**Đầu ra:** `notebooks/PREDICT_kaggle.ipynb`.

**Hoàn thành khi:**

- [ ] Dự đoán lại dev khớp results ZIP (NN và PV)
- [ ] Có số liệu thời gian

### Phát: LLM trên dev (nếu giữ L) hoặc nhãn hiện tượng v0

**Việc cần làm:**

- **Nếu giữ L:** chạy prompt tốt nhất trên toàn dev. Ngoài số mô hình viết, lưu điểm kỳ vọng tính từ xác suất các token chữ số (chuẩn hóa lại rồi lấy tổng điểm × xác suất), ít hòa hạng hơn nên lợi cho Spearman. Lưu sau mỗi 500 câu. Ví dụ few-shot chỉ từ train. Xuất results ZIP (`model='L1'`, hai biến thể text/expect).
- **Nếu không:** bắt đầu viết hàm gắn nhãn hiện tượng (xem tuần 8) vào `src/compartment/metrics.py`, cạnh `error_groups`.

**Hoàn thành khi:**

- [ ] (L) đủ dự đoán cho 100% dòng dev, điểm và thời gian đã vào bảng
- [ ] (không L) hàm gắn nhãn chạy trên results ZIP hiện có

## Tuần 7 (25/11–01/12): Thay đổi thiết kế và khóa recipe

Cuối tuần này recipe được khóa (01/12). Sau đó không đổi backbone, preprocessing, siêu tham số và chính sách checkpoint.

### Khang: Ensemble v1 (nếu có ≥2 hệ thống đủ tốt)

**Việc cần làm:**

- Chuyển dự đoán sang hạng trong từng output, chuẩn hóa về [0, 1].
- Mặc định **trung bình hạng trọng số bằng nhau**. Chỉ có dev (NN 504 dòng, PV 231 dòng) để chọn, nên trọng số tối ưu dễ overfit: chỉ giữ nếu CI (bootstrap theo expression) của hiệu so với trọng số bằng nhau không chứa 0.
- Viết `src/compartment/ensemble.py`: nhận danh sách file dự đoán và trọng số, xuất một file dự đoán; dùng cho cả dev lẫn test. Có test.
- Ghi hệ thống nào đóng góp nhiều nhất.

**Hoàn thành khi:**

- [ ] Ensemble đã chấm trên dev và so với hệ thống đơn lẻ tốt nhất có CI
- [ ] `ensemble.py` có test và chạy được trên file test giả

### Cương: Hoàn thiện mô hình đặc trưng (nếu giữ E)

**Việc cần làm:**

- Ablation: lần lượt bỏ từng nhóm đặc trưng (cosine, độ lệch trong expression, tần suất, cấp expression), huấn luyện lại, ghi điểm giảm bao nhiêu. Bảng này dùng thẳng trong bài báo.
- Tinh chỉnh LightGBM bằng random search 30 cấu hình, đánh giá bằng GroupKFold theo expression trong train.
- Chốt cấu hình cuối trong notebook; nếu không hơn E1 v2 thì giữ E1 v2 và ghi lý do.

**Hoàn thành khi:**

- [ ] Bảng ablation có khoảng tin cậy
- [ ] Cấu hình cuối đã ghi trong notebook

### Đạt: Thay đổi thiết kế, siêu tham số, khóa recipe

**Mục tiêu:** lấy thêm điểm có kiểm soát, rồi khóa recipe đúng hạn.

**Việc cần làm:**

- Thử các thay đổi đã chọn ở G2 (tối đa 4), mỗi thay đổi một lần theo mẫu một-thay-đổi. Ứng viên: pooling trên span thay vì token đầu; loss xếp hạng giữa các context cùng expression (chỉ B2, ghi là biến thể riêng, không đưa vào bảng B1 vs B2); thêm thành phần vào input; layer-wise LR decay. Chạy seed 42, dev; chỉ giữ thay đổi hơn nhiễu seed và có CI không chứa 0. Không tăng epochs chỉ vì chạy nhanh. Calibration để giảm sai số điểm (nếu thử) fit bằng train/dev, ghi thành biến thể riêng.
- Tối đa 8 cấu hình siêu tham số (lr {1e-5, 2e-5, 3e-5}, epochs, max length {96, 128, 160}, dropout đầu hồi quy), chạy seed 42 dev. Ghi số giờ GPU đã dùng.
- **Ngày 01/12: khóa recipe**, ghi vào `plan/recipe_final.md`: backbone và revision, preprocessing/alignment, siêu tham số, chính sách chọn checkpoint (dev Spearman), danh sách biến thể sẽ báo cáo. Từ đây không xem internal test để đổi gì.

**Đầu ra:** `plan/recipe_final.md`, kết quả từng thay đổi trong bảng thí nghiệm.

**Hoàn thành khi:**

- [ ] Mỗi thay đổi có kết quả riêng và quyết định giữ/bỏ kèm số liệu
- [ ] `recipe_final.md` đã được cả nhóm đồng ý trong họp 01/12

### Phát: LLM v3 (nếu giữ L) hoặc nhãn hiện tượng

**Việc cần làm:**

- **Nếu giữ L:** self-consistency 5 mẫu (temperature 0,7) so với greedy trên tập khảo sát; thử LLM thứ hai khác họ, cùng prompt; chỉ chạy toàn dev nếu điểm gần bằng và tương quan với LLM thứ nhất thấp.
- **Nếu không:** hoàn thành hàm gắn nhãn hiện tượng và kiểm tay (xem tuần 8).

**Hoàn thành khi:**

- [ ] (L) ba thử nghiệm có số liệu trên tập khảo sát
- [ ] (không L) nhãn đã gắn và kiểm tay

## Tuần 8 (02/12–08/12): Mốc C — mười hai run cuối và kiểm tra tin cậy

Cuối tuần này có bảng kết quả cuối B0/B1/B2 với nhiều seed, và nhóm chắc điểm không bị thổi phồng.

### Khang: Kiểm tra rò rỉ và khả năng chạy lại

**Việc cần làm:**

- Rà theo checklist, ghi vào `plan/leakage_audit.md`:
  - mọi hệ thống dùng cùng manifest (hash khớp trong `run_metadata.json`);
  - ví dụ few-shot của LLM chỉ lấy từ train; cấu hình E0 chọn trên train;
  - trọng số ensemble và mọi lựa chọn đã cố định trước khi xem internal test;
  - không đặc trưng nào dùng gold, annotation Std hay ContextID;
  - `EVALUATE_LOCAL_TEST` chỉ bật sau khóa recipe; không chọn seed theo test.
- Nhờ một người không phải Khang chạy B1 theo README trên tài khoản Kaggle mới (smoke, rồi một run đầy đủ). Ghi chỗ vướng và sửa README.
- Đọc lại Codabench: format file test đã công bố chưa, có khác train không.

**Hoàn thành khi:**

- [ ] `leakage_audit.md` không còn mục "chưa kiểm"
- [ ] Một người không phải Khang đã chạy thành công theo README

### Cương: Phân tích lỗi định tính

**Việc cần làm:**

- Chọn 30 expression bị dự đoán kém nhất trên dev (chia đều NN/PV). Đọc từng câu cạnh gold và dự đoán. Gán nguyên nhân: expression đa nghĩa, PV tách rời, câu quá ngắn, alignment fallback, nhãn gây tranh cãi (annotation Std cao), lỗi định vị, khác.
- Tổng hợp `results/analysis/error_taxonomy.md`: bảng nguyên nhân theo số expression và task, 5 ví dụ tiêu biểu. Đề xuất 2–3 cải tiến cho phần Limitations/Future work của bài báo (recipe đã khóa nên không áp dụng).
- Đây là mô tả liên hệ, không phải bằng chứng nhân quả.

**Hoàn thành khi:**

- [ ] Đủ 30 expression đã gán nguyên nhân
- [ ] Các đề xuất đã có quyết định trong họp

### Đạt: Mười hai run cuối

**Việc cần làm:**

- Với recipe đã khóa chạy NN/PV × B1/B2 × seed 42/43/44 = 12 run. Tái sử dụng pilot/seed đã chạy nếu recipe thực sự không đổi.
- `EVALUATE_LOCAL_TEST=True` chỉ ở các run cuối; dev chọn checkpoint; internal test chấm một lần mỗi run. Không chọn seed thắng nhờ test.
- Lưu results ZIP và model ZIP của cả 12 run vào `results/` và một Kaggle Dataset riêng tư; ghi vào bảng.
- Báo mean ± sample SD qua seed, NN modifier/head và PV riêng; chênh ghép cặp B2−B1 cho Spearman, B1−B2 cho lỗi (gain dương luôn là tốt hơn); CI bootstrap theo expression (ghi rõ metric, split, số lần lấy mẫu). Ba seed không phải ba dataset độc lập.

**Hoàn thành khi:**

- [ ] Đủ 12 run, mỗi run có results ZIP và model ZIP tải lại được
- [ ] Bảng mean ± SD, coverage/fallback và chênh ghép cặp đã có

### Phát: Nhãn hiện tượng và đánh giá theo nhãn (RQ3)

**Việc cần làm:**

- Gắn nhãn tự động cho từng dòng (thêm cạnh `error_groups`): PV tách rời; alignment fallback/ambiguous; câu ngắn (<8 từ); biến thể hình thái (số nhiều, quá khứ); subword count cao hoặc bị cắt (truncation); annotation Std cao.
- Kiểm tay 50 dòng ngẫu nhiên cho mỗi nhãn để ước lượng độ chính xác nhãn.
- Điểm từng hệ thống theo nhãn, có CI, biểu đồ cột. Viết `results/analysis/phenomena.md`: hệ thống nào mạnh ở hiện tượng nào, gợi ý có nên đặt trọng số ensemble khác nhau theo nhãn. Chỉ nói liên hệ mô tả.

**Hoàn thành khi:**

- [ ] Mỗi nhãn có độ chính xác đã kiểm tay
- [ ] Biểu đồ và nhận xét đã có trong `phenomena.md`

## Tuần 9 (09/12–15/12): Sẵn sàng cho tập test

Cuối tuần này mỗi hệ thống chạy được trên file dữ liệu mới không có gold chỉ bằng một notebook, và nhóm đã chọn xong các bài sẽ nộp. Kích thước test chưa biết; ước lượng thời gian sửa lại theo `task_facts.md`.

### Khang: Chọn ứng viên nộp

**Việc cần làm:**

- Chạy lại ensemble với phiên bản cuối. Chọn các ứng viên khác bản chất, tối đa bằng số bài được nộp (theo `task_facts.md`):
  - A: B1 hoặc B2 tốt nhất (chọn theo dev, mỗi task riêng)
  - B: A trung bình 3 seed
  - C: ensemble với E/L nếu CI cho thấy lợi rõ
- Nếu pha phát triển có leaderboard, nộp thử cả ba. Ghi quyết định và lý do vào `plan/candidates.md`.
- Chốt chính sách refit: mặc định **không refit** (giữ internal test sạch). Nếu refit trên train+dev, số epoch cố định theo trung vị best epoch, ghi rõ tập nào được dùng, không dùng official gold để tune.

**Hoàn thành khi:**

- [ ] `candidates.md` liệt kê từng ứng viên: thành phần, trọng số, điểm dev/test nội bộ và CI
- [ ] Ứng viên đã nộp thử thành công (nếu pha dev có leaderboard)

### Cương: Sẵn sàng cho test (nếu giữ E), nếu không thì kiểm tra nộp bài

**Việc cần làm:**

- **Nếu giữ E:** notebook đọc TSV không gold → dự đoán, luôn trả số hữu hạn (không có span → fallback; expression chỉ một câu → SD trong expression không xác định; từ không có trong `wordfreq`). Kiểm tra parity: chạy trên dev phải khớp results ZIP.
- **Nếu không:** chạy `validate_submission` trên file dự đoán thật của Đạt, thử các trường hợp biên (thứ tự ID, trùng ID, giá trị không hữu hạn) và bắt đầu phần Introduction/Background của bài báo sớm.

**Hoàn thành khi:**

- [ ] Notebook/validator chạy trên file mới, đủ dòng, không NaN
- [ ] Kiểm tra khớp với dev đã qua

### Đạt: Chốt inference cho B1/B2

**Việc cần làm:**

- Hoàn thiện `PREDICT_kaggle.ipynb`: nạp 3 model ZIP (3 seed) của mô hình đã chọn → trung bình → TSV nộp. Chạy trên input chính thức của pha phát triển (nếu có) và trên internal test đã bỏ gold để so với results.
- Upload model ZIP vào Kaggle Dataset, ghi đường dẫn vào `docs/RUN_GUIDE.md`. Đo thời gian.

**Hoàn thành khi:**

- [ ] Đủ 12 model ZIP và tải lại được từ một tài khoản khác
- [ ] Notebook dự đoán chạy được và có số liệu thời gian

### Phát: Dự đoán LLM có cache (nếu giữ L), nếu không thì kiểm tra format

**Việc cần làm:**

- **Nếu giữ L:** notebook có `--cache`: mỗi dòng chấm xong ghi cache theo ID và phiên bản prompt, chạy lại tự bỏ qua dòng đã có. Cố định mô hình, prompt, seed, temperature, cách chọn few-shot cho test (từ train theo quy tắc cố định). Đo thời gian trên cỡ test; nếu quá dài, lập kế hoạch chia nhiều tài khoản.
- **Nếu không:** đối chiếu format nộp với `task_facts.md` lần nữa, tập hợp các ca biên vào test.

**Hoàn thành khi:**

- [ ] (L) ngắt giữa chừng rồi chạy lại vẫn ra đủ dòng, không chấm lặp
- [ ] Có kế hoạch thời gian chạy trên tập test

## Tuần 10 (16/12–22/12): Tổng duyệt lần 1 và bắt đầu viết bài (G3)

Cuối tuần này quy trình ngày thi đã chạy từ đầu đến cuối một lần, và bài báo có bản nháp các phần mô tả hệ thống. Đây là mốc G3. Từ tuần này khối lượng giảm dần để nhường chỗ cho ôn thi.

### Khang: Tổng duyệt lần 1 và họp G3

**Việc cần làm:**

- Chuẩn bị "test giả": input chính thức của pha dev, hoặc internal test đã bỏ gold, đặt tên và định dạng như file test dự kiến.
- Đóng vai ngày thi: gửi file cho cả nhóm cùng lúc; mỗi người chạy notebook dự đoán của mình, gửi lại `test_<hệ thống>.csv`; Khang chạy `ensemble.py`, `write_submission`, `submission_zip`, rồi nộp lên pha phát triển.
- Ghi `plan/rehearsal_1.md`: thời gian từng bước, ai vướng gì, file nào sai định dạng, việc phải sửa trước tuần 12.
- Họp G3: chốt hệ thống nộp chính, xác nhận khóa code 05/01, phân công lại tuần 11–13.

**Hoàn thành khi:**

- [ ] Tổng duyệt chạy hết từ file test giả đến bài nộp
- [ ] `rehearsal_1.md` có thời gian từng bước và danh sách lỗi

### Cương: Bài báo — mở đầu và bối cảnh

**Việc cần làm:**

- Tạo project Overleaf bằng template của SemEval-2027; chưa công bố thì dùng template ACL hiện hành. Mời cả nhóm. Tiêu đề theo mẫu `<Tên đội> at SemEval-2027 Task 5: <mô tả ngắn>` (mẫu xác nhận khi BTC công bố).
- Introduction (~0,5 trang): bài toán, RQ1–RQ3, tóm tắt hệ thống và kết quả chính (để trống số).
- Background (~0,75 trang): nhiệm vụ và dữ liệu (dựa vào `eda.md`); công trình liên quan (shared task trước, bài của ban tổ chức, Garcia et al. 2021 từ `RN006`).
- Tạo `paper/refs.bib` từ ACL Anthology. Bài mô tả nhiệm vụ chưa có thì dùng khóa tạm.

**Hoàn thành khi:**

- [ ] Overleaf có Introduction và Background đủ ý
- [ ] Mọi trích dẫn có trong `refs.bib`, PDF biên dịch không lỗi

### Đạt: Bài báo — hệ thống fine-tune và thiết lập thí nghiệm

**Việc cần làm:**

- System Overview (~0,5 trang): input B1 vs B2, backbone và revision, pooling, loss, cách lấy trung bình 3 seed.
- Experimental Setup (~0,4 trang): split theo expression ≈70/15/15, metrics, bảng siêu tham số cuối, phần cứng, thời gian, phiên bản thư viện.
- Mọi con số lấy từ bảng thí nghiệm; ghi chú trong LaTeX dòng nào của bảng là nguồn.

**Hoàn thành khi:**

- [ ] Hai mục đã có trên Overleaf
- [ ] Bảng siêu tham số khớp `recipe_final.md`

### Phát: Bài báo — tiền xử lý ngôn ngữ (và hệ thống LLM nếu có)

**Việc cần làm:**

- Phần tiền xử lý (~0,3 trang): định vị cụm từ (kể cả PV tách rời), fallback, kèm tỉ lệ coverage từng loại alignment.
- Phần LLM (~0,4 trang, nếu giữ L): mô hình và phiên bản, cách xây prompt, điểm kỳ vọng từ xác suất, self-consistency nếu dùng. Prompt đầy đủ vào phụ lục.

**Hoàn thành khi:**

- [ ] Các phần đã có trên Overleaf
- [ ] (L) prompt đầy đủ nằm trong phụ lục

## Tuần 11 (23/12–29/12): Sửa lỗi sau tổng duyệt, tuần nhẹ

Cuối tuần này lỗi tổng duyệt đã được sửa, pipeline vẫn chạy khi một hệ thống trục trặc, và bài báo có phần kết quả. Tuần này trùng Giáng sinh nên mỗi việc giữ ở mức khoảng 4 giờ.

### Khang: Sửa lỗi và phương án dự phòng

**Việc cần làm:**

- Sửa mọi lỗi trong `rehearsal_1.md`.
- Dự phòng trong `ensemble.py`: dòng thiếu hoặc NaN ở một hệ thống thì dùng trung bình các hệ thống còn lại; một hệ thống không chạy được thì ensemble chạy với phần còn lại và in cảnh báo.
- Kiểm tra file test ngay khi tải về (đủ cột, ID không trùng, số dòng, tỉ lệ expression chưa từng gặp trong train), dựa trên `load_records(labeled=False)`.
- Viết `plan/test_day_checklist.md`: từng bước ngày thi, ai làm, hạn giờ theo cả giờ Việt Nam lẫn giờ Codabench.

**Hoàn thành khi:**

- [ ] Mọi lỗi trong `rehearsal_1.md` đã sửa
- [ ] Đã thử xóa một file dự đoán, ensemble vẫn tạo bài nộp hợp lệ

### Cương: Bài báo — bảng kết quả chính và ablation

**Việc cần làm:**

- Bảng kết quả chính từ Mốc C: B0/B1/B2, mean ± SD qua seed, NN modifier/head và PV riêng, coverage/fallback; chênh ghép cặp B2−B1 với CI.
- Nếu giữ E: mục System Overview cho E0/E1 (~0,4 trang) và bảng ablation, kèm 3–4 câu đọc bảng.
- Cột điểm test chính thức để `\todo{}`.

**Hoàn thành khi:**

- [ ] Bảng và mục đã có trên Overleaf, số liệu khớp bảng thí nghiệm

### Đạt: Ý tưởng cuối có giới hạn thời gian và kiểm tra chạy lại

**Việc cần làm:**

- Chọn đúng một ý tưởng từ `error_taxonomy.md` hoặc `phenomena.md`; giới hạn 4 giờ. Chỉ đưa vào hệ thống chính nếu CI của mức cải thiện không chứa 0 *và* không phá recipe đã khóa (nếu phá thì đây là kết quả phụ cho bài báo). Kết quả âm tính vẫn ghi lại.
- Kiểm tra chạy lại: chạy notebook dự đoán hai lần trên cùng một file, kết quả phải giống nhau đến 4 chữ số thập phân. Nếu không, cố định thêm seed và bật chế độ deterministic của PyTorch.

**Hoàn thành khi:**

- [ ] Ý tưởng có kết quả ghi trong bảng, dương hay âm
- [ ] Kiểm tra chạy lại đã qua

### Phát: Phân tích theo hiện tượng cho bài báo

**Việc cần làm:**

- Một hình quan trọng nhất: điểm từng hệ thống theo hiện tượng (PV tách rời, fallback, câu ngắn...), kèm CI. Vẽ bằng matplotlib vừa một cột template ACL (~7,7 cm), xuất PDF vector, đọc được khi in đen trắng.
- Nháp mục Analysis (~0,4 trang): hiện tượng nào khó nhất, hướng nào xử lý tốt hơn, ví dụ từ `error_taxonomy.md` của Cương. Ghi rõ đây là liên hệ mô tả.

**Hoàn thành khi:**

- [ ] Hình đã chèn vào Overleaf và đọc được khi in đen trắng
- [ ] Mục Analysis đã có bản nháp đủ ý

## Tuần 12 (30/12–05/01): Khóa code và tổng duyệt lần 2 (G4)

Cuối tuần này code bị khóa (05/01), quy trình ngày thi đã chạy trọn vẹn lần thứ hai, và bài báo có bản nháp đầy đủ. Đây là mốc G4.

### Khang: Khóa code và tổng duyệt lần 2

**Việc cần làm:**

- Trước 03/01: nhận các thay đổi cuối qua pull request.
- Ngày 03/01: tổng duyệt lần 2 đúng điều kiện ngày thi, theo `test_day_checklist.md`, bấm giờ thật, nộp lên pha phát triển. Ghi `plan/rehearsal_2.md`.
- Ngày 05/01: khóa code, tạo tag `v1.0-final`, ghi cấu hình ensemble cuối vào `plan/final_ensemble.md`, thêm vào README rằng từ giờ chỉ sửa lỗi gây chặn và phải được cả nhóm đồng ý.
- Viết mục Ensemble (nếu dùng) và khung mục Results: cột điểm test để trống.

**Hoàn thành khi:**

- [ ] Tổng duyệt lần 2 không phát sinh lỗi gây chặn
- [ ] Tag `v1.0-final` và cấu hình ensemble cuối đã có trên repo

### Cương: Bài báo — ghép thành bản nháp hoàn chỉnh

**Việc cần làm:**

- Đọc toàn bộ bài, thống nhất thuật ngữ, văn phong, ký hiệu.
- Kiểm tra theo quy định SemEval khi công bố (bản Word ghi: tối đa 5 trang nội dung không tính tài liệu tham khảo và phụ lục, đúng mẫu tiêu đề, trích dẫn bài mô tả nhiệm vụ, ghi tên tác giả, đường dẫn mã nguồn nếu công bố code). Lưu ý: trước khi công bố code phải xác nhận quyền phân phối dữ liệu; không đưa dữ liệu lên repo.
- Đánh dấu mọi chỗ còn thiếu (điểm test, thứ hạng) bằng `\todo{}`; gửi PDF cho cả nhóm đọc trước buổi họp.

**Hoàn thành khi:**

- [ ] PDF nằm trong giới hạn trang
- [ ] Mọi chỗ còn thiếu đã đánh dấu `\todo{}`

### Đạt: Gói suy luận trên Kaggle

**Việc cần làm:**

- Kaggle notebook dự đoán với Dataset chứa 12 model ZIP gắn sẵn: chỉ cần đưa file test vào là chạy. Đo thời gian trên file cỡ test.
- Chạy thử một phần LLM bằng tài khoản của mình theo hướng dẫn của Phát (nếu giữ L), xác nhận kết quả khớp.
- Ghi số giờ GPU Kaggle còn lại của tài khoản trong tuần thi vào `test_day_checklist.md` (Kaggle cấp quota theo tuần; kiểm tra mức hiện tại).

**Hoàn thành khi:**

- [ ] Notebook dự đoán chạy hết trên file cỡ test
- [ ] (L) phần LLM chạy trên tài khoản của Đạt cho kết quả khớp

### Phát: Chuẩn bị chạy LLM chia phần (nếu giữ L), nếu không thì người dự bị

**Việc cần làm:**

- **Nếu giữ L:** chia file đầu vào thành k phần theo ID và gộp lại (kiểm tra đủ dòng, không trùng ID); notebook dùng chung với tham số là số thứ tự phần, mô hình nạp từ Kaggle Dataset; chạy thử trên dev với 3 tài khoản và so với chạy một máy; khóa prompt và cấu hình (ghi hash file prompt).
- **Nếu không:** chạy thử toàn bộ `test_day_checklist.md` bằng tài khoản của mình như người dự bị cho Đạt, ghi chỗ vướng.

**Hoàn thành khi:**

- [ ] (L) ba phần chạy trên ba tài khoản, gộp khớp kết quả một máy
- [ ] Thời gian dự kiến cho tập test đã ghi vào `test_day_checklist.md`

## Tuần 13 (06/01–15/01): Chạy trên tập test và nộp bài

Tập test dự kiến phát hành 10/01 (Chủ nhật; chưa xác nhận). Nhóm nộp bài chính trước 23:59 ngày 12/01 và các bài phụ trước 15/01, sớm hơn hạn chót 31/01 hơn hai tuần. Tuần này dài 10 ngày.

### Khang: Điều phối ngày thi và nộp bài

**Việc cần làm:**

- 06–09/01: kiểm tra quota GPU từng người, đọc thông báo mới trên Codabench, nhắc lại lịch.
- 10/01 ngay khi test phát hành: tải file, kiểm tra file test, gửi cho cả nhóm, ghi giờ bắt đầu.
- Nhận `test_<hệ thống>.csv`, kiểm tra: đủ dòng, không NaN, tương quan giữa các hệ thống trên test gần với trên dev. Chạy `ensemble.py` (nếu dùng), `write_submission`, `submission_zip`.
- Nộp ứng viên chính (đã chốt ở G3) trước 23:59 ngày 12/01; các ứng viên khác trước 15/01 nếu còn lượt.
- Lưu bằng chứng vào `CompartMent/submissions/` trên Drive: file đã nộp, ảnh chụp trang xác nhận, log. Báo cả nhóm.

**Hoàn thành khi:**

- [ ] Codabench xác nhận bài nộp chính trước 12/01
- [ ] Mọi file nộp và ảnh chụp xác nhận đã lưu trên Drive

### Cương: Hỗ trợ ngày thi và cập nhật bài báo

**Việc cần làm:**

- Nếu giữ E: chạy notebook E trên file test, kiểm tra đủ dòng, không NaN, phân bố dự đoán giống dev; gửi cho Khang trước 21:00 ngày 11/01.
- Nếu không: kiểm tra chéo file dự đoán của Đạt và file nộp của Khang bằng `validate_submission`.
- Sau khi nhóm nộp xong: sửa bài báo theo góp ý tuần 12. Giữ `\todo{}` ở chỗ điểm test, vì ban tổ chức công bố kết quả sau 31/01.

**Hoàn thành khi:**

- [ ] File/kiểm tra giao đúng hạn
- [ ] Mọi góp ý của tuần 12 đã xử lý

### Đạt: Chạy B1/B2 trên test

**Việc cần làm:**

- 10–11/01: chạy notebook dự đoán trên file test, kiểm tra như Cương, gửi `test_ft.csv` cho Khang trước 21:00 ngày 11/01. Sẵn sàng chạy phần LLM được giao (nếu giữ L).
- Lưu log chạy và phiên bản thư viện lên Drive, để bài báo ghi đúng môi trường.

**Hoàn thành khi:**

- [ ] `test_ft.csv` qua mọi kiểm tra và giao đúng hạn
- [ ] (L) phần LLM được giao đã chạy xong

### Phát: LLM trên test (nếu giữ L) và phần kết luận

**Việc cần làm:**

- 10/01: chia phần và khởi động notebook LLM trên các tài khoản theo checklist. Theo dõi, chạy lại phần bị ngắt (cache giữ phần đã chạy). Gộp, kiểm tra, gửi `test_llm.csv` cho Khang trước 12:00 ngày 12/01; nếu không kịp, báo ngay để Khang nộp bài chính không có LLM.
- Sau khi nộp: viết Conclusion và Limitations (~0,3 trang): English-only, heuristic alignment, bất đồng giữa người chấm, ít dữ liệu/seed, giới hạn đối chứng.

**Hoàn thành khi:**

- [ ] (L) `test_llm.csv` đủ dòng và đã giao
- [ ] Conclusion và Limitations đã có trên Overleaf

Sau 15/01, nhóm chờ kết quả chính thức (sau 31/01), điền điểm test vào bài báo, và nộp bài vào tháng 2/2027 theo hạn chính thức sẽ công bố trên semeval.github.io/SemEval2027.

## Rủi ro và phương án dự phòng

Rủi ro lớn nhất là luật cuộc thi khác với giả định của kế hoạch; tuần 1 trả lời phần lớn câu hỏi này. Mỗi rủi ro có sẵn phương án để nhóm không phải bàn lại từ đầu.

| Rủi ro | Dấu hiệu sớm | Phương án |
|---|---|---|
| Luật không cho dùng LLM | `task_facts.md` câu 8 | Bỏ hướng L; Phát làm đọc lỗi và nhãn hiện tượng sớm hơn, các việc LLM ở tuần 9, 12, 13 đổi sang chạy suy luận B1/B2 hoặc backbone thứ hai; bỏ phần LLM trong bài |
| Official metric hoặc phạm vi tính khác giả định | `task_facts.md` câu 1, scorer chính thức | `evaluate` thêm tham số `scope`; đổi chỉ số chọn checkpoint nếu official là trong-expression thì ưu tiên biến thể loss xếp hạng của B2; báo riêng official score và các diagnostics |
| English-only không đủ để xếp hạng/nộp | `task_facts.md` câu 2 | Quyết định ở G1. Việc tiếng Đức (tách từ ghép, chính tả lịch sử, encoder đa ngữ, gbert) nằm sẵn trong bản Word gốc; thêm lại sẽ lùi các mốc 2–3 tuần, cần bàn lại phạm vi |
| Hết quota GPU | Bảng thí nghiệm ghi giờ GPU mỗi tuần | Mỗi người một tài khoản Kaggle; chạy NN trước PV; bỏ backbone lớn; giảm seed xuống 42/43. BERT-base vốn nhẹ nên rủi ro thấp |
| Mất checkpoint (saved version không giữ files; exporter mới chưa xác minh) | Tuần 3: Output của saved version thiếu file | Xác minh ở tuần 3; tải về ngay sau mỗi run; đối chiếu `ARTIFACTS.json`; chưa có thì train lại run đó |
| Một thành viên chậm tiến độ hoặc bận thi | Việc tuần chưa xong khi họp thứ Ba | Notebook độc lập nên ai cũng chạy lại được; ensemble chạy khi thiếu một hệ thống; tuần 11 là tuần đệm; báo trước ít nhất 2 ngày để chia lại việc |
| File test khác định dạng hoặc phát hành trễ | Thông báo trên Codabench | Hàm kiểm tra file test và hai buổi tổng duyệt; nếu test ra trễ, lùi mốc tuần 13, vẫn còn đệm đến 31/01 |
| Test có nhiều expression đã gặp trong train | `task_facts.md` câu 4 | Không chia lại split cố định; thêm đối chứng seen/unseen (RQ2) thiết kế trước khi xem kết quả; xem lại vai trò B1 (nhớ expression) và trọng số ensemble |
| Đồng bộ code dùng chung giữa nhiều notebook | Hai notebook khác nhau ở `align`/metrics | Chỉ giữ hướng E/L khi G2 quyết định; mỗi lần sửa phải cập nhật tất cả notebook và chạy `pytest -q` (có test notebook syntax) |
| Dữ liệu hoặc results chứa context bị đưa lên Git | `git status` thấy file trong `data/`, `results/` | `.gitignore` đã chặn; báo cáo có trích câu để trong `results/analysis/`; chia sẻ qua Drive riêng tư |
| Ban tổ chức đổi lịch hoặc luật | Forum, email | Khang kiểm tra Codabench ở tuần 1, 6, 8, 13, cập nhật `task_facts.md` và sửa các việc bị ảnh hưởng trong tài liệu này |
