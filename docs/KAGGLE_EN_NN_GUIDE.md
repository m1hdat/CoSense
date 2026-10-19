# Hướng dẫn CoSense EN-NN trên Kaggle

Ngày soạn: **17/10/2026**, múi giờ Asia/Saigon. Dành cho lần đầu dùng Kaggle để chuẩn bị dữ liệu và fine-tune B1/B2.

## 1. Bạn nhận được những file nào?

| File trong repo | Dùng để làm gì? | Có upload không? |
| --- | --- | --- |
| `notebooks/kaggle_en_nn.ipynb` | Notebook tự chứa code: kiểm tra dữ liệu, tokenize, B0, tùy chọn smoke/train, tải output | Import vào **Notebook**, không phải Dataset |
| `dist/kaggle/cosense-en-nn-data.zip` | Gói EN-NN nguồn + membership A/B + manifest/hash | Upload vào **Dataset** sau khi xác minh quyền sử dụng |
| `dist/kaggle/cosense-en-nn-data.verification.json` | Kết quả kiểm tra ZIP local, SHA256 và số dòng | Giữ local để đối chiếu |
| `scripts/package_kaggle_en_nn.py` | Đóng gói lại dữ liệu bằng allowlist, kiểm tra trước khi tạo ZIP | Không cần upload riêng |
| `scripts/kaggle_en_nn.py` | Code chuẩn bị/train/inference được nhúng trong notebook | Không cần upload riêng |
| `scripts/build_kaggle_notebook.py` | Sinh lại notebook sau khi sửa code | Không cần upload riêng |
| `requirements-kaggle.txt` | Dependency pin cho notebook; không thay PyTorch CUDA có sẵn | Đã nhúng trong notebook |

**Notebook và Dataset là hai thứ riêng:** Dataset chứa dữ liệu đầu vào; Notebook chứa code chạy trên máy Kaggle. Không cần clone GitHub, token GitHub, `kaggle.json` hay API trả phí để chạy bằng giao diện web.

Trạng thái kiểm thử: **17 tests local đã đạt**, gồm8 tests pipeline cũ và9 tests artifact/pipeline Kaggle. ZIP CRC/hash, loader cả A/B, B0 tái lập, tách input/nhãn, không export final, chặn truncation và code notebook đã được kiểm tra. **Chưa chạy notebook trên Kaggle GPU**; pip, tokenizer thật, tải backbone, optimizer, checkpoint và resume trên Kaggle cần được xác nhận bằng lượt prepare/smoke thực tế. Không có kết quả B1/B2 được điền sẵn.

## 2. Trước khi upload

Repo hiện chưa xác minh license/điều khoản cho phép đưa dữ liệu cuộc thi lên dịch vụ ngoài. Đọc điều khoản tải dữ liệu hoặc hỏi BTC về sử dụng Kaggle. **Private Dataset giới hạn người xem, không tự cấp quyền sử dụng dữ liệu.** Nếu chưa rõ quyền, bạn vẫn có thể kiểm tra ZIP và code ở local, rồi thực hiện phần upload khi có căn cứ cho phép. Tài liệu này không upload/publish/submission thay bạn.

Không chọn một license mở chỉ để vượt form nếu chưa có quyền cấp lại license đó. Không đưa `.env`, token, credentials hay toàn bộ thư mục project vào Dataset.

Nguồn EN-NN được giữ nguyên. Trial đã gộp theo thông tin của bạn; không tải lại trial để nối thêm. Không sửa nhãn, bỏ stopword/dấu câu, stemming, lemmatize hoặc loại mẫu chỉ vì không exact-match được cụm trong câu.

## 3. Kiểm tra hoặc tạo lại gói local

Mở PowerShell tại thư mục project. Các lệnh sau chỉ build/verify, **không train, không upload, không cài dependency nặng**:

```powershell
Set-Location 'D:\24tn-fitus\project\SemEval-2027-Task5'
python scripts/build_kaggle_notebook.py
python scripts/package_kaggle_en_nn.py
python -m unittest discover -s tests -v
Get-FileHash -Algorithm SHA256 .\dist\kaggle\cosense-en-nn-data.zip
```

Nếu đã nhận file ZIP được tạo sẵn, không cần chạy lại các bước audit/split. Chỉ rebuild gói khi file đầu vào hoặc gói cần cập nhật. Không chạy `create_splits.py` chỉ vì bạn chuẩn bị mở Kaggle; membership đang khóa phải được giữ nguyên.

Kỳ vọng khi package thành công:

- ZIP có **12 file**, không có code/credentials/EN-PV TSV/tiếng Đức/checkpoint.
- Kiểm tra split A pass18 assertions, B pass20 assertions.
- A: train2443, validation524, final523; B: train2444, validation523, final523.
- Ghi `uploaded: false`, `final_evaluated: false` trong verification JSON.
- Mọi lỗi source/config/membership sẽ chặn đóng gói. Không chữa bằng cách xóa manifest hoặc random split lại.

Cấu trúc **bên trong ZIP**:

```text
bundle_manifest.json
configs/data_protocol.json
data/train/en-nn-train.tsv
data/splits/en_nn/seen_context/
    train.csv
    validation.csv
    final.csv
    manifest.json
data/splits/en_nn/unseen_compound/
    train.csv
    validation.csv
    final.csv
    manifest.json
reports/audit/en_nn_near_candidates.json
```

`final.csv` chỉ chứa membership, phục vụ coverage/replay/integrity. TSV nguồn chứa toàn bộ3490 mẫu nên final gold vẫn có mặt trong nguồn; notebook đọc chúng để kiểm tra validity, **không export final train-ready và không đưa final vào train/chọn checkpoint/metric**. Đây là khóa bằng quy trình và code, không phải mã hóa final.

Config và split manifest gốc vẫn nhắc đến EN-PV để giữ provenance. Bundle này cố ý chỉ kèm EN-NN; notebook chỉ kiểm tra nguồn EN-NN và không gọi lại các CLI audit yêu cầu đủ hai TSV tiếng Anh. Nếu muốn audit lại toàn bộ, làm trong repo gốc.

## 4. Upload Dataset bằng giao diện web

Thực hiện sau bước quyền sử dụng ở mục2. Nhãn nút có thể thay đổi theo giao diện Kaggle; đối chiếu [Kaggle Datasets](https://www.kaggle.com/docs/datasets).

1. Đăng nhập Kaggle bằng tài khoản của bạn.
2. Vào **Datasets**, chọn **New Dataset** hoặc **Create → New Dataset**.
3. Trong phần upload từ máy tính, chọn đúng file:

   ```text
   D:\24tn-fitus\project\SemEval-2027-Task5\dist\kaggle\cosense-en-nn-data.zip
   ```

4. Đặt title ví dụ **CoSense EN-NN v2 Private**. Slug ví dụ `cosense-en-nn-v2-private`; notebook không phụ thuộc slug này.
5. Kiểm tra **Visibility/Sharing = Private**. Giữ cả Notebook private ở các bước sau.
6. Điền mô tả ngắn: EN-NN source, split protocol `cosense-data-v2`, internal validation only, original source unmodified. License/terms phải theo quyền thực tế của BTC.
7. Bấm **Create** và chờ xử lý xong. Đây chỉ là tạo Dataset cá nhân, không phải submission cuộc thi.
8. Mở danh sách file của Dataset. Kaggle thường giải nén ZIP để các file dùng được trong Notebook. Nếu nó giữ nguyên ZIP, notebook cũng đọc được.
9. Không đổi tên file TSV/CSV bên trong hoặc mở bằng Excel rồi Save lại. Việc đổi newline/encoding cũng có thể làm hash khác.
10. Giữ file verification JSON ở máy để đối chiếu. Khi đổi Dataset, tạo version mới và ghi rõ lý do; không âm thầm trộn các bản membership.

Không cần upload `reports/audit/en_nn_eligible.csv`, `en_nn_duplicates.csv`, `en_nn_compound_variation.csv` riêng. Chúng là báo cáo, không phải input trực tiếp cho model. Allowlist đã chứa phần near-candidate cần cho replay split.

Nếu upload web chuyển TSV sang CSV hoặc sửa nội dung, cell integrity sẽ dừng. Đừng cập nhật hash cho khớp file đã bị đổi. Dùng cách upload giữ nguyên file; phương án CLI của Kaggle có `--keep-tabular` để không chuyển bảng sang CSV. Nếu cần CLI, xem mục11; mặc định hướng dẫn dùng web.

## 5. Import notebook và gắn Dataset

1. Vào **Code → New Notebook** hoặc **Create → New Notebook**.
2. Chọn **File → Import Notebook / Upload Notebook** và chọn:

   ```text
   D:\24tn-fitus\project\SemEval-2027-Task5\notebooks\kaggle_en_nn.ipynb
   ```

3. Đặt tên ví dụ `CoSense EN-NN B1 B2 pilot`; kiểm tra sharing vẫn **Private**.
4. Trong sidebar **Input**, chọn **Add Input / Add Data**, tìm Dataset vừa tạo trong phần dataset của bạn và bấm **Add**. [Kaggle hướng dẫn gắn nguồn dữ liệu ở đây](https://www.kaggle.com/docs/notebooks).
5. Bật **Internet = On** trong Session options/Settings để tải dependency/tokenizer/model pretrained lần đầu.
6. Lần prepare đầu chọn **Accelerator = None/CPU** để tiết kiệm GPU. Chỉ bật GPU trước smoke/train.
7. Giữ `DATA_ROOT = ""`. Notebook tự tìm manifest trong `/kaggle/input`, kể cả mount dạng `/kaggle/input/datasets/<owner>/<slug>/...`.
8. Nếu gắn nhiều bản bundle, chỉ giữ bản cần dùng hoặc đặt `DATA_ROOT` bằng **thư mục cha của `bundle_manifest.json`**, không phải đường dẫn đến TSV.

`/kaggle/input` là nguồn chỉ đọc. Notebook ghi dữ liệu dẫn xuất, code nhúng và checkpoint vào `/kaggle/working`. Không nhập đường dẫn `D:\...` vào cấu hình Kaggle.

## 6. Lượt đầu: prepare, chưa dùng optimizer

Ở cell1 giữ:

```python
MODE = "prepare"
SETTING = "unseen_compound"
VARIANTS = ["B1", "B2"]
SEED = 42
MAX_LENGTH = 256
DATA_ROOT = ""
INSTALL_DEPS = True
```

Chọn **Run All**. Mỗi cell có lời giải thích ngay trong notebook:

| Bước | Việc thực hiện | Kỳ vọng |
| --- | --- | --- |
| 1 | Kiểm tra config, tạo output folder riêng | In mode/setting/output |
| 2 | Cài transformers/hub/safetensors đã pin, kiểm tra torch | Không cài lại torch; CUDA=False được chấp nhận với prepare |
| 3 | Ghi/import code đã nhúng và kiểm tra hash | In hash hai source Python |
| 4 | Tìm Dataset, kiểm tra file/schema/leakage/replay | B train2444, validation523, final523; B0 E≈0.670054 |
| 5 | Resolve model SHA, tokenize B1/B2, lưu tokenizer/length report | `truncated=0`; có prepared và token JSONL |
| 6 | So sánh và tùy chọn train | In thông báo không train khi prepare |
| 7 | Tùy chọn reload checkpoint | Prepare bỏ qua vì chưa có checkpoint |
| 8 | Zip output và tạo link tải | Có ZIP cho lần chạy này |

Chuẩn bị dữ liệu ở đây nghĩa là nối membership với TSV theo khóa mẫu, tách **inputs/labels/metadata**, tokenize và padding động khi batching. Không train tokenizer riêng và không fit gì trên final.

File JSONL prepared có dạng minh họa sau; các chuỗi/số chỉ để giải thích schema, không phải gold thật:

```json
{"metadata":{"row_key":"en_nn:example","ContextID":"example"},"inputs":{"text":"coffee cup","text_pair":"She placed the coffee cup on the table."},"labels":[3.0,4.0]}
```

`labels` không được nối vào văn bản. B1 chỉ có `inputs.text`; B2 thêm `inputs.text_pair`. `ContextID`, `ModAvg`, `HeadAvg`, `ModStd`, `HeadStd` không được đưa vào feature. Token JSONL chỉ có `input_ids`, `attention_mask`, `labels`; giữ thứ tự membership khi xuất predictions.

**Ghi lại `resolved_revision`** từ `model_lock.json`, rồi thay `MODEL_REVISION = "main"` bằng SHA đó cho smoke/train/các seed sau. SHA khóa đúng phiên bản pretrained; B1/B2 trong một lượt luôn dùng chung revision. Tokenizer `uncased` xử lý chữ hoa/thường theo model; nguồn TSV không bị lowercase lại.

Nếu length vượt256, pipeline dừng; kiểm tra các mẫu bị chặn và chọn max length phù hợp cho cả hai model, không tự cắt một phần context hoặc chỉ tăng length riêng cho model đang thua. Không vượt giới hạn512 của DistilBERT.

## 7. Lượt thứ hai: smoke có optimizer trên GPU

Sau prepare thành công, tải output và lưu Notebook Version private. Bật **Accelerator = GPU** trong Settings/Session options, chọn loại GPU tài khoản được cấp. Bật accelerator thường làm session restart; chạy lại từ đầu. Nếu Kaggle yêu cầu xác minh tài khoản để cấp GPU, hoàn thành theo giao diện của Kaggle; nếu chưa có GPU thì dừng ở prepare.

Đổi cell1:

```python
MODE = "smoke"
SETTING = "unseen_compound"
VARIANTS = ["B1", "B2"]
SEED = 42
SMOKE_STEPS = 8
# MODEL_REVISION giữ SHA đã ghi từ lượt prepare.
```

Giữ Internet On. Run All. Notebook dùng một GPU (`cuda:0`), FP32; T4 x2 vẫn chỉ dùng GPU0. Mỗi model học tối đa128 mẫu train trong tối đa8 bước, một epoch. Validation vẫn dùng toàn bộ523 mẫu để kiểm tra đường đi evaluation.

Tiêu chí smoke đạt:

- CUDA=True, có tên GPU, không có loss/prediction NaN hoặc Inf.
- B1/B2 xuất hai số cho mỗi mẫu; ghi được history, best/last checkpoint và validation predictions.
- Cell reload báo **B1 reload/inference PASS**, **B2 reload/inference PASS**.
- ZIP chứa output của đúng lần chạy. Smoke có thể có sai số lớn và chưa hơn B0; mục tiêu là xác minh pipeline.

Không dùng kết quả smoke để kết luận model nào tốt hơn hoặc chọn seed. Không có cam kết thời gian vì còn phụ thuộc tải model/runtime/GPU được cấp.

## 8. Lượt thứ ba: fine-tune đầy đủ

Sau smoke đạt, đổi cell1:

```python
MODE = "train"
SETTING = "unseen_compound"
VARIANTS = ["B1", "B2"]
SEED = 42
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
EPOCHS = 5
PATIENCE = 2
# MODEL_REVISION = SHA cố định giống prepare/smoke.
```

Run All. Mỗi model dùng toàn bộ train của setting đó. Kiến trúc hai model: DistilBERT → masked mean pooling loại padding, có special tokens → linear2 output. Full fine-tune cập nhật cả backbone và head; đây không phải pretrain từ đầu. Loss MSE; AdamW; weight decay0.01; warmup10%/linear decay; clip gradient1.0. Validation chọn checkpoint bằng compound-macro two-output MAE sau clipping prediction về[0,5], không chọn bằng train loss.

Mặc định tối đa5 epochs, dừng sớm nếu2 epochs liên tiếp không cải thiện. Đó là cấu hình pilot, không đảm bảo tối ưu. Lưu cả raw và clipped predictions, MAE/RMSE/tương quan từng đầu, E tổng và tỷ lệ output ngoài[0,5].

Kết quả cần đọc trong `comparison_validation.json`:

- B0 E≈0.670054 trên B là mốc tham chiếu, càng thấp càng tốt.
- So B1 với B0 để xem fine-tune chỉ dùng cụm có cải thiện không.
- So B2 với B1 để khảo sát lợi ích context. Delta=`E(B1)-E(B2)` dương thì B2 ít sai hơn.
- Một seed không chứng minh cải thiện ổn định. Đọc lỗi validation rồi chạy seed13,2027 cho so sánh quan trọng.

Sau B, đổi `SETTING = "seen_context"` để chạy A (train2443, validation524); B0 A≈0.745341. So sánh các model trong cùng setting. A/B dùng chung nguồn, có thể có final A trong train B và ngược lại; khóa lựa chọn chung trước khi mở bất kỳ final nào.

Nếu tuning, dùng grid1e-5,2e-5,3e-5, tối đa3 trial/model/setting với seed42, budget công bằng, lưu mọi trial/status. Notebook chỉ chạy một LR/seed/setting mỗi lượt, không tự grid search hoặc tự ensemble. Metric ở đây là nội bộ; không nhận là metric chính thức của cuộc thi.

## 9. Output, tải về và resume

Mỗi Run All sinh output folder có timestamp UTC để phân biệt, ví dụ:

```text
/kaggle/working/cosense_outputs/unseen_compound_train_seed42_<run_tag>/
    integrity.json
    environment.json
    code_hashes.json
    model_lock.json
    b0_validation.json
    comparison_validation.json
    token_lengths.json
    tokenizer/
    code/                       # đúng code tạo artifact này
    prepared/{B1,B2}/{train,validation}.jsonl
    tokens/{B1,B2}/{train,validation}.jsonl
    runs/B1/                    # tương tự runs/B2/
        config.json
        integrity.json
        environment.json
        status.json
        history.json
        metrics.json
        validation_predictions.tsv
        backbone_config/
        tokenizer/
        best.safetensors
        last.pt
    reload_check.json
```

`best.safetensors` là trọng số inference, gồm cả backbone và regression head. `last.pt` chứa optimizer/scheduler/RNG để resume, thường lớn hơn best. Không dùng `AutoModel.from_pretrained(run_dir)` rồi bỏ quên custom head; dùng `load_for_inference`/`predict_texts` đã đi kèm code.

Cell cuối tạo một ZIP cạnh folder output, in SHA256 và link tải. Nếu FileLink không mở được, dùng phần **Output/Files** của Kaggle để tìm ZIP theo tên vừa in rồi Download. Đối chiếu kích thước/hash khi cần. Prepared data/predictions chứa gold nên tiếp tục giữ private.

**Lưu version:** chọn Save Version, kiểm tra notebook vẫn private. Nếu UI cho lựa chọn Quick Save so với Save & Run All, chọn cách phù hợp để giữ kết quả đã chạy; Save & Run All sẽ thực thi lại notebook trong một phiên mới và có thể tốn thêm GPU. Chờ phiên/version hoàn tất, kiểm tra tab Output có ZIP trước khi kết thúc session. Không mặc định file trong live working sẽ còn sau mọi lần reset. Xem cơ chế lưu output trong [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

**Nếu bị ngắt khi train:**

1. Tìm run folder có `last.pt`, `best.safetensors`, `history.json`, config/tokenizer/backbone_config. Nếu chưa hoàn thành epoch đầu, chưa có checkpoint resume; chạy lại từ đầu.
2. Tải output/giữ một version private trước khi bỏ session nếu còn truy cập được. Có thể chạy lại cell đóng ZIP sau khi training lỗi nếu biến OUTPUT vẫn còn.
3. Gắn output version trước bằng Add Input → Notebook output, hoặc upload trọn run folder thành một Dataset private khi quyền sử dụng cho phép.
4. Mở session mới, giữ nguyên MODE/setting/variant/seed/LR/batch/length/epochs/patience/model SHA và phiên bản code/dependency của lần cũ.
5. Điền ví dụ (đường dẫn chỉ minh họa, lấy mount thực tế từ Input):

   ```python
   VARIANTS = ["B1"]
   RESUME_FROM = {"B1": "/kaggle/input/<previous-output>/<run-folder>/runs/B1/last.pt", "B2": ""}
   ```

6. Giữ `best.safetensors` cạnh `last.pt`. Notebook ghi sang output mới, không sửa input cũ. Không resume smoke thành train hoặc tăng epochs rồi giả định là cùng run.
7. Resume bắt đầu sau **epoch hoàn thành gần nhất**; phần epoch đang chạy lúc bị ngắt sẽ chạy lại. Kết quả bit-for-bit còn phụ thuộc runtime/GPU; code chỉ kiểm tra config/data/code/torch/transformers giống nhau.
8. Nếu checkpoint đã đủ epochs hoặc đạt early stopping thì không cần resume; dùng best checkpoint đã có.

Chỉ load `last.pt` do chính notebook này tạo và bạn kiểm soát; đó là checkpoint PyTorch có trạng thái Python. Nếu lần trước B1 đã hoàn tất còn B2 lỗi, chạy chỉ B2 bằng `VARIANTS=["B2"]`; không cần train lại B1 để sửa lỗi B2. Khi tổng hợp, dùng artifact cùng setting/seed/config để đối chiếu.

## 10. Lỗi thường gặp

| Hiện tượng | Kiểm tra/cách xử lý |
| --- | --- |
| Không tìm thấy bundle | Dataset chưa Add Input; upload nhầm notebook vào Dataset; hoặc file đã bị đổi tên. Tìm `bundle_manifest.json`; đặt DATA_ROOT bằng thư mục cha |
| Multiple bundles attached | Gỡ các version không dùng hoặc chỉ rõ DATA_ROOT |
| Missing/changed file/hash mismatch | Đối chiếu ZIP/verification; giữ TSV và newline/encoding gốc; tránh Kaggle CSV conversion; không sửa hash để bỏ qua lỗi |
| Stale manifest/config/replay failed | Bundle đang trộn version. Rebuild từ repo gốc đã kiểm tra; không random split lại trên Kaggle |
| pip/Hugging Face network lỗi | Bật Internet, kiểm tra trạng thái dịch vụ, chạy lại với cùng model SHA. Notebook chưa hỗ trợ tự dựng môi trường offline |
| CUDA=False ở smoke/train | Bật GPU accelerator, chờ cấp GPU và restart/Run All; không chuyển optimizer sang local CPU |
| Dependency/version/import lỗi | Dùng session mới với requirements đã pin; không cài CPU torch. Nếu vẫn lỗi, giữ traceback và log environment để chẩn đoán; pin hiện tại chưa được xác nhận trên mọi image |
| CUDA out of memory | Restart để giải phóng model cũ; giảm BATCH_SIZE cho cả B1/B2 và ghi deviation, chạy lại fair comparison. Implementation chưa có gradient accumulation; đổi batch sẽ đổi effective batch |
| Token length vượt MAX_LENGTH | Không tự cắt bỏ mục tiêu; kiểm tra length/cấu hình cho cả hai model, tối đa giới hạn backbone |
| Run directory exists | Bạn chạy lại cell train trên cùng OUTPUT. Chạy từ cell1 để sinh RUN_TAG mới; không ghi đè checkpoint |
| Resume config mismatch | Giữ code/config/model SHA/data và runtime tương thích; nếu đổi phương pháp thì là run mới |
| Smoke MAE cao/chưa hơn B0 | Chưa đủ bước học; kiểm tra pipeline/reload trước. Không tuning theo kết quả final |
| Không có checkpoint khi MODE=prepare | Đúng thiết kế; prepare chỉ tokenize và B0 |
| Mất output sau reset | Tải ZIP hoặc lưu private Notebook Version có output trước khi kết thúc session |

## 11. Tùy chọn CLI nếu cần tránh chuyển TSV sang CSV

Không cần phần này nếu upload web giữ nguyên hash. Chỉ dùng khi bạn đã có quyền upload và muốn dùng Kaggle CLI chính thức. Cài/auth CLI theo [Kaggle CLI](https://github.com/Kaggle/kaggle-cli); credentials để ở vị trí riêng của tài khoản, không trong repo/ZIP/notebook.

Các lệnh bên dưới là hướng dẫn, không được notebook thực thi. Tạo một thư mục staging chỉ chứa ZIP gốc và metadata của CLI. Với PowerShell tại repo (nếu folder đã tồn tại, kiểm tra nó chỉ chứa file dành cho upload):

```powershell
New-Item -ItemType Directory -Path .\dist\kaggle\upload-en-nn
Copy-Item -LiteralPath .\dist\kaggle\cosense-en-nn-data.zip -Destination .\dist\kaggle\upload-en-nn\cosense-en-nn-data.zip
kaggle datasets init -p .\dist\kaggle\upload-en-nn
```

Sửa `dataset-metadata.json` mới được tạo: title, id=`<username>/cosense-en-nn-v2-private`, license đúng quyền thực tế. Không tự gán CC0/CC-BY khi chưa được phép. Nếu không xác định được license phù hợp, giải quyết với BTC trước upload.

Sau khi metadata đúng:

```powershell
kaggle datasets create -p .\dist\kaggle\upload-en-nn --keep-tabular
```

Không thêm `--public`/`-u`. Theo [CLI datasets documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md), mặc định create là private, `--keep-tabular` giữ bảng thay vì chuyển sang CSV. Staging chỉ chứa gói ZIP nguyên bản nên không cần upload thư mục con hay tách thành nhiều ZIP. Sau khi attach, notebook đọc được cả bundle đã giải nén và `cosense-en-nn-data.zip` còn nguyên. Mọi hash vẫn phải khớp.

Không upload cả thư mục project. Không đưa `dataset-metadata.json` vào bundle gốc rồi thay allowlist; file này chỉ phục vụ CLI staging.

## 12. Cập nhật code mà giữ nguyên dữ liệu

Sửa `scripts/kaggle_en_nn.py` hoặc `scripts/data_utils.py` → chạy `python scripts/build_kaggle_notebook.py` → chạy tests → import notebook mới lên Kaggle. Notebook đã nhúng code nên không dùng code từ GitHub hay code cũ trong Dataset.

Nếu chỉ sửa code, không cần tạo lại Dataset. Nếu đổi source/config/membership, cần audit/review lại protocol, rebuild bundle và version hóa quyết định; không đổi data giữa chừng khi so B1/B2. Lưu code hashes và model revision để nhận biết đúng artifact.

Tham khảo kỹ thuật: [HF Auto Classes4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/auto), [padding/truncation4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/pad_truncation), [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased). Đây là các API/model được chọn cho pilot, không phải khẳng định backbone tốt nhất cho CoSense.
