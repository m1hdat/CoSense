# CoSense — Kế hoạch project SemEval 2027 Task 5: ComPartMent

Ngày khảo sát: **14/10/2026** (Asia/Saigon). Đối tượng đọc: sinh viên mới bắt đầu NLP.

**Trạng thái cập nhật 17/10/2026:** M1 tiếng Anh đã hoàn thành: audit EN-NN/EN-PV, split EN-NN A/B và kiểm tra tái lập; M2 B0 validation đã chạy. Đã có [DATA_REPORT.md](DATA_REPORT.md), [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), [notebook Kaggle](../notebooks/kaggle_en_nn.ipynb) và [hướng dẫn upload/prepare/smoke/train](KAGGLE_EN_NN_GUIDE.md). Code B1/B2 đã chuẩn bị và kiểm tra local; chưa chạy Kaggle GPU hoặc có kết quả neural. Demo, submission và paper chưa triển khai. Dữ liệu nguồn giữ nguyên; chưa đánh giá final. Tài liệu là kế hoạch cập nhật, không cam kết kết quả nghiên cứu.

**Nơi thực hiện đã chốt:** neural training trên **Kaggle GPU**, gồm smoke có optimizer, fine-tune B1/B2, tuning và train lại nghiên cứu. Local chuẩn bị/audit/split/code, phân tích kết quả và demo. B0 không cần GPU; đã chạy local theo yêu cầu giai đoạn audit/split này, có thể tái chạy trong notebook Kaggle cùng config/membership. Chưa bắt đầu train dài.

**Phạm vi hiện tại:** chỉ tiếng Anh. Audit cả EN-NN/EN-PV; split/B0 và neural pipeline đầu tiên cho EN-NN. Tiếng Đức tạm ngoài phạm vi triển khai. Protocol hiện hành dùng B1 **chỉ Compound**, B2 **Compound+Context**, thay cho đề xuất thêm Mod/Head riêng ban đầu.

## 1. Bài toán và những sản phẩm khác nhau

[Trang chính thức ComPartMent](https://sites.google.com/view/semeval-compartment) mô tả nhiệm vụ: nhận một biểu thức nhiều từ và câu chứa biểu thức đó, dự đoán điểm số phản ánh đánh giá hợp thành của con người trong ngữ cảnh. Có noun compounds (NN, danh từ ghép) và particle verbs (PV, động từ đi với tiểu từ), ở tiếng Anh và tiếng Đức. Trang giới thiệu có dữ liệu hiện đại và lịch sử; điều này **không cung cấp nhãn thời gian cho từng dòng TSV đang có**.

Theo schema thực tế trong workspace:

| Nhánh | Đầu vào dùng cho mô hình | Nhãn cần dự đoán | Không đưa vào đặc trưng inference |
| --- | --- | --- | --- |
| NN | `Context`, `Compound`, `Mod`, `Head` | `ModAvg`, `HeadAvg`, hai số trên thang 0–5 | `ContextID`, hai cột Avg, `ModStd`, `HeadStd` |
| PV | `Context`, `ParticleVerb`, `Base`, `Particle` | `Avg`, một số trên thang 0–5 | `ContextID`, `Avg`, `Std` |

`ContextID` chỉ dùng nối dự đoán với dòng nguồn, kiểm tra và xuất file; không học từ ID. Các cột Std mô tả độ phân tán đánh giá, không phải thông tin biết được ở test. Điểm trung bình là số thực dù đánh giá nền là ordinal (có thứ tự); không chuyển bài toán thành phân loại sáu lớp ngay từ đầu. Cách diễn đạt chính xác hai đầu thang trong hướng dẫn annotation vẫn cần đối chiếu BTC; kế hoạch dùng cách hiểu làm việc: điểm cao biểu thị thành phần đóng góp nghĩa thông thường nhiều hơn.

**Ví dụ tự tạo để giải thích, không phải mẫu gold và không có điểm gold:**

- “She placed the **coffee cup** on the table.” Trong `coffee cup`, modifier `coffee` bổ nghĩa loại/mục đích; head `cup` thường quyết định loại vật được nói đến. Có thể giải thích nghĩa toàn cụm từ nghĩa các thành phần: tính hợp thành (compositionality) tương đối rõ.
- “She looked through the **glass ceiling**.” và “The promotion policy created a **glass ceiling**.” Cùng chuỗi có thể nói về trần bằng kính hoặc rào cản nghề nghiệp. Context sensitivity là độ nhạy của dự đoán với cách dùng trong câu. Không tự gán hai điểm số hay khẳng định mọi cặp như vậy đều có độ hợp thành khác nhau theo annotation BTC.
- “The plane **took off**.” và “He **took his coat off**.” `take off` có thể biến hình và bị tách bởi tân ngữ. PV không thể chỉ tìm một chuỗi nguyên dạng liên tiếp.

Modifier và head có vai trò khác nhau nên không ép `ModAvg = HeadAvg`. Từ “head” trong bài này là thành phần ngôn ngữ; “regression head” ở phần huấn luyện là lớp xuất điểm của mạng, hai khái niệm khác nhau.

Phân biệt bốn sản phẩm:

1. **Mô hình ML:** học ánh xạ đầu vào → điểm từ dữ liệu train; checkpoint chứa trọng số đã học.
2. **Web demo:** giao diện gọi inference của checkpoint, phục vụ trình bày trên lớp; giao diện đẹp không chứng minh mô hình tốt.
3. **Submission:** dự đoán trên dữ liệu cuộc thi theo template và luật BTC; kết quả split nội bộ không phải điểm test cuộc thi.
4. **Báo cáo/paper:** câu hỏi, đối chứng, thí nghiệm tái lập và kết luận có giới hạn. Pipeline chạy được chưa đủ chứng minh tính mới hoặc khả năng publish.

## 2. Khảo sát repository, môi trường và dữ liệu thực tế

### 2.1 Repository và môi trường

Repository hiện có `README.md` chỉ chứa tiêu đề, bốn file trong `data/train/`, và thư mục `data/evaluation/` rỗng. Không tìm thấy `AGENTS.md` trong repository hoặc các thư mục cha đã kiểm tra (`D:/`, `D:/24tn-fitus/`, `D:/24tn-fitus/project/`). Không thấy code, notebook, cấu hình dependency, `.gitignore`, giấy phép hay hướng dẫn bổ sung trong cây file khảo sát. Trước khi tạo tài liệu, `git status --short` cho thấy `data/` chưa được Git theo dõi; không tự stage/commit dữ liệu.

Môi trường được kiểm tra trực tiếp, không suy đoán từ loại máy:

| Hạng mục | Kết quả |
| --- | --- |
| Hệ điều hành/shell | Windows, PowerShell |
| Python trên PATH | `C:/Users/Dell/miniconda3/python.exe`, Python 3.14.6 |
| Số luồng CPU Python nhìn thấy | 8 |
| Dependency đã có | torch 2.13.0+cpu, transformers 5.14.1, pandas 3.0.3 |
| CUDA của PyTorch | `torch.cuda.is_available() = False`, số GPU = 0 |
| Dependency chưa tìm thấy | `datasets`, `streamlit` |
| Công cụ NVIDIA trên PATH | không tìm thấy `nvidia-smi` |
| CPU cụ thể, RAM, GPU vật lý | truy vấn CIM bị hệ điều hành từ chối; chưa xác minh |

PyTorch hiện tại chạy CPU; kết quả này không chứng minh máy hoàn toàn không có GPU NVIDIA. Local phục vụ EDA, split và demo; môi trường huấn luyện là runtime Kaggle GPU. Chưa kiểm tra GPU, quota, VRAM hay phiên bản runtime Kaggle thực tế. M1 dùng thư viện chuẩn Python là đủ. M3 kiểm tra và khóa dependency trên Kaggle; M5 tạo môi trường inference riêng ở local với phiên bản tương thích checkpoint, thay vì thay đổi môi trường base. Không coi Python 3.14 hay các package local có sẵn là cấu hình cho notebook huấn luyện.

### 2.2 Schema, số mẫu và số cụm

Đã đọc TSV bằng `csv.DictReader`, delimiter tab, encoding `utf-8-sig`; số dòng dưới đây là **record dữ liệu, không tính header**. Cụm duy nhất tính theo giá trị trường gốc, chưa lemmatize hay gộp biến thể.

| File trong `data/train/` | Record | Cụm duy nhất | Ngữ cảnh/cụm: min / median / max |
| --- | ---: | ---: | --- |
| `en-nn-train.tsv` | 3.490 | 446 | 1 / 8 / 10 |
| `en-pv-train.tsv` | 1.567 | 160 | 5 / 10 / 10 |
| `de-nn-train.tsv` | 3.298 | 461 | 5 / 6 / 10 |
| `de-pv-train.tsv` | 1.490 | 158 | 5 / 10 / 10 |
| Tổng | **9.845** | **1.225 mục theo từng nhánh** | không gộp thành một phân phối chung |

NN có đúng thứ tự cột: `ContextID, Compound, Mod, Head, ModAvg, ModStd, HeadAvg, HeadStd, Context`.

PV có đúng thứ tự cột: `ContextID, ParticleVerb, Base, Particle, Avg, Std, Context`.

Phân bố số record mỗi cụm (`số ngữ cảnh: số cụm`):

- EN-NN: `1:1, 3:2, 4:1, 5:129, 6:22, 7:51, 8:21, 9:13, 10:206`; cụm chỉ có một mẫu là `viscosity work`.
- EN-PV: `5:3, 6:2, 7:2, 8:2, 10:151`.
- DE-NN: `5:215, 6:18, 7:31, 8:27, 9:18, 10:152`.
- DE-PV: `5:9, 6:5, 7:4, 8:4, 9:5, 10:131`.

Không có ZIP, thư mục trial riêng hoặc bộ train gốc thứ hai trong workspace đã khảo sát. Theo người dùng, trial đã gộp vào train; chỉ nạp bốn file hiện tại bằng manifest tường minh, không glob mọi ZIP/TSV về sau. Không có bản gốc và manifest gộp nên **chưa thể xác minh nguồn gốc từng dòng hoặc cách gộp**. Trial không còn là đánh giá độc lập. Con số tổng quan 1.539 biểu thức trên trang cuộc thi không phải số cụm train local; không tự suy ra các mục thiếu là lỗi dữ liệu.

### 2.3 Chất lượng dữ liệu và điểm

Trong cả bốn file: không có ô rỗng/whitespace ở các cột, không có record sai số trường, không có ID lặp trong file hoặc giữa file, không có dòng trùng toàn bộ, không có cặp `(cụm, câu chuẩn hóa)` trùng. Tất cả Avg/Std parse được thành số hữu hạn. Không thấy ký tự thay thế Unicode U+FFFD. Kiểm tra này không chứng minh văn bản/annotation không có lỗi ngôn ngữ.

| File | Khoảng Avg quan sát | Trung bình toàn file (chỉ EDA) | Khoảng Std |
| --- | --- | --- | --- |
| EN-NN | Mod 0–5; Head 0–5 | Mod 3,65983; Head 3,91230 | Mod 0–2,54733; Head 0–2,58199 |
| EN-PV | 0,1–5 | 2,46278 | 0–2,40601 |
| DE-NN | Mod 0,1–5; Head 0,1–5 | Mod 4,31746; Head 3,59774 | Mod 0–2,58199; Head 0–2,63523 |
| DE-PV | 0,1–4,9 | 2,62810 | 0,316228–2,51440 |

Các trung bình ở bảng **không được dùng làm B0 trên validation/test**; B0 phải tính lại chỉ từ train của từng split.

Chuẩn hóa câu để kiểm tra trùng: `casefold()` và gộp whitespace, giữ dấu câu. EN-NN có 4 nhóm câu trùng, mỗi nhóm 2 dòng cho cụm khác nhau (4 dòng dư); EN-PV có 1 nhóm 2 dòng (1 dòng dư). Hai file tiếng Đức không có câu trùng theo phép chuẩn hóa này. Không thấy câu chuẩn hóa trùng giữa các file.

Các cặp ID EN-NN phải được xét cùng nhóm chống rò rỉ:

- `11382 / 13476`: `glass cell / zinc plate`.
- `13134 / 13458` và `13142 / 13457`: `summer temperature / winter temperature`.
- `13316 / 13333`: `unit area / unit time`.

Cặp EN-PV: `30312 / 31455`, `move down / move up`. Đây có thể là cùng câu có nhiều mục tiêu hợp lệ; **không tự xóa một dòng chỉ vì câu giống nhau**.

Đã sàng lọc gần trùng sơ bộ EN-NN bằng Jaccard trên tập token `\w+` sau casefold, ngưỡng ≥0,90, loại cặp câu chuẩn hóa bằng nhau: không có cặp đạt ngưỡng. Đây là phép kiểm tra hẹp, mất thứ tự và số lần token; chưa kiểm tra gần trùng ba nhánh khác, chưa kiểm tra bằng character n-gram hoặc paraphrase. M1 cần mở rộng kiểm tra và đọc các cặp đáng nghi trước khi cố định split.

EN-NN có 350/3.490 mẫu với `|ModAvg − HeadAvg| ≥ 2`; DE-NN có 601/3.298. 445/446 cụm EN-NN có khoảng điểm khác 0 giữa các câu ở mỗi đầu ra; median khoảng max−min là 1,4 (Mod) và 1,3 (Head). Một cụm chỉ có một câu không đo được biến thiên. Biến thiên nhãn là hiện tượng quan sát, chưa phải bằng chứng nhân quả về nghĩa thay đổi do ngữ cảnh.

### 2.4 Cụm trong câu: không thể chỉ exact-match

Tìm chuỗi cụm nguyên dạng liên tiếp sau casefold/gộp whitespace, có biên từ Unicode:

| File | Có chuỗi nguyên dạng | Không khớp chuỗi nguyên dạng |
| --- | ---: | ---: |
| EN-NN | 2.565 | 925 |
| EN-PV | 523 | 1.044 |
| DE-NN | 2.268 | 1.030 |
| DE-PV | 304 | 1.186 |

**Không khớp không có nghĩa thiếu mục tiêu.** Ví dụ thực tế chỉ trích đoạn văn bản, không đưa điểm nhãn:

- EN-NN `10001`, `account book`: “… account books …” → head số nhiều.
- DE-NN `50003`, `Abiturzeugnis`: “… meines Abiturzeugnisses …” → biến cách.
- EN-PV `30002`, `boil down`: “… be boiled down …” → động từ biến hình.
- EN-PV `30020`, `put down`: “I put it down to stubbornness and laziness of thought.” → thành phần không liên tiếp.
- DE-PV `70014`, `ablehnen`: “Die Linken lehnen sie ab, …” → base và particle tách nhau.

Có 129 ứng viên EN-PV mà từng token nguyên dạng xuất hiện nhưng cả chuỗi không khớp; kiểm tra `Base`/`Particle` riêng ở DE-PV cho 57 ứng viên tương tự. Đây là ứng viên tìm span, chưa là xác nhận tự động mọi lần xuất hiện đúng mục tiêu. 25 ứng viên EN-NN theo kiểm tra token rời có thể là trùng token ở nơi khác, không chứng minh NN bị tách. Độ dài câu theo whitespace: EN-NN 8–48, EN-PV 8–46, DE-NN 8–47, DE-PV 9–46 từ; chưa đo độ dài subword bằng tokenizer.

Không có cột năm, thời kỳ, nguồn corpus hoặc nhãn lịch sử/hiện đại trong schema. Không gán nhóm thời gian từ văn phong, ngày trong câu hoặc dải ID. Muốn nghiên cứu chuyển miền thời gian phải có mapping nguồn đáng tin cậy từ BTC.

### 2.5 Vấn đề ảnh hưởng nghiên cứu và dấu vân tay nguồn

Ưu tiên xử lý: câu chung giữa nhiều mục tiêu; biến hình/tách thành phần; cụm ít câu; phân bố điểm khác giữa ngôn ngữ/thành phần; nhãn phân tán lớn ở một số mẫu; thiếu metadata thời gian và provenance gộp. Không dùng Std làm trọng số loss ban đầu để tránh thêm giả định chưa kiểm chứng. Thiếu phép đo tần suất pretrain nên chưa thể khẳng định mô hình ghi nhớ một cụm nào.

SHA-256 tại thời điểm khảo sát, để M1 kiểm tra dữ liệu nguồn không đổi:

```text
en-nn-train.tsv  98ca6db405a5cdea6cebaaed323ef6e50e3ddd2926b908bbfded8badeee8d895
en-pv-train.tsv  492aa8f2393c154e3301267cbe1a8a434e345c7214ec07682a3171e326b31072
de-nn-train.tsv  43ba25c396a903501a2113f9aab141ff6f1c7dbca6afe33a470009e84ed32cac
de-pv-train.tsv  b632854c8fc2a9e4d84759c9df319791f6bed90c594363b2cb1bf1f8af9e0b02
```

Thống kê phần2 ghi nhận khảo sát ban đầu. M1 sau đó đã tạo `scripts/audit_data.py`, chạy audit tiếng Anh và xuất JSON/CSV/PNG; xem báo cáo cập nhật [DATA_REPORT.md](DATA_REPORT.md). Các cách đếm token/chuẩn hóa mới có định nghĩa riêng, không lẫn với whitespace/exact match khảo sát ban đầu.

## 3. Phạm vi MVP

**MVP: EN-NN → hai điểm `ModAvg` và `HeadAvg`.** Có 3.490 mẫu/446 cụm; đủ nhỏ để học cách thiết kế pipeline và đối chứng, có nhiều câu mỗi cụm để khảo sát context sensitivity. Tiếng Anh giúp giảm số quyết định tokenizer/backbone ban đầu. Hai đầu ra giữ đúng khác biệt thành phần thay vì gộp thành một điểm.

MVP gồm kiểm tra dữ liệu → hai thiết lập split nội bộ → B0/B1/B2 → đánh giá và xuất prediction nội bộ → checkpoint → demo local. PV và tiếng Đức là mở rộng sau khi quy trình EN-NN đáng tin cậy; phải đổi output/schema và xử lý hình thái, không chỉ đổi tên file. Chưa xác minh luật cho phép chỉ nộp một nhánh; phạm vi MVP học tập không phải tuyên bố đủ điều kiện submission cuộc thi.

## 4. Cấu trúc repository đề xuất

Giữ nguyên vị trí nguồn hiện có, tránh di chuyển hoặc ghi đè TSV. Cây dưới đây là **đề xuất**, chưa tạo toàn bộ:

```text
data/
  train/                      # TSV nguồn hiện có, chỉ đọc
  evaluation/                 # nguồn evaluation BTC khi có, hiện rỗng
  processed/                  # bản chuẩn hóa + row key + thông tin span
  splits/en_nn/{A,B}/          # manifest train/val/final, seed và hash nguồn
src/cosense/
  data.py                     # loader và kiểm tra schema
  preprocess.py               # chuẩn hóa an toàn, tìm mục tiêu
  split.py                    # tạo split, kiểm tra leakage
  model.py                    # backbone + regression head
  train.py                    # train B1/B2 và resume
  evaluate.py                 # B0 và các metric nội bộ
  infer.py                    # inference dùng chung CLI/demo
  export_submission.py        # chỉ triển khai sau khi có template BTC
configs/                      # cấu hình dữ liệu, split và thí nghiệm
notebooks/kaggle_en_nn.ipynb   # gọi code cùng phiên bản repo
scripts/                      # audit/smoke test và lệnh chạy thuận tiện
tests/                        # kiểm tra hợp đồng dữ liệu/split/inference
checkpoints/                  # trọng số, tokenizer, trạng thái optimizer
results/                      # audit, metric, prediction, log, bảng phân tích
demo/app.py                   # Streamlit gọi infer.py
docs/                         # roadmap, hướng dẫn chạy, literature/error notes
README.md                     # điểm vào và lệnh chạy đã kiểm thử
```

M1 thêm `.gitignore` cho môi trường, cache, secret, nguồn chưa rõ license, processed data, checkpoint và output lớn. Chỉ commit code/config/tài liệu và báo cáo nhỏ đã rà soát; kiểm tra license dữ liệu, quyền chia sẻ lên Kaggle và giới hạn host Git trước khi đưa artifact lên ngoài. Không mặc định private Dataset thay thế việc kiểm tra quyền sử dụng. API key ở biến môi trường/Kaggle Secrets; pipeline baseline không cần API trả phí. Không commit `.env`, token hoặc cấu hình cá nhân.

## 5. EDA và thiết kế split nội bộ

### 5.1 Nguyên tắc chung

Với seed chia tập khởi đầu **42**, tạo riêng hai bộ manifest A/B. Tỷ lệ mục tiêu khoảng 70% train / 15% validation / 15% final; tỷ lệ thực tế thay đổi do group và cụm ít mẫu, phải báo cáo số cụm và số record. Giữ final nội bộ độc lập trong từng thiết lập, không dùng để chọn loss/hyperparameter hay sửa phương pháp. Hai final A/B có thể dùng lại một số mẫu vì cùng nguồn, không xem chúng là hai replication độc lập.

Split chỉ ghi row key `(file nguồn, ContextID)` và group; lưu config, seed, phiên bản thuật toán, hash nguồn, danh sách thành viên. Không sửa source để gắn split. Nếu cần dữ liệu processed, lưu riêng mapping về nguồn. “Validation” là tập chọn cấu hình; “final” là tập nội bộ dành cho lần đánh giá cuối. **Không gọi các split tự tạo là split chính thức.**

EDA M1: histogram điểm riêng Mod/Head; scatter hai điểm; số câu/cụm; độ dài tokenizer; phân bố Std để hiểu nhiễu; span/alignment; câu trùng/gần trùng; số mẫu mỗi nhóm. EDA mô tả toàn nguồn được phép phục vụ audit, nhưng sau khi khóa final không dùng nhãn final để lựa chọn cấu hình. Các ngưỡng phân tích lỗi chọn bằng train/val hoặc định trước.

### 5.2 A — Cụm đã thấy, ngữ cảnh mới

Cùng Compound có thể nằm trong train và validation/final, nhưng câu/mẫu không được trùng hoặc gần trùng qua các tập. Group theo câu chuẩn hóa và cụm câu gần trùng đã rà soát; phân bổ group sao cho mỗi cụm được đánh giá có ít nhất một mẫu train. Không chia từng dòng ngẫu nhiên rồi bỏ qua group câu.

`viscosity work` chỉ có một mẫu: đặt train-only, không tạo bản sao để đủ ba tập. Các cụm có 3–4 mẫu cần kiểm tra khả năng phủ từng tập; không ép tỷ lệ từng cụm bằng tỷ lệ chung. Nếu liên kết câu gây mâu thuẫn coverage, ghi rõ cụm bị loại khỏi đánh giá A hoặc đặt train-only theo quy tắc đã lưu. Câu hỏi A: mô hình dự đoán một **cách dùng mới của cụm đã học** tốt đến đâu?

### 5.3 B — Cụm chưa thấy khi fine-tune

Group theo Compound chuẩn hóa (casefold/gộp whitespace; ghi lại mapping để kiểm tra biến thể). Không có cùng cụm trong nhiều split. **Chưa thấy khi fine-tune không có nghĩa chưa thấy khi pretrain**: DistilBERT có thể từng gặp cụm trong corpus tiền huấn luyện, chưa biết được mức độ.

Group Compound đơn thuần vẫn rò rỉ câu chung. Tạo graph nối hai Compound nếu chúng có câu trùng/gần trùng đáng tin cậy; chia theo connected component (nhóm liên thông), để tất cả câu và cụm liên quan nằm chung tập. Đọc lại cạnh gần trùng trước khi nối để tránh một component quá lớn. Câu hỏi B: mô hình khái quát tới **cụm ngoài dữ liệu fine-tune** tốt đến đâu?

Chọn B làm thiết lập chính cho câu hỏi khái quát, A làm thiết lập bổ sung về cách dùng mới. Không đổi split để làm đẹp kết quả.

### 5.4 Kiểm tra trước khi khóa split

Assert: đủ mỗi record đúng một lần trong mỗi thiết lập; không overlap row key/câu group; B không overlap Compound; A có train coverage cho các cụm đánh giá; nhãn parse hợp lệ; checksum nguồn không đổi. Sàng lọc gần trùng bằng token Jaccard và character n-gram, đọc các cặp cao nhất và kiểm tra riêng cặp đi qua ranh giới split. Ghi cả ngưỡng và giới hạn: không công cụ đơn giản nào bảo đảm bắt hết paraphrase. Không dùng nhãn để quyết định một cặp câu có trùng hay không.

## 6. Baseline theo thứ tự

| Baseline | Input | Output và mục đích |
| --- | --- | --- |
| B0 | không đọc văn bản | hai trung bình **trên train của split đang chạy**; chuẩn tham chiếu rẻ |
| B1 | chỉ `Compound`, không có Context hoặc Mod/Head riêng | đo tín hiệu từ chuỗi cụm |
| B2 | `Compound` giống B1 + `Context` | đo lợi ích tăng thêm của ngữ cảnh |

Backbone khởi đầu: [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), encoder tiếng Anh 6 lớp, model card mô tả khoảng 67 triệu tham số. Nhỏ hơn BERT base, hợp lý để thử Kaggle sau khi đo VRAM; chưa bảo đảm chạy trên mọi GPU/session. Dùng cùng checkpoint revision, tokenizer, pooling, head hai đầu ra và cách scale điểm cho B1/B2.

Theo protocol v2, B1 chỉ đọc chuỗi `Compound` gốc; B2 đọc cùng Compound và Context bằng text pair/template khóa sau smoke. Không đưa Mod/Head riêng, Avg, Std hoặc ID vào input. Mean pooling masked hidden states → linear layer hai đầu ra; cùng thiết kế cho cả hai baseline. Lưu prediction thô và clip [0,5], báo ngoài khoảng. Component/marker input nếu thử về sau là ablation riêng, không sửa đối chứng chính âm thầm. Submission theo BTC khi xác minh.

Đối chứng công bằng: cùng split/seeds, optimizer, effective batch, epochs tối đa, early stopping, số lần thử hyperparameter, tiêu chí chọn checkpoint và max length. B2 tốn token/thời gian hơn vì có câu; báo cả GPU time và số optimizer steps để hiểu chi phí. Không thêm augmentation, kiến trúc hoặc marker riêng cho B2 ở so sánh chính. Nếu thử marker/span pooling về sau, ghi thành ablation riêng. B0 dự đoán hằng nên tương quan có thể không xác định; không thay NaN bằng “0” mà không chú thích.

## 7. Kế hoạch huấn luyện và Kaggle

Phần này mô tả cách chạy pipeline. Để quyết định **chạy thí nghiệm nào, vì sao chạy và kết quả cho phép kết luận gì**, đọc thêm [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md). Tài liệu bổ sung hướng dẫn từ câu hỏi nghiên cứu đến protocol, đối chứng, độ bất định và cấu trúc paper; toàn bộ bước có cập nhật trọng số vẫn chạy trên Kaggle GPU.

### 7.1 TSV → Dataset → tokenization

1. Loader nhận manifest file tường minh, kiểm tra đúng schema, finite score/range và row key. Tạo record processed giữ nguyên văn bản gốc, các trường input, labels riêng và metadata audit riêng.
2. Load split manifest; train Dataset có `labels=[ModAvg, HeadAvg]`, inference Dataset không chứa gold/Std. M1 dùng list/dict thư viện chuẩn; M3 chọn PyTorch Dataset hoặc Hugging Face Dataset sau khi kiểm tra môi trường, không bắt buộc cài cả hai.
3. Dùng fast tokenizer với offset mapping để kiểm tra span trên văn bản gốc. Làm tìm mục tiêu theo tầng: chuỗi exact có biên → biến thể hình thái/hyphen → component candidates. Ghi trạng thái `exact / variant / discontinuous / ambiguous / unresolved` và offset nhiều span nếu cần. Không chọn lần xuất hiện đầu tiên khi có nhiều ứng viên mà chưa kiểm tra.
4. B2 đầu tiên đọc sentence + target riêng, **không bắt buộc chèn marker**. Phần target riêng giúp định danh mục tiêu ngay cả khi không tìm exact span. Tuy nhiên phải bảo toàn phần câu chứa cách dùng mục tiêu; target ở segment riêng không thay thế ngữ cảnh đã bị cắt mất.
5. Max length khởi đầu 256 token, dynamic padding. Đo token length toàn bộ trước khi cắt; không suy từ 48 từ rằng chắc chắn dưới giới hạn. Khi câu quá dài và có alignment đáng tin, chọn cửa sổ giữ toàn bộ span mục tiêu và ngữ cảnh hai phía. Với span không liên tiếp, giữ các thành phần; nếu khoảng cách vượt cửa sổ, ghi ngoại lệ và xử lý bằng cấu hình dài hơn đã kiểm tra VRAM. Nếu không resolve được span và câu cần cắt, không cắt âm thầm: đưa vào danh sách kiểm tra/quarantine, báo coverage. Không tự bỏ mẫu mà không ghi lý do.

Tokenization là chuyển văn bản thành token/subword ID; truncation là cắt để vừa giới hạn. Regression là học dự đoán số thực. Checkpoint là trọng số và thông tin cần tải lại, không chỉ một file điểm số.

### 7.2 Cấu hình khởi đầu, chưa tối ưu

| Tham số | Điểm bắt đầu |
| --- | --- |
| Output NN | linear regression head 2 đầu ra, thứ tự Mod rồi Head |
| Loss | MSE trung bình trên hai thành phần; đơn giản, phạt sai số lớn và phù hợp học mean rating |
| Optimizer | AdamW, weight decay 0,01 |
| Learning rate | 2e-5; nếu tuning, cùng grid nhỏ `{1e-5, 2e-5, 3e-5}` cho B1/B2 |
| Batch | 16/GPU, accumulation 1, effective batch 16 |
| Epochs | tối đa 5, validation mỗi epoch |
| Early stopping | patience 2 epoch; chọn validation MAE trung bình theo Compound và hai đầu ra nhỏ nhất (E trong roadmap nghiên cứu) |
| Max length | 256, xác minh target không mất |
| Warmup / gradient clipping | 10% optimizer steps / norm 1,0 |
| Seed | khởi đầu 42; nghiên cứu thêm 13, 2027 |
| Mixed precision | chỉ bật sau khi xác minh GPU/runtime hỗ trợ; CPU dùng float32 |

MSE giả định khoảng cách điểm có ý nghĩa; đây là xấp xỉ làm việc, không tuyên bố đúng hoàn toàn với ordinal ratings. Huber/MAE là lựa chọn đối chứng sau phân tích outlier; không chỉnh loss dựa trên final. Quy tắc clip nội bộ ban đầu: tính metric chính trên điểm clip [0,5], đồng thời báo số dự đoán ngoài khoảng và MAE/RMSE thô để tránh che lỗi.

**Smoke test trước train đầy đủ, chạy trên Kaggle GPU:** 16–32 mẫu train và 8 mẫu validation, có biến hình, hai đầu ra khác nhau và input dài; chạy 1–2 optimizer steps, không ghi đè checkpoint chính. Kiểm tra labels không lọt input, shape `[batch,2]`, loss/gradient hữu hạn, cập nhật trọng số, tokenizer không mất target, lưu/tải lại cho prediction nhất quán, và xuất đúng ID/số dòng. Có thể thử overfit tập rất nhỏ trên Kaggle để chẩn đoán pipeline, không dùng điểm đó làm chất lượng mô hình. Local chỉ kiểm tra loader/split/metric và inference, không chạy optimizer steps. Nếu Kaggle chưa có GPU khả dụng, tiếp tục chuẩn bị dữ liệu/code và chờ phiên GPU; không tự chuyển fine-tune về local.

### 7.3 Artifact và cách chạy Kaggle

Notebook dự kiến gồm các phần theo thứ tự:

1. In runtime Python/package/GPU/VRAM; kiểm tra GPU PyTorch thực dùng.
2. Gắn code và Dataset chứa nguồn hợp pháp + split manifests/config. Input Kaggle là read-only; kiểm tra commit/code hash và hash TSV trước khi chạy.
3. Cài dependency tối thiểu, khóa phiên bản đã kiểm thử. Bật Internet để tải backbone nếu cần; hoặc gắn bản backbone/tokenizer được phép sử dụng. Không đưa credential vào notebook.
4. Chạy loader audit, kiểm tra split, tokenization coverage và smoke test.
5. Chạy train với config A hoặc B và baseline B1/B2 tường minh, output vào `/kaggle/working/`.
6. Evaluate validation; final chỉ chạy khi đã khóa lựa chọn. Lưu config, seed, log, environment, checkpoint best/last, tokenizer, mapping output, prediction và metric.
7. Tải artifact output về local, kiểm tra tải lại checkpoint và inference CPU trước khi nối demo.

**Chuẩn bị input từ local:** tạo hai gói riêng: code (`src/`, `configs/`, notebook và dependency manifest đã kiểm thử) và dữ liệu (TSV nguồn được phép sử dụng, processed data nếu cần, split manifests và source hashes). Dùng Kaggle Dataset riêng tư khi phù hợp quyền dữ liệu. Notebook nhận đường dẫn input qua config, không hard-code đường dẫn Windows hoặc tên Dataset chưa tạo; không chia split ngẫu nhiên lại mỗi lần chạy. Khi chỉ sửa code, cập nhật gói code và giữ phiên bản dữ liệu/split đã khóa.

**Output cần giữ và tải về:** checkpoint best cho demo; checkpoint last cùng trạng thái train để resume; tokenizer, config và thứ tự output; log, package versions, code/source/split hashes; validation predictions và metrics. Kiểm tra artifact sau khi lưu phiên notebook. Chưa có checkpoint thì demo báo chưa có mô hình, không tự train local. B0 chạy ở bước đánh giá baseline trong notebook và không cần dùng GPU; B1/B2, tuning, ablation có train lại và các seed M6 đều chạy trên Kaggle GPU.

Run directory tách theo setting/baseline/seed/run ID để tránh đè kết quả. Lưu cả checkpoint tốt nhất theo validation và checkpoint gần nhất có optimizer/scheduler/scaler/RNG, epoch/step và early-stopping state để resume. Lưu split hash, source hash, model revision và code version cùng run; prediction nội bộ lưu ID, hai điểm và split, gold chỉ ở artifact đánh giá có kiểm soát.

Khi hết VRAM: giảm microbatch 16→8→4, tăng accumulation để giữ effective batch 16; kiểm tra mixed precision, rồi cân nhắc giảm max length **chỉ sau kiểm tra target**. Các thay đổi phải áp dụng công bằng B1/B2 và ghi lại config. Khi session ngắt: lấy output checkpoint đã được lưu ra ngoài phiên, dựng đúng code/config/data, resume từ last với trạng thái optimizer/RNG; kiểm tra resume smoke trước. File chỉ nằm trong session có thể mất khi phiên bị hủy; chủ động lưu notebook version/output và tải checkpoint định kỳ, không giả định luôn tự khôi phục được.

Không đưa training vào demo: demo chỉ tải checkpoint và gọi inference.

### 7.4 Chuyển một lần train thành một thí nghiệm khoa học

Trước khi mở notebook Kaggle, viết một phiếu thí nghiệm: câu hỏi, giả thuyết, đối chứng, biến được thay đổi, split/hash, seed, ngân sách tuning, metric chính và cách diễn giải kết quả thuận/nghịch. Notebook đọc config của phiếu đó; lưu cả lần chạy thất bại, không chỉ checkpoint có điểm tốt nhất.

Thứ tự nghiên cứu: **khóa protocol → smoke → pilot trên validation → phân tích lỗi → chọn hướng → chạy xác nhận nhiều seed → đánh giá final sau khi khóa lựa chọn → viết claim theo bằng chứng**. Pilot dùng để học cách hệ thống hoạt động; không trình bày giả thuyết nghĩ ra sau pilot như giả thuyết đã định trước. Ba seed là mức khởi đầu của project, không phải quy định đảm bảo paper hợp lệ; độ mạnh bằng chứng còn phụ thuộc effect size, số nhóm độc lập, đối chứng và phạm vi claim.

Các chỉ số loss/MAE sau một lần train chỉ chứng minh mức dự đoán trong cấu hình đó. Muốn nghiên cứu context sensitivity phải có đối chứng bỏ/đổi context và phân tích trong cùng cụm; muốn nghiên cứu khái quát phải dùng split B và báo giới hạn pretraining. Ma trận thí nghiệm, công thức metric và tiêu chí chuyển giai đoạn nằm trong roadmap nghiên cứu bổ sung.

## 8. Đánh giá và giới hạn diễn giải

### 8.1 Những gì đã/ chưa xác minh chính thức

Đã xác minh trang giới thiệu nhiệm vụ. Trang đó dẫn sang [CodaBench competition 17971](https://www.codabench.org/competitions/17971/) để xem chi tiết. Lần truy cập 14/10/2026 trả về nội dung không liên quan đến hướng dẫn cuộc thi; không lấy nội dung đó làm luật. Tìm bổ sung chưa có nguồn chính thức đủ xác nhận.

**Chưa xác minh:** metric chính thức, cách aggregate/ranking, xử lý missing prediction, cột/định dạng output, tên/số file, cách đóng ZIP, giới hạn submission, có bắt buộc cả bốn nhánh không, chính sách external data/model và license. Không lấy luật của SemEval task khác thay vào. Khi có template/scorer BTC, đọc và kiểm thử rồi cập nhật mục này.

### 8.2 Metric nội bộ và báo cáo

- MAE: trung bình sai số tuyệt đối, đơn vị điểm 0–5, dễ diễn giải.
- RMSE: căn trung bình bình phương sai số, nhạy với lỗi lớn.
- Pearson/Spearman: chẩn đoán quan hệ tuyến tính/thứ hạng; **không nhận là metric chính thức**. Báo NaN khi prediction/nhãn không đủ biến thiên.
- Báo riêng Mod, Head và macro mean MAE/RMSE của hai đầu ra. Metric nghiên cứu chính đề xuất là MAE trung bình theo Compound và hai đầu ra (E trong `RESEARCH_ROADMAP.md`), dùng cả cho chọn checkpoint validation; báo MAE theo dòng làm metric phụ để thấy ảnh hưởng số câu/cụm. Đây chưa phải metric chính thức BTC.
- Tách setting A/B; khi mở rộng tách ngôn ngữ và NN/PV. Nhóm phân tích: số câu/cụm, độ dài, exact/variant/unresolved, dải điểm định trước, `|ModAvg−HeadAvg| ≥ 2`, và nhóm biến thiên nhãn trong cụm. Báo số mẫu/cụm từng nhóm; không diễn giải mạnh nhóm quá nhỏ.

Tập con chênh lệch Mod/Head kiểm tra mô hình có dự đoán hai thành phần khác nhau hợp lý hay co về cùng điểm; xem cả sai số riêng và sai số của chênh lệch dự đoán. Với cùng cụm nhiều câu, đo dispersion dự đoán, sai số và thứ hạng nội cụm khi đủ mẫu; dispersion lớn tự nó không phải thành công.

Kết luận nghiên cứu cần ít nhất ba training seed trên split cố định, báo mean/std. Khoảng tin cậy cho B2−B1 có thể dùng paired bootstrap theo Compound (B dùng component chống rò rỉ khi thích hợp) thay vì lấy từng câu độc lập; lưu phương pháp và số bootstrap. Khoảng tin cậy trên split cố định không phản ánh hết bất định do chọn split. Nếu đủ tài nguyên, lặp group split làm sensitivity study tách biệt, không chọn split tốt nhất.

**Giới hạn:** khác điểm giữa hai câu có thể phản ánh nghĩa, chủ đề, nguồn, nhiễu người đánh giá hoặc khó khăn annotation. Không phải cặp can thiệp kiểm soát, nên không tự kết luận ngữ cảnh gây đổi nghĩa. B2 hơn B1 chỉ cho thấy thêm input có lợi trong thiết lập kiểm tra; chưa chứng minh cơ chế “hiểu”. B1 mạnh cũng chưa đủ chứng minh “ghi nhớ”.

## 9. Hướng nghiên cứu và literature

Kế hoạch triển khai khoa học chi tiết nằm trong [roadmap nghiên cứu](RESEARCH_ROADMAP.md), liên kết với M1–M6 ở mục 12. Phần dưới giữ câu hỏi và literature khởi đầu; các giả thuyết chưa phải kết quả đã chứng minh.

Câu hỏi khởi đầu: **“Mô hình có sử dụng ngữ cảnh để dự đoán điểm hợp thành của từng thành phần, và có khái quát đến danh từ ghép chưa thấy khi fine-tune không?”**

| Giả thuyết có thể bị bác bỏ | Thí nghiệm/ablation | Quan sát làm yếu hoặc bác bỏ giả thuyết |
| --- | --- | --- |
| H1: thêm Context giúp dự đoán trên cụm chưa thấy khi fine-tune | B2 vs B1 trên B, đối chứng công bằng, nhiều seed, paired CI | không giảm MAE ổn định hoặc CI chưa phân biệt được lợi ích |
| H2: B2 dùng ngữ cảnh đúng của từng cách dùng | bỏ context; hoán đổi context giữa các câu **cùng cụm**; so với B2 chuẩn trên cùng mẫu | điểm/sai số không đổi đáng kể, hoặc context sai không làm xấu đi |
| H3: hai đầu ra thể hiện khác biệt Mod/Head | phân tích nhóm gap≥2, lỗi từng head, ablation hai output so với ràng buộc cùng output | dự đoán thường trùng nhau và không tốt hơn đối chứng ràng buộc |
| H4: lợi ích context không chỉ đến từ định danh cụm đã fine-tune | so A/B; báo error theo cụm; kiểm tra tín hiệu lexical và shared components | lợi ích chỉ xuất hiện ở A; B không ổn định |

Hoán đổi cùng cụm giữ target hợp lệ tốt hơn lấy câu bất kỳ, nhưng chưa tạo một thí nghiệm nhân quả hoàn hảo; đánh giá chỉ trên cụm có ≥2 câu khác nhau và báo coverage. Các ablation phải ghi là train lại hay thay input lúc inference; thay input gây distribution shift có thể làm điểm giảm dù mô hình không dùng context theo cơ chế mong đợi. Dùng cùng ngân sách tuning, không chọn ablation theo final.

**Mức đọc literature hiện tại: đã đọc abstract trên ACL Anthology của cả bốn bài, chưa đọc toàn văn/phụ lục/code.** Các tóm tắt sau chỉ dựa abstract, không gán thiết kế split, loss, số seed hoặc cơ chế chưa đọc:

| Nguồn | Điều abstract hỗ trợ | Cần đọc toàn văn để phục vụ CoSense |
| --- | --- | --- |
| [NCTTI — Garcia et al., ACL 2021](https://aclanthology.org/2021.acl-long.212/) | dữ liệu NN có annotation type/token ở Anh/Bồ Đào Nha; khảo sát biến thiên idiomaticity theo câu | định nghĩa thang nhãn, sampling, đánh giá token vs type, quan hệ nguồn với dữ liệu task |
| [A Systematic Search… — Miletic & Schulte im Walde, EACL 2023](https://aclanthology.org/2023.eacl-main.110/) | khảo sát biểu diễn BERT cho compositionality NN; preprocessing/embedding/layer và vai trò Mod/Head ảnh hưởng kết quả | pooling, layer, estimator và đối chứng thực tế; không mặc định supervised B2 của ta tái lập bài này |
| [Rolling the DICE… — Mi et al., ACL 2025](https://aclanthology.org/2025.acl-long.362/) | bộ contrastive có kiểm soát để kiểm tra disambiguation idiom theo context; khảo sát frequency/probability và hạn chế mô hình | cách tạo cặp, kiểm soát shortcut, annotation và phép đo; idiom detection không đồng nhất regression thành phần NN |
| [Memorization or Reasoning?… — Kim et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1099/) | MIDAS đa ngôn ngữ; tác giả báo bằng chứng hướng kết hợp kiến thức nội tại/context/reasoning | thao tác kiểm tra memorization, protocol và hạn chế; không suy kết luận về DistilBERT của CoSense |

M4/M6 lập `docs/LITERATURE_NOTES.md`: mỗi bài ghi task, nguồn/split, input, model, metric, controls, giới hạn, phần đã đọc và liên hệ CoSense; tìm công trình gần đây qua primary sources và đọc task paper/baseline BTC khi có. Chỉ xác định khoảng trống sau đối chiếu đầy đủ; câu hỏi ở đây là hướng khảo sát, chưa phải claim mới.

Sau lỗi baseline mới chọn cải thiện: ví dụ span pooling nếu lỗi nằm ở định vị thành phần; loss robust nếu outlier ảnh hưởng; representation layer nếu có bằng chứng liên quan. Mỗi phương án cần hypothesis và đối chứng, không triển khai tất cả cùng lúc. Không kết luận “hiểu” hoặc “ghi nhớ” từ một bảng metric; tách dữ liệu quan sát khỏi lời giải thích cơ chế. Kết quả âm vẫn có giá trị báo cáo nếu thí nghiệm đúng, không bảo đảm đủ cho paper được nhận.

## 10. Web demo đơn giản

Chọn **Streamlit** cho MVP vì repository chưa có web stack, dễ chạy local và dùng Python inference chung. Chưa cài hoặc xây trong lượt đầu.

Luồng demo: tải checkpoint local hợp lệ → nhập câu + Compound + modifier/head theo input contract B2 → kiểm tra input → tokenizer/inference → hiển thị hai điểm clip 0–5. Gắn nhãn rõ **“Điểm dự đoán của mô hình, không phải điểm của người đánh giá”**. Có các ví dụ tự tạo ở mục 1 để thử, không điền gold và không bảo đảm số cụ thể.

Cache model/tokenizer theo checkpoint; `eval()` và inference không gradient, chạy CPU local nếu đủ RAM và đo latency thực tế. Tải đúng config, revision và thứ tự Mod/Head. Không nhập Avg/Std/ID vào inference, không huấn luyện khi người dùng bấm dự đoán, không dùng số ngẫu nhiên khi chưa có model.

Xử lý rõ: checkpoint chưa tồn tại → thông báo cách tải artifact M3 và khóa nút dự đoán; thiếu trường/chuỗi rỗng → lỗi dễ hiểu; target không tìm exact → kiểm tra biến thể, cho biết alignment không chắc; câu dài cần cắt mà không giữ được mục tiêu → yêu cầu sửa input hoặc báo không xử lý được. Không tự đổi input âm thầm. Nếu model tải lỗi, hiển thị lỗi và hướng dẫn, không giả lập kết quả. Khi có nhiều lần xuất hiện mục tiêu, yêu cầu chọn cách dùng cần dự đoán nếu inference contract cần span.

Demo local là đủ trình bày trên lớp; triển khai public là việc riêng, chưa thực hiện. M5 chỉ hiển thị kết quả của checkpoint đã qua kiểm tra load/inference.

## 11. Submission cuộc thi

Cần lấy **template, scorer và luật chính thức** qua CodaBench/BTC; nếu trang vẫn không trả hướng dẫn, người dùng lấy file từ tài khoản hoặc hỏi BTC. Chưa xác định tên cột output, số file, cách đóng ZIP hay có cần tất cả nhánh.

Sau xác minh mới viết `export_submission.py`: đọc template/input đúng phiên bản → gọi infer giữ thứ tự/ID → điền đúng trường được yêu cầu → kiểm tra số dòng, uniqueness, missing/finite/range → round-trip đọc lại → chạy scorer/validator BTC hoặc thử sample submission theo quyền người dùng. Lưu version/hash template và config model. Các file prediction nội bộ ở M2/M3 không tự nhận là submission hợp lệ.

Không dùng final nội bộ làm “test cuộc thi”; không tune trên leaderboard test hoặc gold evaluation. Chưa đăng ký, upload, nộp kết quả hoặc publish. Không tự bịa bundle khi chưa có template.

## 12. Milestone, tiêu chí hoàn thành và lịch dự kiến

### M1 — Hiểu dữ liệu và cố định split

- **Trạng thái:** hoàn thành cho phạm vi hiện tại: audit EN-NN/EN-PV, split EN-NN A/B. Báo cáo tại `docs/DATA_REPORT.md`, config tại `configs/data_protocol.json`; scripts nằm trong `scripts/`, chưa cần package neural `src/cosense/`.
- **Mục tiêu:** dữ liệu audit được, A/B không rò rỉ đã biết và có thể tái lập.
- **Việc:** tạo loader/audit stdlib, manifest nguồn/hash, EDA, kiểm tra duplicate/near-duplicate/span; tạo A/B và kiểm tra coverage; thêm ignore policy; ghi provenance trial đã gộp và license chưa rõ.
- **Đầu ra dự kiến:** `src/cosense/data.py`, `preprocess.py`, `split.py`, `scripts/audit_data.py`, `configs/split_en_nn.json`, `data/processed/`, `data/splits/en_nn/`, `results/eda/`, `docs/DATA_NOTES.md`, `.gitignore`.
- **Hoàn thành khi:** script chạy lại cho thống kê khớp; TSV hash không đổi; split coverage/disjointness pass; cặp nghi ngờ và ngoại lệ được ghi; seed/manifest lưu. Chưa bắt buộc resolve hết hình thái bốn nhánh để hoàn thành MVP EN-NN, nhưng phải ghi giới hạn.
- **Lỗi thường gặp:** train+trial nạp hai lần, tách câu chung qua split, group Compound mà bỏ câu trùng, tự gán thời kỳ, xóa mẫu đa mục tiêu hợp lệ.
- **Phụ thuộc:** nguồn hiện có và roadmap; không phụ thuộc GPU.

### M2 — Baseline trung bình và pipeline đánh giá

- **Trạng thái:** B0 validation và evaluator dùng chung đã hoàn thành: `scripts/run_mean_baseline.py`, `scripts/data_utils.py`; A đạt18/18 và B đạt20/20 assertion split, có integrity tests trong `tests/`. Kết quả `results/b0/`; chưa đánh giá final hoặc chạy B1/B2.
- **Mục tiêu:** có đường đi tối thiểu data → prediction → metric, chưa cần transformer.
- **Việc:** viết và kiểm tra metric ở local; tính B0 trên train riêng A/B trong notebook Kaggle (không cần GPU); xây MAE/RMSE và tương quan có xử lý undefined; viết output nội bộ, kiểm tra mapping ID; đánh giá validation, để final khóa.
- **Đầu ra dự kiến:** `src/cosense/evaluate.py`, cấu hình B0, `results/B0/{A,B}/` chứa train means, validation predictions, metrics và run metadata.
- **Hoàn thành khi:** kiểm tra metric bằng ví dụ tính tay nhỏ; B0 đúng hai output, không lấy mean toàn nguồn; đủ đúng dòng validation, không input leakage.
- **Lỗi thường gặp:** lấy mean cả train+val, dùng ID làm feature, báo correlation của hằng số như kết quả hợp lệ.
- **Phụ thuộc:** M1 split đã khóa.

### M3 — Baseline chỉ cụm và cả câu

- **Trạng thái:** chưa bắt đầu.
- **Mục tiêu:** B1/B2 chạy được, có checkpoint và đối chứng công bằng.
- **Việc:** chuẩn bị code/config/input ở local; kiểm tra runtime và dependency trên Kaggle; tokenizer/target retention; model/train/infer; smoke có optimizer steps trên Kaggle GPU; chạy B1/B2 trên Kaggle, đo thời gian/VRAM; chọn checkpoint chỉ từ validation; tải artifact về local và kiểm tra CPU inference.
- **Đầu ra dự kiến:** `model.py`, `train.py`, `infer.py`, `configs/en_nn_b1.*`, `configs/en_nn_b2.*`, `notebooks/kaggle_en_nn.ipynb`, checkpoint/config/tokenizer và `results/` từng run, hướng dẫn chạy README.
- **Hoàn thành khi:** save/load prediction khớp trong tolerance; full run có log/seed/hash; B1/B2 cùng protocol; validation metric và lỗi được báo dù mô hình không thắng B0; inference không cần nhãn.
- **Lỗi thường gặp:** cắt mất mục tiêu, sai thứ tự labels, quên eval mode, session dùng code cũ, chỉ lưu model mà không thể resume, tuning lệch ngân sách.
- **Phụ thuộc:** M1/M2; GPU Kaggle và khả năng tải backbone phải được xác minh ở thời điểm chạy.

### M4 — Phân tích lỗi và chọn hướng cải thiện

- **Trạng thái:** chưa bắt đầu.
- **Mục tiêu:** tìm giới hạn baseline, chọn tối đa một hướng cải thiện có lý do đo được.
- **Việc:** phân tích validation theo Compound/Mod/Head/gap/biến hình; B1/B2 và context ablation; đọc toàn văn literature; chọn hypothesis hoặc quyết định giữ baseline nếu chưa có bằng chứng.
- **Đầu ra dự kiến:** `docs/ERROR_ANALYSIS.md`, `docs/LITERATURE_NOTES.md`, bảng/plot trong `results/analysis/`, config thí nghiệm đề xuất.
- **Hoàn thành khi:** mỗi kết luận có dữ liệu và giới hạn; hướng tiếp theo gắn lỗi cụ thể, đối chứng và tiêu chí bác bỏ; không dùng final để chọn phương pháp.
- **Lỗi thường gặp:** giải thích mọi lỗi là memorization, chọn vài ví dụ đẹp, đổi nhiều yếu tố đồng thời, tuyên bố novelty từ tiêu đề bài.
- **Phụ thuộc:** M3 có prediction/metadata đáng tin cậy.

### M5 — Web demo

- **Trạng thái:** chưa bắt đầu.
- **Mục tiêu:** trình bày được B2 inference thật trên local.
- **Việc:** Streamlit input/checkpoint, cache, output hai điểm, ví dụ tự tạo, xử lý lỗi; kiểm thử bình thường/thiếu checkpoint/input dài/biến hình; đo latency CPU.
- **Đầu ra dự kiến:** `demo/app.py`, dependency demo tối thiểu, `docs/DEMO_GUIDE.md` và hướng dẫn lấy checkpoint.
- **Hoàn thành khi:** chạy từ phiên mới, dùng cùng infer với CLI, không có gold/Std hay training trong luồng demo, không checkpoint thì không tạo điểm.
- **Lỗi thường gặp:** random fallback, tải lại model mỗi click, dùng nhầm checkpoint/B1 input, ngụ ý prediction là gold.
- **Phụ thuộc:** M3 inference ổn định; M4 giúp chọn ví dụ và nêu giới hạn, có thể làm giao diện sau M3 nếu cần lịch trình lớp.

### M6 — Thí nghiệm nghiên cứu và báo cáo/paper

- **Trạng thái:** chưa bắt đầu.
- **Mục tiêu:** trả lời câu hỏi bằng thí nghiệm tái lập, xác định mức độ đóng góp thực tế.
- **Việc:** chạy nhiều seed và các ablation cần huấn luyện trên Kaggle GPU; phân tích prediction/paired CI ở local; mở rộng PV/DE nếu có lý do/tài nguyên; đọc literature đầy đủ; final evaluation sau khóa phương pháp; viết báo cáo và đối chiếu claim với bằng chứng. Submission adapter chỉ khi luật/template đã rõ.
- **Đầu ra dự kiến:** configs/runs nghiên cứu, bảng tổng hợp/CI, `docs/RESEARCH_REPORT.md`, bản thảo nếu kết quả và literature hỗ trợ.
- **Hoàn thành khi:** kết quả cả tốt/xấu được ghi, môi trường và split tái lập được, final không dùng tuning, giới hạn được nêu; báo cáo không cần hứa thắng cuộc thi hoặc được publish.
- **Lỗi thường gặp:** một seed rồi kết luận, pseudo-replication theo câu, gọi unseen fine-tune là unseen pretrain, dùng nhãn thời gian tự suy đoán, paper claim quá evidence.
- **Phụ thuộc:** M4; M5 hỗ trợ trình bày nhưng không phải điều kiện khoa học; quyền dữ liệu/luật BTC cho phần cuộc thi.

### Lịch theo tuần (ước lượng có điều kiện)

Giả định bắt đầu tuần **14–20/10/2026**, dành 6–10 giờ/tuần, có Kaggle GPU cho M3, không có API trả phí; thời gian gồm học khái niệm, debug và ghi chú. Nếu quota/GPU/license chưa giải quyết hoặc chương trình học bận, lùi lịch; chưa đo tốc độ nên không hứa số giờ train.

| Tuần | Công việc chính | Mốc dự kiến |
| --- | --- | --- |
| 1: 14–20/10 | audit tái chạy, duplicate/span, split A/B | M1 |
| 2: 21–27/10 | B0, metric và prediction pipeline | M2 |
| 3–4: 28/10–10/11 | tokenizer, smoke, Kaggle B1/B2, checkpoint/inference | M3 |
| 5: 11–17/11 | phân tích lỗi + đọc papers đầy đủ | M4 |
| 6: 18–24/11 | Streamlit và kiểm tra trình bày | M5 |
| 7–9: 25/11–15/12 | seeds/ablation, cải thiện có kiểm chứng, báo cáo | M6 ban đầu |
| Sau đó | mở rộng có điều kiện, xác minh template/luật, chuẩn bị evaluation | phụ thuộc BTC và tài nguyên |

[Trang BTC](https://sites.google.com/view/semeval-compartment) tại ngày khảo sát ghi evaluation 10–31/01/2027; lịch có thể cập nhật. Chưa có deadline paper đã xác minh. Dành khoảng đệm trước evaluation, kiểm tra nguồn lại trước khi lên lịch submission.

## 13. Việc cần làm ngay và cách tiếp tục

1. **Bắt đầu M1:** tạo script audit chỉ đọc, manifest/hash bốn TSV và báo cáo lưu được. Học TSV/schema, label vs feature và hash; hiểu vì sao Std/ID không được vào input.
2. **Rà soát mục tiêu và câu trùng EN-NN:** kiểm tra ví dụ số nhiều, các cặp ID mục 2, rồi làm báo cáo gần trùng. Học token/subword, span và leakage: cùng thông tin đi qua train/val làm kết quả dễ quá mức.
3. **Cố định hai split A/B và final nội bộ:** lưu seed/config/membership, chạy asserts. Học group split và validation: A/B trả lời câu hỏi khác nhau, final dùng để kiểm tra kết luận cuối.
4. **Chuẩn bị notebook Kaggle cho M2/M3:** gắn code và dữ liệu/split đã khóa; chạy B0 để tính mean train, xuất validation prediction và MAE/RMSE; sau đó kiểm tra GPU và smoke trước B1/B2. Học baseline, regression, Kaggle input/output và checkpoint; chỉ bắt đầu fine-tune đầy đủ khi được yêu cầu.
5. **Chuẩn bị thông tin nguồn chính thức:** lấy license/provenance gộp, template/scorer/luật từ BTC khi có; đọc toàn văn literature theo bảng mục 9. Việc này làm song song, không ngăn audit local, nhưng cần trước khi đưa dữ liệu lên Kaggle hoặc tạo submission.

Khi người dùng nói **“bắt đầu M1”**, triển khai các file và chạy kiểm tra đến khi đáp ứng tiêu chí M1, rồi cập nhật trạng thái, artifact và vấn đề còn lại ở đây. Mỗi giai đoạn giải thích kết quả bằng tiếng Việt, tự quyết các chi tiết kỹ thuật thông thường; chỉ hỏi khi thông tin thiếu thay đổi hướng triển khai. Không chạy huấn luyện dài hoặc API trả phí khi chưa được yêu cầu.

## Nhật ký trạng thái

| Ngày | Hoàn thành | Chưa hoàn thành |
| --- | --- | --- |
| 14/10/2026 | khảo sát file/schema/thống kê, kiểm tra CPU/CUDA runtime, đọc trang task và abstract bốn papers; tạo roadmap | M1–M6; luật/metric/template/license; full-text review; metadata thời gian; runtime Kaggle |
| 14/10/2026 | cập nhật theo lựa chọn người dùng: huấn luyện và smoke có optimizer trên Kaggle GPU; local chuẩn bị dữ liệu/code, phân tích và demo | chưa triển khai notebook hoặc chạy huấn luyện |
| 14/10/2026 | thêm `docs/RESEARCH_ROADMAP.md` và mục 7.4: protocol, đối chứng context, kế hoạch thống kê, artifact Kaggle và khung paper | chưa triển khai thí nghiệm; chưa xác lập novelty hoặc kết quả nghiên cứu |
| 14/10/2026 | hoàn thành audit English, split EN-NN A (2443/524/523), B (2444/523/523), kiểm tra split, B0 validation, DATA_REPORT và RESEARCH_PROTOCOL; raw giữ nguyên | B1/B2 Kaggle, span review cho phương pháp cần span, luật/metric/license BTC; chưa đánh giá final |
| 17/10/2026 | tạo notebook tự chứa code prepare/smoke/train B1/B2, package ZIP EN-NN allowlist, hướng dẫn Kaggle chi tiết;17 tests local đạt, không đổi raw/membership | xác minh quyền upload; chạy prepare/smoke trên Kaggle; kết quả B1/B2 và demo; chưa đánh giá final |
