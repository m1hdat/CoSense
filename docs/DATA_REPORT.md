# CoSense — Báo cáo dữ liệu tiếng Anh và split nghiên cứu

Ngày khảo sát: 14/10/2026. Sinh từ script `scripts/build_data_report.py`, dựa trên artifact đã chạy thật.

## 1. Tóm tắt và mức sẵn sàng

EN-NN có **3,490 mẫu / 446 cụm**; EN-PV có **1,567 mẫu / 160 cụm**. Không có lỗi parse/schema, ô thiếu, ID lặp, mẫu trùng toàn bộ hoặc nhãn không hợp lệ. Có câu dùng chung giữa các mục tiêu: bốn nhóm EN-NN và một nhóm EN-PV, đã giữ lại vì là các mục tiêu khác nhau.

Hai split EN-NN đã được tạo và kiểm tra; B0 đã chạy trên **validation**, chưa tạo prediction/metric final. Có thể chuẩn bị notebook Kaggle B1/B2 với input Compound hoặc Compound+Context. Span heuristic còn cần rà soát nếu muốn dùng target markers/span pooling; chưa đủ để triển khai phương pháp cần span tự động đáng tin cậy.

**Quan sát:** nhãn EN-NN thiên về điểm cao và có biến thiên trong cùng cụm. **Giả thuyết cần kiểm chứng:** context có thể giúp dự đoán cách dùng cụ thể. **Đề xuất:** so B1/B2 theo [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), không kết luận mô hình hiểu nghĩa từ EDA.

## 2. Nguồn dữ liệu và từ điển cột

Chỉ đọc `data/train/en-nn-train.tsv` và `data/train/en-pv-train.tsv`. Hai file tiếng Đức hiện có được liệt kê trong manifest nhưng không nạp. Không tìm thấy bản train tiếng Anh thứ hai hay trial/ZIP riêng trong cây khảo sát. Theo người dùng, trial đã gộp; chưa có manifest gộp để xác minh từng dòng. Không nạp lại trial, không xem trial là evaluation độc lập. Giữ raw tại vị trí hiện có; mọi flag, chuẩn hóa và membership đều lưu riêng.

Một **mẫu** là một mục tiêu trong một context; số mẫu không phải số cụm. **Schema** là danh sách trường của file. **Metadata** phục vụ truy vết/phân tích, không tự trở thành feature. Std là độ phân tán đánh giá của người, không phải độ tự tin mô hình. Không có năm/nguồn corpus/thời kỳ; không suy nhãn lịch sử/hiện đại từ ID hoặc văn phong.

| Nhánh | File | Byte | SHA-256 |
| --- | --- | ---: | --- |
| en_nn | `data/train/en-nn-train.tsv` | 739246 | `98ca6db405a5cdea6cebaaed323ef6e50e3ddd2926b908bbfded8badeee8d895` |
| en_pv | `data/train/en-pv-train.tsv` | 268351 | `492aa8f2393c154e3301267cbe1a8a434e345c7214ec07682a3171e326b31072` |

| Nhánh | Cột | Ý nghĩa | Ví dụ thật từ dòng đầu | Vai trò |
| --- | --- | --- | --- | --- |
| en_nn | `ContextID` | ID dùng nối dòng nguồn và prediction | 10001 | metadata (không vào B1/B2) |
| en_nn | `Compound` | Danh từ ghép mục tiêu | account book | input |
| en_nn | `Mod` | Modifier, thành phần bổ nghĩa | account | metadata (không vào B1/B2) |
| en_nn | `Head` | Head, thành phần trung tâm | book | metadata (không vào B1/B2) |
| en_nn | `ModAvg` | Điểm trung bình cho modifier | 3.500000 | target |
| en_nn | `ModStd` | Độ phân tán đánh giá modifier | 1.779510 | metadata (không vào B1/B2) |
| en_nn | `HeadAvg` | Điểm trung bình cho head | 3.800000 | target |
| en_nn | `HeadStd` | Độ phân tán đánh giá head | 1.549190 | metadata (không vào B1/B2) |
| en_nn | `Context` | Câu sử dụng mục tiêu | Correspondence, letterbooks, account books, and writings together with family papers concerning Charles, Clement, Edward, James, Nicholas, Thomas, and William S. Biddle. | input |
| en_pv | `ContextID` | ID dùng nối dòng nguồn và prediction | 30001 | metadata (không vào B1/B2) |
| en_pv | `ParticleVerb` | Động từ-tiểu từ mục tiêu | crack down | input |
| en_pv | `Base` | Động từ cơ sở | crack | metadata (không vào B1/B2) |
| en_pv | `Particle` | Tiểu từ | down | metadata (không vào B1/B2) |
| en_pv | `Avg` | Điểm trung bình PV | 0.300000 | target |
| en_pv | `Std` | Độ phân tán đánh giá PV | 0.483046 | metadata (không vào B1/B2) |
| en_pv | `Context` | Câu sử dụng mục tiêu | The manifesto also committed the Tories to backing new ' drugalyser ' technology to crack down on drug driving. | input |

Mod/Head và Base/Particle được audit để hiểu cấu trúc; protocol hiện tại B1 **chỉ đọc Compound**, B2 **Compound + Context**, không bổ sung Mod/Head riêng. Không có gold/Std/ID trong input mô hình.

## 3. Thống kê tổng quan

| Nhánh | Mẫu / ID duy nhất | Cụm | Thành phần duy nhất | Context/cụm min / median / mean / max |
| --- | --- | ---: | --- | --- |
| EN-NN | 3490 / 3490 | 446 | {'Mod': 313, 'Head': 88} | 1 / 8.0 / 7.825 / 10 |
| EN-PV | 1567 / 1567 | 160 | {'Base': 108, 'Particle': 4} | 5 / 10.0 / 9.794 / 10 |

| Nhánh/cột | Min | Q05 | Q25 | Median | Mean | SD | Q75 | Q95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NN/ModAvg | 0.0000 | 1.6300 | 3.2000 | 3.9000 | 3.6598 | 0.9964 | 4.4000 | 4.9000 | 5.0000 |
| NN/ModStd | 0.0000 | 0.3162 | 1.0594 | 1.5776 | 1.4392 | 0.5776 | 1.8856 | 2.2010 | 2.5473 |
| NN/HeadAvg | 0.0000 | 2.1000 | 3.4000 | 4.1000 | 3.9123 | 0.8866 | 4.6000 | 4.9000 | 5.0000 |
| NN/HeadStd | 0.0000 | 0.3162 | 0.7071 | 1.4757 | 1.3134 | 0.6274 | 1.8257 | 2.1705 | 2.5820 |
| PV/Avg | 0.1000 | 0.7000 | 1.6333 | 2.4000 | 2.4628 | 1.0891 | 3.2000 | 4.3000 | 5.0000 |
| PV/Std | 0.0000 | 0.7071 | 1.3166 | 1.5951 | 1.5324 | 0.4011 | 1.8135 | 2.0790 | 2.4060 |

SD ở bảng là độ lệch chuẩn population **giữa các dòng của cột**, khác với giá trị Std trong từng dòng. Quantile dùng nội suy tuyến tính tại vị trí `(n−1)×p`. Không có Avg ngoài [0,5], Std âm hoặc số không hữu hạn.

| Nhánh | Độ dài | Min | Median | Mean | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| EN-NN | token regex | 7 | 24.0 | 25.16 | 56 |
| EN-NN | ký tự Unicode | 35 | 138.0 | 142.39 | 319 |
| EN-PV | token regex | 7 | 24 | 24.09 | 48 |
| EN-PV | ký tự Unicode | 31 | 124 | 127.49 | 291 |

Đếm “từ” ở audit bằng regex Unicode `\w+`: tách dấu nối/dấu nháy, khác cách đếm whitespace trong khảo sát ban đầu và khác subword tokenizer. Chưa tải tokenizer nên chưa kết luận max length 256 giữ hết mọi câu.

## 4. Trùng và gần trùng

| Loại | EN-NN: nhóm / dòng dư | EN-PV: nhóm / dòng dư |
| --- | --- | --- |
| id | 0 / 0 | 0 / 0 |
| full_row | 0 / 0 | 0 / 0 |
| target_context_raw | 0 / 0 | 0 / 0 |
| context_raw | 4 / 4 | 1 / 1 |
| context_normalized | 4 / 4 | 1 / 1 |
| target_context_normalized | 0 / 0 | 0 / 0 |
| conflicting_labels | 0 / 0 | 0 / 0 |
| context_different_target | 4 / 4 | 1 / 1 |

“Dòng dư” chỉ là số đếm n−1 trong mỗi nhóm, **không phải yêu cầu xóa**. Bốn cặp NN: `11382/13476`, `13134/13458`, `13142/13457`, `13316/13333`. Cặp PV: `30312/31455`. Không có cùng Compound+context với nhãn xung đột.

Near-duplicate là **ứng viên gần trùng**, không phải xác nhận cùng nghĩa. Dùng hai chỉ mục độc lập, token-set Jaccard ≥0.8 **hoặc** character-5gram Jaccard ≥0.85; bỏ cặp context chuẩn hóa bằng nhau vì đã audit riêng. Tìm được 0 ứng viên NN và 0 ứng viên PV. File candidate vẫn được xuất với header dù rỗng. Thuật toán bao phủ cặp đạt hai ngưỡng lexical này nhưng không bảo đảm phát hiện paraphrase/ngữ nghĩa tương đương.

Policy bảo thủ: mọi ứng viên phát hiện được nằm cùng group trước split, vẫn giữ trạng thái chưa xác nhận ngữ nghĩa. Trong lần chạy này không có ứng viên cần giải quyết; **không tuyên bố loại hết rò rỉ near-duplicate**. Group context lớn nhất là 2 dòng; group B nối thêm Compound lớn nhất 20 dòng/2 cụm, do câu dùng chung; không có chuỗi near-duplicate tạo group lớn.

## 5. Mục tiêu xuất hiện trong câu

| Loại theo thuật toán | EN-NN | EN-PV |
| --- | ---: | ---: |
| case_insensitive_contiguous | 7 | 11 |
| discontinuous_candidate | 0 | 243 |
| exact_contiguous | 2485 | 502 |
| inflection_or_punctuation_candidate | 914 | 701 |
| multiple_candidates | 84 | 64 |
| unresolved | 0 | 46 |

Exact và case-insensitive cần biên từ, trả offset chỉ khi đúng một vị trí surface. “Inflection or punctuation candidate” dùng quy tắc số nhiều/đuôi động từ và danh sách bất quy tắc nhỏ; **không phải lemmatizer hoặc xác nhận hình thái**. Heuristic không gán selected span cho variant, nhiều ứng viên hoặc thành phần rời. PV xét base biến hình đứng trước particle trong khoảng ≤32 token regex; có thể nhầm quan hệ cú pháp hoặc bỏ lỡ thứ tự khác.

`unresolved` nghĩa thuật toán chưa xác định được, không phải cụm vắng mặt. Các mẫu variant/ambiguous/discontinuous/unresolved vẫn eligible cho Compound+Context; span review bắt buộc nếu chọn phương pháp cần span. Bảng đầy đủ có cả context gốc/chuẩn hóa và candidate offsets trong `reports/audit/*_manual_alignment.csv`.

### Ví dụ thật, chọn theo ID định trước

| Nhánh / ID | Mục tiêu | Điểm thật | Context gốc |
| --- | --- | --- | --- |
| en_nn/10001 | account book | ModAvg=3.500000, HeadAvg=3.800000 | Correspondence, letterbooks, account books, and writings together with family papers concerning Charles, Clement, Edward, James, Nicholas, Thomas, and William S. Biddle. |
| en_nn/11382 | glass cell | ModAvg=4.600000, HeadAvg=4.400000 | It consists of a small glass cell, containing a solution in which two zinc plates are immersed. |
| en_nn/13476 | zinc plate | ModAvg=4.300000, HeadAvg=3.000000 | It consists of a small glass cell, containing a solution in which two zinc plates are immersed. |
| en_pv/30002 | boil down | Avg=0.300000 | His philosophy of life may be boiled down into a single sentence: Carry on and you will be happy. |
| en_pv/30020 | put down | Avg=1.000000 | I put it down to stubbornness and laziness of thought. |
| en_pv/30312 | move down | Avg=3.800000 | They would move slowly up and down in front of the house, then pause and listen to the music, the tripping of feet, and the sound of merry voices. |
| en_pv/31455 | move up | Avg=3.000000 | They would move slowly up and down in front of the house, then pause and listen to the music, the tripping of feet, and the sound of merry voices. |

## 6. Các hình và cách đọc

### 6.1 Nhãn modifier/head

![Phân bố NN](../reports/figures/nn_scores.png)

Hai cột dùng cùng bin rộng 0,25 điểm; màu xanh là modifier, cam là head. Nhãn tập trung về phía cao nên baseline trung bình là mốc cần thiết; biểu đồ chỉ mô tả gold, chưa nói mô hình tốt hay context hữu ích.

### 6.2 Nhãn PV

![Phân bố PV](../reports/figures/pv_scores.png)

PV dùng cùng thang 0–5 và bin 0,25 như NN, nhưng phân bố khác. Vì khác target và dữ liệu, không dùng trực tiếp điểm B0 NN làm chuẩn cho PV; hiện chưa split hoặc chạy B0 PV.

### 6.3 Số context mỗi cụm

![Số context](../reports/figures/context_counts.png)

NN có nhiều cụm 5 hoặc 10 câu, PV phần lớn 10 câu. Cụm `viscosity work` chỉ có một mẫu, được train-only ở A; số câu không chứng minh số nghĩa hay số quan sát độc lập.

### 6.4 Độ dài

![Độ dài](../reports/figures/sentence_lengths.png)

Hai panel đo token regex và ký tự Unicode, thể hiện chi phí input khác nhau. Cần đo lại bằng tokenizer Kaggle trước truncation; dữ liệu ngắn theo từ vẫn có thể dài theo subword.

### 6.5 Quan hệ modifier/head

![Mod và head](../reports/figures/nn_mod_head.png)

Đường chéo là hai điểm bằng nhau, độ đậm ô là số mẫu. Có 350/3490 mẫu với gap tuyệt đối ≥2, đủ lý do báo riêng hai output; chưa chứng minh hai đầu ra model sẽ phân biệt được.

### 6.6 Biến thiên nội cụm

![Biến thiên theo context](../reports/figures/nn_context_variation.png)

Chọn sáu Compound đầu theo thứ tự alphabet có ≥5 context, không chọn theo điểm/lỗi. Trục ngang là thứ tự ID, **không phải thời gian**; đường nối chỉ giúp nhìn điểm giữa câu, không chứng minh chuyển nghĩa hoặc xu hướng lịch sử.

## 7. Phân tích EN-NN phục vụ nghiên cứu

Trung bình ModAvg−HeadAvg = -0.2525; mean gap tuyệt đối = 0.8773. Đây là mô tả annotation, không phải độ khó hoặc hiệu quả mô hình.

| Output | Tổng variance | Trong cụm | Giữa cụm | Tỷ lệ trong cụm |
| --- | ---: | ---: | ---: | ---: |
| ModAvg | 0.9928 | 0.2618 | 0.7310 | 26.4% |
| HeadAvg | 0.7861 | 0.2517 | 0.5344 | 32.0% |

Variance phân rã có trọng số theo số câu: tổng = mean bình phương độ lệch trong cụm + variance mean cụm theo trọng số. Khác biệt giữa các cụm chiếm phần lớn biến thiên, nhưng biến thiên nội cụm vẫn đáng phân tích; nó có thể gồm nghĩa, nguồn/chủ đề và bất đồng annotation, không chỉ context gây đổi nghĩa.

| Output | Pearson độ dài–điểm | Pearson Std–điểm | Pearson độ dài–Std |
| --- | ---: | ---: | ---: |
| ModAvg | 0.0100 | -0.5534 | -0.0018 |
| HeadAvg | -0.0213 | -0.6992 | 0.0233 |

Độ dài liên hệ tuyến tính rất yếu với điểm trong dữ liệu này; Std và điểm có tương quan âm. Thang bị chặn 0–5 và cách phân phối rating có thể góp phần tạo quan hệ này; không kết luận nhân quả và không đưa Std vào model. Không tính reliability/CI của annotation từ Std khi thiếu raw rating/số người.

## 8. Policy eligible và kết quả split

Giữ toàn bộ 3.490 NN và 1.567 PV eligible; không loại vì alignment heuristic chưa chắc. Không chỉnh điểm, không lấy trung bình duplicate nhãn. Lỗi schema/parse/ID/nhãn nghiêm trọng sẽ ngăn split và B0; audit flags ghi từng row key, eligible và exclusion reason. Không có mẫu bị loại trong lần chạy này.

A=`seen_context`: group context trùng/gần trùng bảo thủ, đưa đủ train coverage trước rồi cân bằng số dòng về 70/15/15. Cụm có ≤2 context group là train-only. B=`unseen_compound`: nối cả Compound chuẩn hóa và context group; giữ nguyên connected component. Cả hai dùng seed 42, tie order giả ngẫu nhiên, không dùng nhãn/metric chọn seed. Tái tạo được membership y hệt.

| Thiết lập | Tập | Mẫu | Cụm | Tỷ lệ mẫu | Context/component group |
| --- | --- | ---: | ---: | ---: | ---: |
| seen_context | train | 2443 | 446 | 70.000% | 2440 |
| seen_context | validation | 524 | 321 | 15.014% | 523 |
| seen_context | final | 523 | 316 | 14.986% | 523 |
| unseen_compound | train | 2444 | 316 | 70.029% | 315 |
| unseen_compound | validation | 523 | 64 | 14.986% | 63 |
| unseen_compound | final | 523 | 66 | 14.986% | 65 |

Tỷ lệ xấp xỉ70/15/15, số thực tế ở bảng; lệch làm tròn vì record/group nguyên. A không ép mọi cụm có cả validation và final; mọi cụm được đánh giá có trong train. B không chung Compound nhưng được chung modifier/head; chưa phải held-out cả hai thành phần hoặc unseen pretrain.

| Thiết lập/tập | Context/cụm: min / median / mean / max |
| --- | --- |
| seen_context/train | 1 / 5.0 / 5.478 / 10 |
| seen_context/validation | 1 / 1 / 1.632 / 5 |
| seen_context/final | 1 / 1.0 / 1.655 / 5 |
| unseen_compound/train | 1 / 8.0 / 7.734 / 10 |
| unseen_compound/validation | 3 / 10.0 / 8.172 / 10 |
| unseen_compound/final | 5 / 10.0 / 7.924 / 10 |

| Thiết lập/tập | Mod mean / min / max | Head mean / min / max |
| --- | --- | --- |
| seen_context/train | 3.6834 / 0.0 / 5.0 | 3.9367 / 0.1 / 5.0 |
| seen_context/validation | 3.6065 / 0.0 / 5.0 | 3.8722 / 0.4 / 5.0 |
| seen_context/final | 3.6029 / 0.0 / 5.0 | 3.8384 / 0.0 / 5.0 |
| unseen_compound/train | 3.6177 / 0.0 / 5.0 | 3.8899 / 0.0 / 5.0 |
| unseen_compound/validation | 3.7488 / 1.0 / 5.0 | 3.9704 / 1.4 / 5.0 |
| unseen_compound/final | 3.7677 / 0.0 / 5.0 | 3.9587 / 0.1 / 5.0 |

Có 375 mẫu final A nằm trong train B, và 359 mẫu final B nằm trong train A. A/B là protocol tách biệt **nhưng không độc lập thống kê**. Khóa lựa chọn phương pháp cho cả hai trước khi xem bất kỳ final; không sửa method theo final A rồi gọi final B là untouched. Không quy chênh lệch MAE A/B hoàn toàn cho generalization.

| Thiết lập | Assertion đạt | Trạng thái |
| --- | ---: | --- |
| seen_context | 18 / 18 | PASS |
| unseen_compound | 20 / 20 | PASS |

Các assertion gồm coverage, ID uniqueness/disjointness, source hash/schema/score, câu chuẩn hóa, A train coverage, B Compound disjointness, candidate near grouping và deterministic replay. Báo cáo JSON chứa kết quả từng assertion. Không còn candidate near-duplicate phát hiện được chưa được xử lý, nhưng paraphrase chưa phát hiện vẫn là giới hạn.

## 9. B0 trên validation, chưa đánh giá final

B0 tính hai mean chỉ từ train của từng setting, dự đoán cùng hai giá trị cho mọi mẫu validation. Đây là mốc tham chiếu đơn giản; model học cần chứng minh lợi ích vượt nó, không mặc định sẽ thắng.

| Thiết lập | Train mean Mod / Head | Val MAE Mod | Val MAE Head | MAE hai đầu theo dòng | E theo cụm (primary) | RMSE Mod / Head |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| seen_context | 3.6834 / 3.9367 | 0.8043 | 0.7092 | 0.7567 | 0.7453 | 1.0589 / 0.9292 |
| unseen_compound | 3.6177 / 3.8899 | 0.6829 | 0.6143 | 0.6486 | 0.6701 | 0.8331 / 0.7440 |

Pearson và Spearman là **NA** vì prediction hằng, không đổi thành 0. E là mean lỗi hai thành phần trong mỗi Compound rồi mean các Compound; MAE theo dòng là diagnostic để thấy khác biệt trọng số. Không có B0 prediction/metric final; phân bố nhãn final ở mục 8 chỉ phục vụ kiểm tra dữ liệu.

## 10. Lệnh chạy và bước tiếp theo

Core audit/split/B0 dùng Python ≥3.11 và thư viện chuẩn; biểu đồ cần Matplotlib. Không dùng torch, transformers hoặc tải backbone trong giai đoạn này. Matplotlib 3.11.2 đã được cài vào `.runtime/` riêng bằng Python 3.11.9; core scripts cũng được chạy dưới Python 3.14.6 trong sandbox. Không dùng binary wheel 3.11 với interpreter 3.14. Tám integrity tests đã chạy đạt, gồm quoted TSV, lỗi nhãn, word boundary/ambiguity, gần trùng, metric tính tay, membership bị sửa sai và CLI nonzero/B0 bị chặn.

Từ repository root, trong môi trường Python tương thích:

```powershell
python -m pip install -r requirements-audit.txt
python scripts/audit_data.py
python scripts/create_splits.py
python scripts/validate_splits.py
python scripts/run_mean_baseline.py
python scripts/build_data_report.py
python -m unittest discover -s tests -v
```

Nếu chỉ kiểm tra core không có Matplotlib: `python scripts/audit_data.py --no-figures`; sáu PNG cần chạy audit đầy đủ trước khi render report. Các script có `--config` cho config thay thế; báo cáo mặc định dùng config repository. `validate_splits.py` trả exit code khác 0 khi fail; B0 gọi validation lại trước khi chạy.

Tiếp theo: rà soát alignment nếu định dùng span; chuẩn bị notebook Kaggle đọc membership đã khóa và chạy smoke B1/B2. Không train lại split, không đưa ID/Avg/Std vào input. Chưa upload/commit dữ liệu dẫn xuất khi license chưa xác minh.

## Phụ lục: thuật toán và artifact

Parser: `csv.DictReader`, UTF-8-sig, tab delimiter, quotechar dấu nháy kép, strict; newline do csv xử lý. Record index khác số dòng vật lý khi có newline trong quoted cell. Chuẩn hóa NFC/casefold/gộp whitespace, giữ punctuation và accent; không sửa context gốc.

Split: groups được sắp thứ tự gốc, shuffle bằng `random.Random(42)`, bảo đảm coverage A, rồi phân bổ theo thứ tự shuffle để giảm tổng squared row deficit. Phiên bản `seed42_shuffled_group_squared_row_deficit_v2`; config/seed/python/source hashes/file hashes trong manifest. Không stratify theo nhãn hoặc chọn seed theo model scores. Bản development v1 dùng largest-first làm train thiên về cụm nhiều context nên đã thay v2 trước B1/B2; quyết định dựa số context/cụm, không theo metric B0/final. Chỉ v2 là membership hiện hành.

Artifact chính: `reports/audit/*_summary.json`, `*_flags.csv`, `*_duplicates.csv`, `*_near_candidates.csv/json`, `*_manual_alignment.csv`, `en_nn_compound_variation.csv`, `split_validation_*.json`; sáu PNG + manifest; `data/splits/en_nn/{seen_context,unseen_compound}/{train,validation,final}.csv`; `results/b0/*/{metrics.json,validation_predictions.tsv}`. Files dẫn xuất bị `.gitignore` chặn commit mặc định.

[Trang BTC](https://sites.google.com/view/semeval-compartment) được đối chiếu ngày 14/10/2026. Liên kết [CodaBench 17971](https://www.codabench.org/competitions/17971/) trả nội dung không liên quan trong lần truy cập, nên metric/template/rules/license vẫn **chưa xác minh**. Không nhận metric nội bộ là leaderboard metric. Thống kê file source được kiểm tra hash trước/sau audit; không có sửa source.
