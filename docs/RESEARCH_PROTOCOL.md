# CoSense — Protocol nghiên cứu EN-NN, phiên bản 2

Ngày: 14/10/2026. **Split v2 đã cố định và kiểm tra; B0 validation đã chạy. B1/B2 chưa huấn luyện; chưa đánh giá final.** Chi tiết: [DATA_REPORT.md](DATA_REPORT.md). Hướng nghiên cứu dài hạn: [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md).

Cập nhật17/10/2026: đã có [notebook Kaggle](../notebooks/kaggle_en_nn.ipynb), code prepare/smoke/train và [hướng dẫn upload/chạy](KAGGLE_EN_NN_GUIDE.md). Mặc định prepare; đã kiểm tra local, chưa xác nhận chạy Kaggle GPU. Không thay đổi membership hoặc protocol đối chứng dưới đây.

## 1. Câu hỏi và giả thuyết

“Thêm context có cải thiện dự đoán ModAvg/HeadAvg so với chỉ đọc Compound, trên cụm đã thấy và cụm chưa thấy khi fine-tune không?”

So sánh chính: B2−B1 trên B (`unseen_compound`). A (`seen_context`) là thiết lập bổ sung. Giả thuyết: thêm câu giúp giảm sai số và phân biệt cách dùng trong cùng Compound; kết quả không cải thiện ổn định hoặc diagnostics context không hỗ trợ sẽ làm yếu giả thuyết. Không mặc định pipeline đúng thì giả thuyết đúng.

## 2. Dữ liệu và split đã khóa

- Nguồn EN-NN duy nhất: `data/train/en-nn-train.tsv`, SHA-256 `98ca6db405a5cdea6cebaaed323ef6e50e3ddd2926b908bbfded8badeee8d895`. Trial đã gộp theo người dùng, chưa có merge manifest; không nạp lại hoặc dùng trial làm test.
- 3.490 mẫu eligible; không loại vì alignment heuristic chưa chắc. Source giữ nguyên; flags/membership lưu riêng; không chỉnh hoặc average nhãn.
- Seed split42, thuật toán `seed42_shuffled_group_squared_row_deficit_v2`, thư viện chuẩn Python. Config tại `configs/data_protocol.json`; hashes/runtime trong manifest. V1 development dùng largest-first khiến train thiên về cụm nhiều câu; chuyển v2 theo số context/cụm, không theo metric, trước B1/B2. Chỉ membership v2 hiện hành.
- A: 2.443 train (446 cụm), 524 validation (321), 523 final (316); mọi Compound evaluation có trong train. Cụm ≤2 context group train-only; thực tế singleton là `viscosity work`.
- B: 2.444 train (316 cụm), 523 validation (64), 523 final (66); không chung Compound. Nối Compound có câu chung, component lớn nhất20 mẫu/2 cụm.
- Cả hai bảo vệ context chuẩn hóa và candidate near-duplicate bằng grouping bảo thủ. Không có candidate vượt ngưỡng trong audit này; không bảo đảm không còn paraphrase. Ngưỡng token Jaccard≥0,80 hoặc char-5gram≥0,85; grouping không có nghĩa xác nhận ngữ nghĩa.
- B được chung Mod/Head riêng; unseen fine-tune không phải unseen pretrain hoặc cả hai thành phần mới.

A/B là protocol riêng nhưng cùng nguồn nên không độc lập thống kê. Có final A trong train B và ngược lại. **Khóa lựa chọn chung cho cả hai trước khi mở bất kỳ final nào**; không sửa method theo final A rồi gọi final B untouched. Không quy chênh lệch MAE A/B hoàn toàn cho generalization.

CSV là membership, không phải input đã trộn nhãn. Script kiểm tra coverage, ID/context/Compound, source/schema/hash/score và deterministic replay; fail trả nonzero và B0 bị chặn.

## 3. Đối chứng và input

| Model | Input học/dự đoán | Output |
| --- | --- | --- |
| B0 | hai mean fit **chỉ từ train** | ModAvg/HeadAvg hằng |
| B1 | **chỉ Compound**, chuỗi gốc | hai điểm regression |
| B2 | **Compound + Context**, text pair/template cố định sau smoke | hai điểm regression |

Không thêm Mod/Head riêng vào B1/B2 trong đối chứng chính. ContextID chỉ nối prediction với mẫu, không là feature; Avg/Std không vào input. Phương pháp có component/marker về sau là ablation riêng và cần version hóa trước final.

B1/B2 cùng DistilBERT tiếng Anh/revision/tokenizer, masked mean pooling, linear head2 output và MSE. Cùng seed list, effective batch, epochs tối đa, search budget và chọn checkpoint. Learning rate grid ban đầu `{1e-5,2e-5,3e-5}`, tối đa3 trial/model/setting với tuning seed42; ghi rõ nếu dùng config B cho A để tiết kiệm. Không cấp thêm trial riêng cho model thua.

Khởi đầu batch16, max length256, tối đa5 epochs, early stopping patience2. Đo VRAM/token length trên Kaggle; chỉ cắt khi giữ cách dùng mục tiêu trong Context. Compound ở segment riêng không thay context đã mất. Chưa dùng marker/span pooling vì heuristic chưa đáng tin; thay config do VRAM phải áp dụng công bằng, ghi deviation và không chỉnh theo final.

## 4. Model selection và final

Smoke có optimizer, fine-tune/tuning/train repeats chạy **Kaggle GPU**. B0 chỉ fit thống kê, đã chạy local theo yêu cầu lượt này và không cần GPU. Local chuẩn bị dữ liệu/code và phân tích artifact.

Validation chọn checkpoint/config bằng E ở mục5. Giữ mọi trial/status/cost và prediction row key, không chỉ kết quả thuận lợi. Seed train khởi đầu `{42,13,2027}` cho so sánh quan trọng khi có tài nguyên; ba seed chưa bảo đảm đủ evidence. Ghi seed split/train/permutation/bootstrap riêng.

Final chỉ mở sau khóa method, hyperparameters, ablations, seed list, metric/subgroup rules cho cả A/B. Hiện chỉ kiểm tra validity/phân bố nhãn final, không prediction/metric final. Nếu sửa method sau xem final, công khai reuse và thiết kế kiểm chứng độc lập phù hợp. Không chọn seed/ensemble/ngưỡng bằng final.

## 5. Metric nội bộ

Primary E cho mỗi Compound trọng số bằng nhau:

```text
e_i = (abs(clip(pred_Mod_i,0,5)-gold_Mod_i)
     + abs(clip(pred_Head_i,0,5)-gold_Head_i))/2
E = mean_c(mean_i_in_c(e_i))
Delta = E(B1)-E(B2)  # dương: B2 ít sai hơn
```

Secondary: MAE hai đầu theo dòng, MAE/RMSE riêng Mod/Head, Pearson/Spearman. Lưu raw/clipped prediction và tỷ lệ ngoài [0,5]; B0 vốn trong khoảng. Tương quan prediction hằng báo **NA** (`null` trong JSON), không đổi thành0. E khác mean theo dòng; kiểm tra bằng ví dụ tính tay đã có trong tests.

[Trang BTC](https://sites.google.com/view/semeval-compartment) xác nhận task và dẫn sang [CodaBench](https://www.codabench.org/competitions/17971/). Truy cập 14/10/2026 chưa trả hướng dẫn phù hợp, nên **metric/ranking/template chính thức chưa xác minh**. E là nội bộ, không nhận là leaderboard metric. Khi biết metric BTC, báo thêm và version hóa quyết định, không âm thầm đổi comparison.

## 6. Paired comparison và độ bất định

Lưu prediction trên cùng row key, so B1/B2 paired; báo từng seed, mean/std và Delta, không lấy best seed làm headline. Comparison chính là B2−B1 trên B; head/subgroup/ablation là secondary hoặc exploratory.

CI đề xuất: 10.000 paired cluster bootstrap draws, analysis seed2027, percentile CI95%; resample nguyên component anti-leakage ở B. A cần nhóm Compound nối bởi context khi phân tích nhiều câu cùng cụm, không dùng từng context group đơn lẻ làm độc lập. Giữ prediction paired, tính lại E đúng weighting rồi mean Delta qua các seed. Báo số nhóm; CI có điều kiện trên split/models đã fit, không bao gồm mọi bất định train/pretrain/split. Báo variability train riêng; không bootstrap từng câu như độc lập hoàn toàn.

`delta_min` chưa chốt: chọn effect đáng quan tâm bằng pilot/ý nghĩa thực tế trên validation, ghi lý do trước final. CI cắt0 hoặc seed trái dấu là evidence chưa rõ, không chứng minh tương đương. Nếu nhiều claim xác nhận, định trước family/correction; tránh tìm p-value thuận lợi sau nhiều phép thử.

## 7. Ablation và phân tích lỗi trước final

1. B1 train lại là đối chứng bỏ Context chính; context rỗng ở inference không đủ vì distribution shift.
2. B2 inference đổi context trong cùng Compound trên subset có ≥2 câu khác nhau **trong tập đang đánh giá**. Dùng5 permutation seeds `{7,17,27,37,47}`, không fixed point, không lấy câu train thay final. Lưu mapping/coverage, so đúng/sai context trên cùng subset; prediction thay đổi phải xét cùng error mới nói context hữu ích.
3. Nhóm định trước: gap `|ModAvg−HeadAvg|≥2`, độ dài, số câu/cụm và alignment class. Báo n/error riêng hai đầu; không chọn ngưỡng bằng final.
4. Residual nội cụm tách khác biệt giữa cụm và cách dùng trong cụm. Gold mean nội cụm chỉ dùng phân tích sau prediction, không vào inference; dispersion lớn một mình chưa là thành công.
5. Sau lỗi validation chọn tối đa một cải thiện, ghi hypothesis/remove-component ablation trước final. Ý tưởng sau khi xem dữ liệu/kết quả gắn nhãn exploratory.

## 8. Giới hạn và artifact

Metric/ablation cung cấp evidence hành vi, chưa chứng minh hiểu/reasoning/không ghi nhớ. Khác nhãn giữa câu có thể từ nghĩa, nguồn/chủ đề hoặc disagreement, không tự gọi là đổi nghĩa. English-only, một backbone và nguồn hiện tại giới hạn claim; chưa full-text literature review nên chưa claim novelty. Std không đủ suy CI annotation/reliability khi thiếu raw ratings/số người.

Hiện có audit/figures, hashes, membership/checks, B0 validation predictions/metrics. M3 cần notebook/config/model revision, best+last checkpoint/tokenizer, registry/log/runtime/cost và prediction paired. Data/derived outputs bị ignore; không commit/upload/publish khi license chưa rõ. Chưa có kết quả B1/B2, không điền số giả.
