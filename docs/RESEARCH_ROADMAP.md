# CoSense — Roadmap nghiên cứu từ câu hỏi đến bằng chứng cho paper

Ngày: 14/10/2026. **Cập nhật:** đã audit EN-NN/EN-PV tiếng Anh, cố định split EN-NN và chạy B0 validation; xem [DATA_REPORT.md](DATA_REPORT.md) và [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). Chưa huấn luyện B1/B2 hoặc đánh giá final. Tài liệu bổ sung [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md). Chỉ triển khai English; neural training/smoke có optimizer/train lại ablation trên **Kaggle GPU**, local chuẩn bị và phân tích artifact.

Hãy bắt đầu bằng câu hỏi: **“Sau thí nghiệm này, mình sẽ biết điều gì mà trước đó chưa biết?”** Nếu câu trả lời chỉ là “mô hình đạt một điểm số”, cần làm rõ câu hỏi trước khi tiêu thêm GPU. Một paper có thể đóng góp phương pháp, đánh giá hoặc phân tích hiện tượng; không bắt buộc phải có kiến trúc mới. Tuy nhiên, chạy một backbone với một dataset cũng chưa tự tạo đóng góp khoa học.

Các lựa chọn dưới đây là **protocol đề xuất cho CoSense**, cần khóa sau audit/pilot, không phải luật bắt buộc của ACL hay lời hứa được nhận paper. Tham khảo yêu cầu minh bạch về thiết lập, tài nguyên và giới hạn từ [ARR Responsible NLP Research](https://aclrollingreview.org/responsibleNLPresearch/). Luật SemEval, metric và template chính thức vẫn ở trạng thái chưa xác minh trong roadmap chính.

## 1. Chọn một câu hỏi đủ hẹp và đo được

**Câu hỏi trung tâm:** thêm câu sử dụng có cải thiện dự đoán độ hợp thành của modifier/head trên danh từ ghép chưa xuất hiện trong fine-tune không, và cải thiện đó có phù hợp với việc dùng ngữ cảnh riêng của từng cách dùng không?

Tách thành hai câu hỏi để tránh suy diễn:

| Câu hỏi | Phép đo/thiết lập | Kết luận được phép |
| --- | --- | --- |
| RQ1: Context có tăng chất lượng dự đoán cho cụm chưa fine-tune? | B2 vs B1 trên split B | lợi ích dự đoán của input context trong model/data/protocol đã thử |
| RQ2: Model có phụ thuộc vào context đúng của từng cách dùng? | thay context trong cùng cụm, phân tích biến thiên nội cụm | độ nhạy hành vi với context và độ phù hợp với nhãn; chưa suy ra cơ chế hiểu nghĩa |
| RQ3 phụ: Model có xử lý khác biệt modifier/head? | sai số hai output và nhóm gap định trước | mức thành công/thất bại của dự đoán thành phần |

A (cụm đã thấy, câu mới) là đối chiếu để hiểu điều kiện áp dụng; B (cụm chưa thấy khi fine-tune) là thiết lập chính. Không so trực tiếp MAE A/B rồi quy mọi chênh lệch cho “memorization”: thành phần cụm, độ khó và nhãn của hai final khác nhau. Chưa có metadata thời gian nên không thêm claim historical generalization. Một encoder tiếng Anh không cho phép kết luận về mọi LLM, mọi ngôn ngữ hoặc toàn bộ idioms.

**Bài tập trước M1:** viết ba câu: “Tôi nghiên cứu …”; “Tôi so sánh … với … trong điều kiện …”; “Kết quả … sẽ làm tôi phải sửa giả thuyết …”. Đây là cách biến ý tưởng rộng thành một câu hỏi có thể kiểm tra.

## 2. Đọc literature để tìm khoảng trống thực sự

Hiện mới đọc abstract của bốn bài dưới đây. Chưa có căn cứ nói CoSense là phương pháp đầu tiên hoặc mới. Không lấy thứ hạng cuộc thi làm bằng chứng novelty.

| Thứ tự đọc | Nguồn | Câu hỏi cần trả lời sau khi đọc toàn văn |
| --- | --- | --- |
| 1 | [NCTTI, ACL 2021](https://aclanthology.org/2021.acl-long.212/) | type/token được định nghĩa và đánh giá thế nào? Nguồn/annotation có quan hệ gì với task hiện tại? |
| 2 | [Compound semantics in BERT, EACL 2023](https://aclanthology.org/2023.eacl-main.110/) | biểu diễn/layer/pooling và estimator nào được so sánh? Đâu là khác biệt Mod/Head? |
| 3 | [DICE, ACL 2025](https://aclanthology.org/2025.acl-long.362/) | thiết kế contrastive kiểm soát yếu tố nào? Có thể học cách đánh giá context, nhưng khác task regression NN ở đâu? |
| 4 | [Memorization or Reasoning?, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1099/) | tác giả dùng bằng chứng nào để bàn về memorization/reasoning? Giả định nào không chuyển được sang CoSense? |

Đọc theo vòng: abstract/introduction để xác định câu hỏi → experimental setup để hiểu dữ liệu/đối chứng → bảng kết quả/ablation/limitations → method và appendix/code khi cần tái lập. Viết một trang ghi chú mỗi bài, có số trang/section hỗ trợ từng nhận định. Phân biệt “tác giả kết luận”, “thí nghiệm quan sát” và “mình suy đoán”. Nếu chưa đọc appendix thì ghi rõ.

Tạo `docs/LITERATURE_NOTES.md` ở M4, dùng bảng: `paper | task | label/input | split | model | controls | metric | evidence | limitation | quan hệ với RQ1/RQ2`. Tìm thêm công trình trực tiếp về context-conditioned compositionality, component-level prediction, held-out lexical groups và evaluation shortcuts; đọc baseline/task paper BTC khi có. Không so các con số khác dataset/split/metric như cùng bảng xếp hạng.

**Điều kiện chuyển bước:** có thể nói chính xác một thiếu hụt có kiểm chứng trong phạm vi literature đã đọc. Nếu các bài đã trả lời câu hỏi tương tự, chọn replication có kiểm soát, kiểm tra điều kiện mới, hoặc sửa câu hỏi. Replication cần giải thích giá trị; chỉ đổi dataset chưa chắc đủ thành đóng góp mới.

## 3. Viết protocol trước khi xem final

Protocol là bản cam kết nội bộ, không phải đăng ký ngoài. Bản v1 đã có tại `docs/RESEARCH_PROTOCOL.md`; per-run experiment cards sẽ thêm ở M3. Có thể sửa trong development bằng train/validation nhưng ghi ngày/lý do; khóa bản xác nhận trước final.

Mẫu phiếu có thể dùng ngay:

```text
Experiment ID / ngày / trạng thái: planned | running | failed | completed
RQ và hypothesis:
Đối chứng / yếu tố duy nhất thay đổi:
Nguồn + source hash / split A hoặc B + manifest hash:
Model revision / code version / input format / pooling / output order:
Train seeds / split seed / seed permutation và bootstrap:
Hyperparameter search space / số trial / checkpoint selection:
Metric chính / metric phụ / subset / quy tắc clipping:
Ngân sách GPU dự kiến / điều kiện dừng:
Kết quả nào hỗ trợ, làm yếu hoặc chưa đủ phân biệt hypothesis:
Artifact cần lưu / deviation so với protocol:
```

Phân biệt ba loại seed: seed split quyết định thành viên tập; seed train ảnh hưởng khởi tạo/dropout/thứ tự batch; seed phân tích quyết định permutation/bootstrap. Thay train seed không làm số cụm độc lập tăng lên. Một bộ final không trở thành “mới” vì đổi seed train.

### Metric chính được đề xuất

Để mỗi cụm có trọng số như nhau, dùng **MAE trung bình theo cụm và hai thành phần** làm metric nghiên cứu chính (càng thấp càng tốt):

```text
e_i = (|clip(pred_Mod_i, 0, 5) - gold_Mod_i|
     + |clip(pred_Head_i, 0, 5) - gold_Head_i|) / 2
MAE_c = mean(e_i trên các câu của cụm c)
E = mean(MAE_c trên các cụm được đánh giá)
Delta = E(B1) - E(B2)       # dương: B2 tốt hơn
```

Báo thêm MAE theo dòng để thấy ảnh hưởng weighting, MAE/RMSE riêng Mod/Head, prediction thô và tỷ lệ ngoài [0,5], Pearson/Spearman có xử lý undefined. Đây là **metric nội bộ đề xuất**, không nhận là metric BTC. Roadmap chính đã dùng E cho chọn validation checkpoint; khi triển khai, giữ tên metric và công thức rõ trong config và kiểm tra bằng ví dụ tính tay.

Định trước `delta_min`: mức giảm MAE nhỏ nhất bạn coi đáng quan tâm trên thang 0–5. Chưa có căn cứ định con số ở lượt này; chọn và giải thích bằng pilot/ý nghĩa thực tế **trước final**, không chọn theo effect quan sát trên final. Ý nghĩa thống kê và ý nghĩa thực tế là hai vấn đề khác nhau.

Không chọn ra seed tốt nhất để báo headline. Nếu checkpoint mỗi seed được chọn theo validation, final phải báo toàn bộ seed đã định trước. Nếu còn quyết định ensemble, phải chọn trên validation và báo riêng, không thay lén bảng single-model.

## 4. Bảo vệ tính hợp lệ của dữ liệu và split — M1

Đơn vị dự đoán là một mục tiêu trong một câu. Đơn vị cần xét độc lập khi phân tích thường là cụm hoặc nhóm cụm có liên kết câu, vì các câu cùng cụm không phải những quan sát hoàn toàn độc lập.

EN-NN có 3.490 mẫu/446 cụm, singleton và bốn cặp câu chung giữa mục tiêu; split v1 đã hoàn thành. 3.490 dòng không tương đương 3.490 trường hợp khái quát độc lập. Chi tiết/hashes nằm ở DATA_REPORT và manifests.

M1 phải xuất:

- Bảng schema/missing/range/duplicate; provenance trial đã gộp, không dùng trial làm test.
- A có train coverage cho cụm đánh giá, không chia câu trùng qua tập; singleton train-only theo quy tắc.
- B giữ Compound trong một tập, nối các cụm có câu trùng/gần trùng đáng tin vào cùng component; lưu component membership để phân tích thống kê.
- Tokenization/alignment coverage, mọi loại trừ và lý do; so phân bố mẫu giữ/lọc để phát hiện selection bias. Hai model phải đánh giá trên cùng tập hợp mẫu.
- Final nội bộ khóa; official evaluation là tập khác. Không đưa Avg/Std/ContextID vào input học.

**Kiểm tra tư duy:** nếu mô hình chỉ đọc tên cụm mà vẫn làm tốt, split A có thể cho phép học đặc tính type-level. Điều đó hợp lệ cho câu hỏi A, nhưng không đủ trả lời câu hỏi B. Chưa fine-tune trên một Compound không bảo đảm modifier/head của nó cũng chưa fine-tune: lưu tỷ lệ shared components, phân tích như biến mô tả trên B. Component-held-out là thí nghiệm mở rộng khác; không thêm tùy tiện nếu dữ liệu quá nhỏ.

**Điều kiện chuyển bước:** audit và assertions pass; membership tái lập; leakage đã biết được xử lý; giới hạn gần trùng được ghi. Nếu split hỏng, sửa split trước train, không giải thích metric của split hỏng như kết quả nghiên cứu.

## 5. Baseline và pilot — M2/M3

Baseline giúp xác định mô hình thêm được gì. Theo RESEARCH_PROTOCOL v2, B0 chỉ có trung bình train; B1 **chỉ đọc Compound**; B2 đọc **Compound+Context**, không thêm Mod/Head riêng. Giữ DistilBERT/tokenizer/pooling/head, search budget và checkpoint selection giống nhau. Khác input là yếu tố chủ động thay đổi; số token/compute phát sinh phải báo.

Một đối chứng phụ rẻ cho A: **trung bình train theo Compound**, fallback về B0 nếu cụm không có trong train. Nó đo mức lợi ích vượt việc ước lượng điểm riêng của cụm đã học. Trên B mọi cụm đều unseen nên đối chứng này trở về B0, không xem là baseline mạnh riêng của B. Chỉ tính mean từ train, không lấy nhãn val/final của cụm.

### Trình tự pilot tiết kiệm Kaggle

1. Smoke một lần cho mỗi input contract B1/B2 với 1–2 optimizer steps; save/load; kiểm tra schema/labels/target retention. Chưa đánh giá final.
2. Chạy B1/B2 trên B với seed 42, config khởi đầu; theo dõi validation và lỗi. Một seed ở đây phục vụ debug và sizing, chưa đủ kết luận.
3. Đo thời gian mỗi epoch, peak VRAM và cost/artifact. Kiểm tra B0 và mức sai số từng đầu ra.
4. Nếu cần tuning, cùng grid learning rate ba giá trị và cùng budget cho B1/B2; chọn bằng validation E. Không tăng trial cho một model chỉ vì nó đang thua.
5. Chốt cấu hình và mở A; sau đó mới chạy xác nhận nhiều seed/ablations cần thiết. Không chạy cả lưới thí nghiệm trước khi biết pipeline đúng.

Pilot trả lời: pipeline có học không, ngân sách có đủ không, metric có nhất quán không? Nó **không** trả lời chắc chắn model dùng context hay có đóng góp mới. Số trial, cả failed/interrupted run, cần giữ trong registry.

## 6. Ma trận thí nghiệm gắn với từng claim

| ID | Thí nghiệm | Vai trò | Cần train thêm? |
| --- | --- | --- | --- |
| E0 | B0 trên A/B; mean theo Compound trên A | chuẩn tham chiếu | không, chỉ fit thống kê train |
| E1 | B1/B2 trên B, cùng train seeds | đối chứng chính RQ1 | có, Kaggle GPU |
| E2 | B1/B2 trên A, cùng protocol | đối chiếu cụm đã thấy | có, Kaggle GPU |
| E3 | checkpoint B2, đổi Context sang câu khác cùng Compound | kiểm tra hành vi RQ2 | không; inference và phân tích |
| E4 | cùng prediction E1/E2, đo biến thiên trong cụm và gap Mod/Head | hiểu lỗi và RQ2/RQ3 | không |
| E5 | một cải thiện có lý do + remove-component ablation | đánh giá đóng góp phương pháp nếu có | có, Kaggle GPU |
| E6 | backbone/split bổ sung đã định trước | kiểm tra phạm vi/độ bền nếu claim cần | tùy thiết kế, có thể tốn nhiều run |

### E3: hoán đổi context phải làm cẩn thận

Chỉ dùng cụm có ít nhất hai câu khác nhau trong **tập đang đánh giá**; hoán đổi không có fixed point, giữ target fields của dòng, giữ gold gốc để đo độ lệch so với dự đoán cách dùng gốc. Không lấy câu từ train để thay vào final. Lưu mapping và 5 permutation seed định trước; báo coverage và kết quả trung bình từng permutation, không chọn permutation xấu nhất.

So B2 đúng/sai context trên **cùng subset**, đo thay đổi prediction và thay đổi sai số. Prediction thay đổi nhưng error không xấu đi có thể là nhạy với input mà chưa dùng context hữu ích. Đúng context cải thiện ổn định là bằng chứng phù hợp với việc khai thác context trong protocol này, chưa chứng minh cơ chế reasoning. Hoán đổi làm thay đổi dữ liệu quan sát, các yếu tố chủ đề/độ dài cũng thay đổi; không gọi đây là bằng chứng nhân quả về semantics.

Không dùng context rỗng tại inference làm đối chứng duy nhất: nó có thể gây distribution shift. B1 được train lại là đối chứng bỏ context chính. Nếu muốn train B2 với context hoán đổi, đó là thí nghiệm khác, phải ghi quy tắc trong train/val và ngân sách; không trộn với E3.

### E4: tránh nhầm type-level với context-level

Model có thể dự đoán tốt khác biệt **giữa cụm**, nhưng gần như không phân biệt **các câu của cùng cụm**. Ngoài MAE chung, với cụm nhiều câu tính residual bằng cách trừ mean trong cụm cho gold và prediction rồi đo sai số/tương quan residual. Mean gold này chỉ phục vụ phân tích sau prediction, không phải feature hoặc target preprocessing rò rỉ vào inference. Báo số cụm/câu và trường hợp không có biến thiên; tương quan từng cụm với vài câu rất nhiễu, không dùng nó một mình.

Báo gap prediction so với gap gold trên nhóm `|ModAvg−HeadAvg| ≥ 2` định trước. Không dùng nhãn subgroup để thay model trên final. Các ngưỡng/nhóm khác nghĩ ra khi xem lỗi là exploratory, chưa thành kiểm chứng độc lập.

## 7. Độ bất định, effect size và kết luận — M6

**Một số trung bình không đủ.** Với split đã khóa, dùng seeds `{42, 13, 2027}` làm khởi đầu; nếu biến động cao hoặc claim cần mạnh hơn, tăng số seed theo ngân sách trước khi mở final. Ba seed không bảo đảm đủ power. Không dừng thêm seed chỉ vì vừa đạt một kết quả thuận lợi.

Tách hai nguồn bất định:

1. **Train randomness:** báo E từng seed, mean/std và paired Delta theo seed B1/B2. Cùng số seed không khiến hai model có quỹ đạo huấn luyện giống nhau.
2. **Mẫu đánh giá:** paired cluster bootstrap trên các component chống rò rỉ ở B và group tương ứng đã lưu ở A. Lấy lại nguyên nhóm, không lấy từng câu như độc lập; giữ cùng mẫu resample cho hai model.

Đề xuất 10.000 bootstrap draws, seed phân tích cố định và percentile CI 95% cho Delta. Trong mỗi draw, resample component nguyên khối, tính lại E đúng weighting đã định, rồi lấy mean Delta qua các training seed. CI này **có điều kiện trên tập seed/model đã fit và split**, không tự bao gồm mọi biến động train/pretrain hay split. Báo train variability riêng; nếu cần kết luận mạnh hơn, thiết kế hierarchical resampling hoặc nhiều split với hướng dẫn thống kê phù hợp, không gắn nhãn “all uncertainty” cho CI đơn giản.

Số component final thực tế phải được báo. Khi ít nhóm hoặc CI rộng, kết luận thiếu bằng chứng thay vì kết luận “hai model bằng nhau”. Nếu định p-value, chọn phép kiểm phù hợp metric và phụ thuộc nhóm; không mặc định t-test trên tất cả câu. Đọc [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/) để hiểu việc lựa chọn phép kiểm; hiện mới xem abstract, chưa dùng bài này để xác nhận chi tiết bootstrap đề xuất.

Đặt một comparison chính (B2−B1 trên B). Các head/subgroup/ablations là secondary hoặc exploratory. Nếu kiểm định nhiều claim xác nhận, định trước family và cách hiệu chỉnh như Holm; không tìm p<0,05 trong hàng chục bảng rồi chỉ báo bảng thuận lợi. Effect size/CI và tính hợp lệ thiết kế quan trọng hơn việc có một p-value nhỏ.

| Kết quả sau chạy xác nhận | Cách diễn giải |
| --- | --- |
| Delta dương ổn định, CI nằm trên 0 và effect đủ lớn theo delta_min | hỗ trợ lợi ích dự đoán của B2 trong thiết lập B đã thử |
| Delta dương nhỏ, CI hẹp nhưng effect không đáng quan tâm | có chênh lệch, nhưng lợi ích thực tế hạn chế |
| CI cắt 0/rộng hoặc seed trái dấu | bằng chứng chưa đủ nhất quán; không chọn seed tốt nhất để kết luận |
| B2 không hơn B1; E3 cũng không cho lợi ích context đúng | giả thuyết ban đầu chưa được hỗ trợ; kiểm tra ceiling/nhiễu/capacity/shortcut trước khi giải thích cơ chế |
| B2 hơn B1 nhưng không tốt hơn ở residual nội cụm | lợi ích có thể chủ yếu từ tín hiệu giữa cụm/chủ đề; hạn chế claim context-specific |

Không dùng Std có sẵn để suy ra confidence interval của mean annotation: chưa có số người đánh giá và raw rating nên thiếu thông tin. Nó giúp mô tả label disagreement, không phải uncertainty của model. Không đo inter-annotator reliability bằng Avg/Std đơn thuần.

## 8. Phân tích lỗi trước khi đề xuất phương pháp — M4

Lấy toàn bộ validation predictions, phân nhóm định trước và xem một mẫu ngẫu nhiên cộng các lỗi lớn. Ghi row key, input, gold, pred B0/B1/B2, loại lỗi và mức chắc chắn. Ví dụ tự tạo trong demo không thay thế phân tích lỗi dữ liệu thật.

| Quan sát cần kiểm chứng | Hướng có thể thử | Đối chứng cần có |
| --- | --- | --- |
| lỗi concentrated ở variant/ambiguous target | span-aware pooling/marker nếu alignment đáng tin | cùng data/backbone, bỏ span feature; coverage và lỗi alignment |
| hai output thường co về giống nhau, gap bị mất | biểu diễn riêng Mod/Head | baseline head hiện tại; remove-component ablation; chi phí tham số |
| một số lỗi lớn chi phối MSE, gắn disagreement cao | Huber loss | cùng input/head/budget; MSE; không chọn mẫu bằng nhãn final |
| lợi ích context yếu trên B | kiểm tra pooling/input/sensitivity trước khi tăng model | B1/B2 đúng protocol; nếu tăng backbone, so cả B1 lẫn B2 |

Đây là cây quyết định **có điều kiện**, chưa phải đề xuất phương pháp mới. Chọn một cải thiện, viết cơ chế kỳ vọng và thí nghiệm có thể bác bỏ nó. Nếu chỉ đổi loss cho tăng điểm một ít mà không có phân tích mới, giới hạn claim tương ứng.

Nêu rõ exploratory vs confirmatory: lỗi validation giúp tạo hypothesis; sau đó khóa hypothesis và đánh giá bằng final độc lập. Nếu đã xem final rồi sửa method, final đó trở thành development evidence. Phải công khai việc reuse và thiết kế đánh giá mới phù hợp; không gọi nó là final untouched.

## 9. Dùng Kaggle như hệ thống thực nghiệm

Notebook là giao diện chạy; config, code và registry mới xác định thí nghiệm. Không để các biến vô hình trong nhiều cell tạo một cấu hình không tái lập được.

Các cell dự kiến: runtime/GPU → input/code/data hashes → dependency/model revision → config và guards split → smoke → training hoặc resume → validation/checkpoint → export artifacts. Cell final evaluation mặc định tắt cho development; chỉ bật khi đã khóa protocol. Mỗi run có ID duy nhất, không đè output.

```text
results/experiments/<run_id>/
  config.json               # input/model/train/evaluation đầy đủ
  environment.json          # Python/packages/GPU và code version
  data_manifest.json        # source/split/template hash khi có
  history.jsonl             # epoch/step, train loss, validation, thời gian
  predictions_val.tsv       # đủ row key và hai prediction
  metrics.json              # weighting, raw/clipped, coverage
  run_status.json           # completed/failed/interrupted + lý do
checkpoints/<run_id>/
  best/                     # checkpoint chọn bằng validation + tokenizer
  last/                     # model/optimizer/scheduler/RNG cho resume
results/experiment_registry.csv  # mọi trial, seed, status, cost, artifact path
```

Các file này là **đầu ra dự kiến**, chưa được tạo. Registry nên có hypothesis ID, setting, baseline, config hash, trial/seed, GPU type, thời gian, best validation epoch, và lý do exclusion nếu có. Giữ metric/prediction cả run không thuận lợi. Run lỗi không có metric thì ghi missing với lý do, không điền số giả.

Ước lượng budget sau pilot: tổng giờ GPU ≈ tổng thời gian từng run + tuning + smoke + failed/resume overhead. Ví dụ với grid ba learning rate trên B: 2 model × 3 config × 1 seed = 6 tuning runs; nếu cấu hình được chọn không đổi, seed 42 hợp lệ có thể tái dùng trong ba seed xác nhận, thêm 4 run cho seed 13/2027. A với config đã khóa cần 6 run nữa. Đây là phép đếm ví dụ, không yêu cầu chạy tất cả trước M3 và chưa bao gồm E5/E6. Phân biệt pilot dùng lại được với run đổi code/split phải chạy lại.

Khi quota hạn chế: ưu tiên E0/E1 và tính đúng pipeline, rồi E2/E3/E4; thu hẹp claim thay vì bỏ controls thiết yếu. Một backbone có thể hỗ trợ kết luận về backbone đó. Nếu claim chung về nhiều mô hình/ngôn ngữ, cần evidence mở rộng; chưa có nguồn lực thì chưa viết claim đó.

Nếu đổi GPU type hoặc phải đổi batch/precision, lưu môi trường và áp dụng protocol công bằng. Resume phải giữ optimizer/RNG; ghi session boundary, không giả định bitwise identical qua mọi GPU. Thực hiện lại một run từ phiên sạch và so artifact theo tolerance khi cần kiểm tra tái lập. Báo cả ngân sách tìm cấu hình; abstract của [Dodge et al., EMNLP 2019](https://aclanthology.org/D19-1224/) giải thích vì sao test score đơn lẻ không đủ để đánh giá so sánh model. Chưa đọc toàn văn bài này trong lượt lập kế hoạch.

## 10. Từ kết quả đến một paper có luận điểm

Trước khi viết abstract, tạo **bảng claim–evidence**. Mỗi claim phải có thí nghiệm, bảng/hình, phạm vi và counter-evidence. Không điền số cho đến khi có artifact thật.

| Claim dự kiến, chưa chứng minh | Evidence tối thiểu | Giới hạn bắt buộc |
| --- | --- | --- |
| Context cải thiện dự đoán EN-NN ngoài cụm fine-tune | E1, cùng protocol, seeds/effect/CI | một nguồn/split/backbone; pretrain exposure không biết |
| Context đúng có ích cho cách dùng cụ thể | E3 + residual nội cụm E4 trên đủ coverage | diagnostic input intervention; không chứng minh hiểu hay nhân quả semantics |
| Cải thiện đề xuất xử lý lỗi thành phần | E5 vs B2 mạnh, remove-component ablation, lỗi theo nhóm | phân tích có điều kiện; chi phí/coverage; chưa kết luận đa ngôn ngữ |

Cấu trúc bản thảo đề xuất:

1. **Introduction:** vấn đề → khoảng trống có citation → RQ → đóng góp cụ thể đã có evidence. Không kể quá trình làm project thành đóng góp.
2. **Related work:** đối chiếu task/setting/controls, giải thích khác biệt có thật; không chỉ liệt kê tóm tắt paper.
3. **Data/task:** schema/scale, provenance trial, số cụm/câu/split, leakage và loại trừ, hạn chế thời gian/annotation/license.
4. **Method:** B0/B1/B2 và cải thiện nếu có, input/output, toán học loss/pooling; đủ chi tiết để triển khai.
5. **Experimental setup:** protocol, model selection, seeds, compute/search space, metric/grouping/CI, final policy.
6. **Results:** bảng chính trả lời RQ1, diagnostics trả lời RQ2/RQ3, ablation và failure cases; cả kết quả âm.
7. **Discussion/limitations:** lời giải thích có thể, confounds, giới hạn scope, uncertainty và nguy cơ overclaim.
8. **Conclusion:** trả lời câu hỏi theo evidence; future work là phần chưa làm.
9. **Appendix/artifacts:** config, run details, split procedure, additional groups, replication commands; không giấu controls quyết định claim chỉ trong appendix.

Bảng/hình nên có: thống kê A/B; B0/B1/B2 với n/seed/E/Mod/Head/cost; Delta+CI; E3 đúng/hoán đổi context cùng subset; predicted vs gold/residual; error taxonomy có quy tắc chọn mẫu. Không chỉ có scatter đẹp hoặc demo screenshot. Figure phải nêu dữ liệu/setting/n và cách tính.

**Ví dụ văn phong, số liệu chưa có:** “Trong split B của EN-NN và backbone đã thử, B2 giảm MAE theo cụm so với B1; mức giảm và khoảng tin cậy được báo ở Bảng X.” Nếu E3/residual không hỗ trợ, viết rõ điều đó. Không viết “CoSense hiểu context” hay “generalizes to unseen expressions” mà bỏ điều kiện unseen khi fine-tune.

Đóng góp có thể là đánh giá/diagnostic được thiết kế tốt và phát hiện có độ bền, hoặc phương pháp cải thiện có ablation. Phải kiểm tra novelty trong literature và chọn venue phù hợp. System description của SemEval và research paper độc lập có mục tiêu khác nhau; đều phải có evidence nhưng không mặc định system paper cần kiến trúc mới. Deadline/template/tiêu chí venue cụ thể phải xác minh sau, không hứa publish.

## 11. Những điều reviewer cần kiểm tra được

Dùng các câu hỏi sau để tự rà soát bản thảo, không coi đây là bảo đảm acceptance:

- Người đọc xác định được RQ, comparison chính và phạm vi claim?
- Baseline đủ mạnh cho claim, cùng budget/model selection và cùng mẫu đánh giá?
- Split không leak cụm/câu theo mục tiêu nghiên cứu; final có được giữ độc lập?
- Effect size và độ bất định được báo; sample grouping, seeds và weighting rõ?
- Diagnostics có phân biệt lợi ích tổng thể với context-specific behavior, có bàn alternative explanations?
- Mọi số trong bảng có thể truy về prediction/config/run, gồm trial không thuận lợi?
- License/citation/version của dataset/backbone/code, nguồn trial gộp và giới hạn annotation được mô tả?
- Reproduction commands/artifacts và lý do không thể chia sẻ một số tài sản được ghi? Không tự công khai dữ liệu chưa có quyền.

ARR hướng dẫn tác giả báo experimental setup, hyperparameter selection, tài nguyên, độ biến động kết quả và limitations; xem [checklist chính thức](https://aclrollingreview.org/responsibleNLPresearch/). Áp dụng phần phù hợp với project: điểm dự đoán không phải chân lý ngôn ngữ, phạm vi English-only và prototype phải rõ. Nếu tạo bộ contrastive có người đánh giá về sau, thiết kế annotation, consent/compensation và quy trình của trường trước khi thu thập; hiện chưa cần tuyển người hoặc thu dữ liệu mới.

## 12. Roadmap thực hành và điều kiện chuyển giai đoạn

| Giai đoạn | Bạn cần học/làm | Artifact dự kiến | Được chuyển khi |
| --- | --- | --- | --- |
| R0 — xác định câu hỏi, song song M1 | RQ, hypothesis, type/token, đọc nguồn trực tiếp | research question + literature notes ban đầu | claim dự kiến có đối chứng và khả năng bác bỏ |
| R1 — dữ liệu, M1 | group split, leakage, provenance, alignment | audit + A/B manifests + protocol draft | không leak đã biết, coverage và ngoại lệ rõ |
| R2 — baseline, M2/M3 | regression, validation selection, Kaggle artifact | B0 và pilot B1/B2, registry | pipeline/smoke pass, có cost estimate, final chưa mở |
| R3 — chẩn đoán, M4 | effect, confounds, residual, ablation | validation error report + một hypothesis cải thiện hoặc quyết định giữ baseline | lỗi và evidence dẫn tới lựa chọn, không thêm method tùy hứng |
| R4 — xác nhận, M6 | nhiều seed, paired groups, CI, effect size | locked protocol + E1–E4; E5/E6 khi cần | controls đủ cho claim, n/cost/uncertainty đầy đủ |
| R5 — final và viết, M6 | claim–evidence, limitations, reproducibility | final tables + report/manuscript + reproduction guide | kết luận truy về artifact, không tune trên final |

M5 demo có thể đi sau checkpoint M3, không phải bằng chứng khoa học thay thế R4/R5. R0/R1 có thể làm ở local; mọi bước train của R2–R4 chạy Kaggle GPU.

Lịch ước lượng bổ sung: trong M1 dành 1–2 buổi chốt RQ/protocol; M3 có pilot và quyết định budget; M4 dành khoảng một tuần đọc toàn văn/phân tích; M6 dành 2–3 tuần xác nhận rồi 1–2 tuần viết/kiểm tra tái lập. Giả định 6–10 giờ/tuần và GPU khả dụng, thời gian này điều chỉnh theo kết quả, không hứa thời hạn paper. Nếu cần dataset/annotation mới, lập kế hoạch riêng thay vì thêm vào lịch như miễn phí.

**Việc tiếp theo:** chuẩn bị notebook Kaggle B1/B2 dùng membership/config đã khóa và RESEARCH_PROTOCOL v2. Chạy kiểm tra GPU/dependency/tokenization/target retention và smoke trước full training khi được yêu cầu. B0 validation đã có; final chưa mở. Nếu chuyển sang span pooling, rà soát heuristic alignment trước. Ngân sách GPU thực tế cần đo pilot, chưa xác định novelty hoặc kết quả neural model.
