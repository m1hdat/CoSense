# Lộ trình: từ hai notebook đến paper và bài nộp Task 5

**Bạn đã hoàn thành pilot NN cho B1/B2. Việc tiếp theo là chạy cặp PV, đọc lỗi, rồi mới chốt thí nghiệm cuối.** Không cần train lại NN chỉ để output đẹp hơn.

## 1. Chúng ta đang xây gì?

Ví dụ `flea market` là một loại market, nhưng không phải chợ bán flea. Bài toán dự đoán mức giữ nghĩa của từng thành phần trong context, trên thang 0–5: thấp là ít giữ nghĩa, cao là giữ nghĩa rõ. Với NN có hai đầu ra `ModAvg`, `HeadAvg`; PV có một đầu ra `Avg`. [Thông báo của ban tổ chức](https://list.elra.info/mailman3/hyperkitty/list/corpora%40list.elra.info/thread/7D6T2BHZF3ZT7CJFQKSYE2B5TBXF27XZ/)

| Model | Được đọc gì? | Vai trò |
|---|---|---|
| B0 | Không đọc input; luôn đoán trung bình nhãn train | Kiểm tra model học có hơn cách đoán đơn giản không |
| B1 | Expression, ví dụ `flea market` | Mốc so sánh khi không có context |
| B2 | Câu chứa expression, pool các thành phần trong câu | Kiểm tra lợi ích của context |

B1/B2 fine-tune BERT có sẵn, không train BERT từ đầu. NN và PV là các model riêng. Hướng nghiên cứu là **đo context giúp ở đâu, khi nào và ổn định đến mức nào**, chưa tuyên bố đây là kiến trúc mới.

```text
Data + frozen splits → Kaggle train/evaluate → results + checkpoints
                                              ↓
                             Web xem lỗi → phân tích → paper
                                              ↓
                          Checkpoint → official predictions → CodaBench
```

Web hỗ trợ hiểu và trình bày kết quả. Muốn thi cần pipeline tạo predictions cho dữ liệu official; web đẹp không thay bước này.

## 2. Hiện đang ở đâu? — kiểm tra ngày 23/10/2026

| Hạng mục | Trạng thái / bằng chứng |
|---|---|
| Dữ liệu English NN/PV và frozen splits | Đã có trong `data/`; không chia lại |
| Baseline B0 NN/PV | Đã có dev results trong `results/baselines/` |
| NN B1/B2 seed42 | Đã chạy GPU, dev results trong `results/nn_B1_B2_seed42/` |
| PV B1/B2 | Chưa có kết quả trong workspace |
| NN/PV nhiều training seeds, final internal test | Chưa hoàn thành |
| Web xem data/results ZIP | Đã có; chưa inference câu mới |
| Checkpoint NN từ run cũ | Hai results ZIP không chứa weights; chưa thấy checkpoint local |
| Pipeline inference + official submission | Chưa hoàn thành/xác nhận theo scorer của ban tổ chức |

NN pilot có 504 dev rows, 67 expressions:

| Đầu ra | B1 Spearman ↑ | B2 Spearman ↑ | B1 RMSE ↓ | B2 RMSE ↓ |
|---|---:|---:|---:|---:|
| Modifier | 0,4464 | 0,5025 | 0,8993 | 0,9743 |
| Head | 0,5488 | 0,5520 | 0,7943 | 0,8079 |

B2 xếp thứ tự tốt hơn, rõ hơn ở modifier, nhưng sai số điểm chưa tốt hơn. Một seed trên dev chưa đủ kết luận context luôn có ích. Xem [cách đọc metrics](METRICS.md), hoặc [phân tích pilot đã có](../results/nn_B1_B2_seed42/README.md) trên máy local; results không nằm trong Git.

## 3. Ba câu hỏi dẫn đường cho paper

| Câu hỏi | Thí nghiệm/bằng chứng cần có |
|---|---|
| RQ1: Có context thì dự đoán tốt hơn không? | B0/B1/B2 cùng dữ liệu, backbone, head capacity và cấu hình; báo từng output |
| RQ2: Trên expression chưa gặp khi train, model làm được gì? | Dùng split target-disjoint hiện tại; báo kết quả và lỗi trên unseen targets |
| RQ3: Context giúp hoặc gây lỗi ở trường hợp nào? | So lỗi theo alignment/fallback, độ dài, subword count, annotation uncertainty; đọc ví dụ thực tế |

Với split hiện tại, mọi expression dev/test đều chưa gặp trong task training; đó không có nghĩa BERT chưa gặp chúng khi pretrain. Thiết kế này **không tự chứng minh unseen khó hơn seen**: muốn so sánh seen/unseen phải thiết kế thêm đối chứng trước khi xem kết quả. RQ3 chỉ là liên hệ mô tả, không tự chứng minh quan hệ nhân quả.

## 4. Làm theo thứ tự này

### Mốc A — có đủ pilot NN và PV

- [x] NN B1/B2 seed42 có results.
- [ ] Kiểm tra saved version Kaggle cũ còn `best.pt`/tokenizer/config không; tải về nếu còn.
- [ ] Chạy PV B1 và B2: `TASK='pv'`, `SEED=42`, giữ `SPLIT_SEED=42`, `SMOKE=False`, `EVALUATE_LOCAL_TEST=False`.
- [ ] Lưu results và model artifacts, mở lại được results ZIP; ghi lại data/config/version.

**Xong mốc:** có bốn pilot runs hợp lệ: NN/PV × B1/B2; đọc được kết quả từng output. [Các bước chạy](RUN_GUIDE.md#2-train-trên-kaggle).

### Mốc B — hiểu kết quả và chốt recipe

Với mỗi output, đối chiếu B0/B1/B2 bằng Spearman, MAE/RMSE; dùng delta MAE để kiểm tra mức thay đổi theo context. Chọn khoảng 10–20 ví dụ dev để đọc: B2 giúp, B2 làm tệ hơn, và fallback/ambiguous. Ghi gold, prediction hai model, spans và lý do nghi ngờ; không tự sửa gold theo dự đoán.

Mỗi lần thay thiết kế, ghi trước: **vấn đề quan sát → giả thuyết → đúng một thay đổi → metric/dev slice sẽ kiểm tra → điều kiện giữ/bỏ**. Ví dụ kiểm tra heuristic alignment ở PV trước khi đổi encoder. Không tăng epochs chỉ vì chạy nhanh. Nếu thử calibration để giảm sai số điểm, fit bằng train/dev, ghi thành biến thể riêng và không nhìn test để quyết định.

Đọc thêm paper khi cần giải quyết một quyết định cụ thể. Mỗi note chỉ cần: phương pháp; bằng chứng ở section/table nào; áp dụng phần nào; khác thiết lập của mình ở đâu; thí nghiệm nhỏ tiếp theo. Dùng [RN002](../literature-review/RN002.md), [RN006](../literature-review/RN006.md) làm tài liệu tham khảo, không coi mô tả phạm vi cũ trong note là protocol hiện tại.

**Xong mốc:** chốt backbone/config, preprocessing, checkpoint policy và danh sách biến thể trước final test. Kết quả âm hoặc chỉ cải thiện một mặt vẫn có giá trị nếu đối chứng rõ.

### Mốc C — thí nghiệm cuối cho paper

- [ ] Với recipe đã chốt, chạy NN/PV × B1/B2 × seeds 42/43/44: tổng 12 runs; tái sử dụng pilot nếu recipe thực sự không đổi.
- [ ] Giữ nguyên frozen split. Báo mean ± sample SD qua training seeds, NN modifier/head và PV riêng.
- [ ] Dùng dev chọn checkpoint; sau khi khóa thiết kế mới đánh giá internal test cho từng seed. Không chọn seed thắng nhờ test.
- [ ] Báo chênh lệch ghép cặp B2−B1 cho Spearman; B1−B2 cho error để gain dương luôn là tốt hơn.
- [ ] Nếu báo CI, bootstrap theo expression vì các context cùng expression phụ thuộc nhau; ghi rõ metric, split, số lần lấy mẫu. CI của MAE gain không phải CI của Spearman gain; không coi ba seeds là ba dataset độc lập.

Team đã xem dữ liệu nguồn khi audit; không tuyên bố internal test hoàn toàn chưa từng được nhìn thấy. Internal test vẫn phải được giữ ngoài việc chọn recipe. Official test là tập riêng của cuộc thi.

**Xong mốc:** một bảng kết quả có B0/B1/B2, seed variability, coverage/fallback và giải thích các trường hợp thắng/thua; chưa gọi đây là leaderboard score.

### Mốc D — hoàn thành sản phẩm và dự thi

1. Hoàn thành inference: nạp checkpoint/tokenizer, nhận dữ liệu mới không có gold và tạo đúng số điểm/ID/order. Đây là việc còn phải triển khai.
2. Tách kết quả internal test khỏi model dùng dự thi. Nếu refit trên nhiều dữ liệu hơn sau khi khóa recipe, ghi rõ dùng tập nào và không dùng official gold để tune.
3. Kiểm tra official rules/scorer, format, tracks được tham gia; chạy thử trên dữ liệu mẫu chính thức. Chưa xác nhận English-only có đủ cho cách nộp hiện hành hay không.
4. Giữ bản submission và receipt/score của đúng version trên CodaBench. Không tự coi ZIP results của web là ZIP submission.
5. Web: tối thiểu xem/so sánh kết quả; inference câu mới là phần demo sau khi có checkpoint và pipeline đúng.

[Trang Task 5](https://sites.google.com/view/semeval-compartment) hiện ghi evaluation 10–31/01/2027, kiểm tra ngày 23/10/2026. Trang dẫn sang [CodaBench 17971](https://www.codabench.org/competitions/17971/) cho format/rules. CodaBench không tải được trong lần kiểm tra này: official metric, giới hạn nộp, format và lịch paper vẫn cần xác nhận; không lấy quy định của task/năm khác thay thế.

## 5. Viết paper từ bằng chứng nào?

| Phần paper | Nội dung cần viết |
|---|---|
| Introduction | Câu hỏi về lợi ích context; phạm vi English NN/PV |
| Related work | Paper liên quan dẫn đến lựa chọn pooling/diagnostics nào; không nhận reproduction nếu khác setup |
| Method + setup | Input/output B0/B1/B2, splits, seed/config, fallback, loss, cách chọn checkpoint |
| Results | Bảng nhiều seed/test; phân biệt metric chính của nghiên cứu với metric official đã xác nhận |
| Analysis | RQ1–RQ3, within-expression errors và ví dụ; không chỉ chọn ví dụ có lợi |
| Limitations | English-only, heuristic alignment, annotation disagreement, ít data/seeds và hạn chế đối chứng |

Đủ để hoàn thành nghiên cứu khi câu hỏi có thí nghiệm trả lời, kết quả tái lập được và kết luận đúng mức bằng chứng. Đủ để dự thi khi official submission được chấp nhận; hai tiêu chí này cần được kiểm tra riêng.
