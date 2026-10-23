# Đọc điểm và chọn metric

**Đọc nhanh:** Spearman xem đúng thứ tự không; MAE/RMSE xem lệch mấy điểm; delta MAE xem model có hiểu sự thay đổi giữa các context không. `variance_correlation` là chẩn đoán bổ sung. Không có một con số thay thế được tất cả.

[Lộ trình](README.md) · [Cách chạy](RUN_GUIDE.md)

## 1. Điểm model phải dự đoán

| Tên | Ý nghĩa |
|---|---|
| `ModAvg` | Trung bình điểm người chấm cho modifier của noun compound |
| `HeadAvg` | Trung bình điểm cho head của noun compound |
| `Avg` | Trung bình điểm cho cả particle verb, dùng ở PV |
| `ModStd`, `HeadStd`, `Std` | Độ phân tán đánh giá của người chấm; không phải đầu ra model hiện tại |

Trong `flea market`, modifier là `flea`, head là `market`. Ví dụ của ban tổ chức có điểm 0,0 / 4,7: ít giữ nghĩa flea, vẫn giữ nghĩa market. Thang 0–5, cao là compositional hơn. Đây là điểm về nghĩa, không phải xác suất/độ tự tin. [Nguồn ban tổ chức](https://list.elra.info/mailman3/hyperkitty/list/corpora%40list.elra.info/thread/7D6T2BHZF3ZT7CJFQKSYE2B5TBXF27XZ/)

Các công thức dưới tính **riêng từng output**. Gọi gold là $y_i$, prediction là $\hat y_i$, số mẫu hợp lệ là $n$.

## 2. Bảng chọn metric — vì sao dùng cả năm?

| Metric | Muốn biết điều gì? | Miền / chiều tốt | Vai trò trong nghiên cứu này |
|---|---|---|---|
| Spearman | Model xếp cao/thấp giống người không? | −1 đến 1; cao tốt | Metric chính hiện chọn checkpoint, phù hợp mục tiêu thứ hạng |
| MAE | Trung bình lệch mấy điểm? | 0 đến ∞; thấp tốt | Dễ diễn giải, bổ sung phần sai số mà thứ hạng bỏ qua |
| RMSE | Có những lần đoán lệch lớn không? | 0 đến ∞; thấp tốt | Phạt lỗi lớn mạnh hơn MAE |
| delta MAE | Mức đổi điểm giữa context của cùng expression có đúng không? | 0 đến ∞; thấp tốt | Kiểm tra context benefit ở bên trong expression |
| variance correlation | Expression nào biến động theo context nhiều/ít, model có nhận ra không? | −1 đến 1; cao tốt, có thể NA | Diagnostic; không dùng một mình để chọn model |

Spearman là lựa chọn **nội bộ** của nghiên cứu, dựa trên mục tiêu ordinal/ranking được [trang Task 5](https://sites.google.com/view/semeval-compartment) mô tả. Chưa xác nhận nó trùng official scorer. Khi có scorer chính thức, báo riêng official score và các diagnostics này. Không cộng năm metric thành một điểm tùy ý.

## 3. MAE và RMSE — sát gold đến đâu?

$$
\mathrm{MAE}=\frac{1}{n}\sum_i |y_i-\hat y_i|
$$

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}
$$

| Mẫu minh họa | Gold | Prediction | Sai số tuyệt đối |
|---|---:|---:|---:|
| A | 1 | 2 | 1 |
| B | 3 | 3 | 0 |
| C | 5 | 3 | 2 |

MAE = (1+0+2)/3 = **1 điểm**. RMSE = √((1+0+4)/3) ≈ **1,29 điểm**. Bình phương làm RMSE nhạy hơn với các lỗi lớn. Cùng tập/mức trọng số, RMSE ≥ MAE. [Công thức MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html), [RMSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html).

**RMSE = 0,9 chưa đủ để gọi tốt hay tệ.** Nó có đơn vị điểm nhãn; không có nghĩa chính xác 82% hoặc mọi mẫu lệch 0,9. So với B0 trên cùng output/split mới có ý nghĩa. Modifier dev hiện có B0 = 0,938; B1 = 0,899; B2 = 0,974. B1 cải thiện nhẹ so với đoán trung bình, B2 chưa cải thiện RMSE này.

Nếu cả gold/prediction nằm trong 0–5 thì MAE/RMSE tối đa 5. Model hiện dùng đầu ra tuyến tính không clipping, nên miền tổng quát không có trần. Không thêm clipping sau khi nhìn test chỉ để tăng điểm.

## 4. Spearman — đúng thứ tự đến đâu?

Đổi gold/prediction thành thứ hạng $r_i,s_i$, rồi tính tương quan Pearson của hai dãy hạng:

$$
\rho=\frac{\sum_i(r_i-\bar r)(s_i-\bar s)}
{\sqrt{\sum_i(r_i-\bar r)^2\sum_i(s_i-\bar s)^2}}
$$

Điểm trùng nhau dùng hạng trung bình. $1$ là cùng thứ tự hoàn toàn, $-1$ đảo ngược hoàn toàn, gần $0$ là ít liên hệ thứ hạng. Một dãy không đổi thì correlation không xác định → NA. [SciPy spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

Ví dụ gold `[1, 2, 3]`, prediction `[3, 4, 5]`: Spearman = **1**, nhưng MAE = RMSE = **2**. Vì vậy cần cả ranking lẫn error. Spearman 0,5 không phải “đúng 50%”, và không có ngưỡng 0,5 là đạt cho mọi dataset.

Trong notebook: checkpoint chọn theo trung bình Spearman của các output hợp lệ trên dev; nếu không có correlation hợp lệ thì fallback dev RMSE. Với NN vẫn phải báo modifier/head riêng, không chỉ trung bình.

## 5. delta MAE — đổi điểm theo context có đúng không?

Gold của cùng expression ở hai context là `[1, 4]`: tăng 3. Model dự đoán `[2, 3.5]`: tăng 1,5. Sai số mức thay đổi là **|3−1,5| = 1,5 điểm**.

Với expression $t$ có $n_t$ context hợp lệ, code lấy mọi cặp:

$$
D_t=\frac{1}{\binom{n_t}{2}}\sum_{i<j}
\left|(y_{ti}-y_{tj})-(\hat y_{ti}-\hat y_{tj})\right|
$$

$$
\mathrm{delta\ MAE}=\frac{1}{T}\sum_t D_t
$$

Mỗi expression có trọng số bằng nhau ở bước cuối (macro); $T$ chỉ đếm expression có ít nhất hai context hợp lệ. Tốt nhất là 0. Nếu mọi điểm nằm trong 0–5 thì metric này tối đa 10; tổng quát không có trần.

Vì sao cần? Spearman toàn tập có thể tốt chỉ vì model phân biệt được expression thường cao/thấp; delta MAE kiểm tra nó có theo đúng khác biệt **giữa context của cùng expression** không. B1 đoán không đổi nên prediction delta luôn bằng 0. B2 cần thắng mốc này mới có bằng chứng giảm lỗi biến thiên context.

Gold `[1,4]`, prediction `[2,5]` thì delta MAE = 0 dù MAE = 1: mức thay đổi đúng nhưng điểm bị lệch đều. Vì vậy delta MAE cũng không thay MAE/RMSE.

## 6. variance correlation — độ biến động có hợp lý không?

Với mỗi expression, tính độ lệch chuẩn qua các context, dùng mẫu số $n_t$ (`ddof=0`):

$$
\sigma_t^{\mathrm{gold}}=\sqrt{\frac{1}{n_t}\sum_i(y_{ti}-\bar y_t)^2}
$$

$$
\mathrm{variance\ correlation}=\mathrm{Spearman}
\left(\{\sigma_t^{\mathrm{gold}}\},\{\sigma_t^{\mathrm{pred}}\}\right)
$$

Code dùng **SD qua context** rồi Spearman giữa các expression, dù tên cột là `variance_correlation`. Không dùng `ModStd`/`HeadStd` của người chấm.

Nếu gold của expression A biến động nhiều hơn B, model có nhận ra thứ tự biến động đó không? Metric này không kiểm tra context cụ thể nào cao/thấp, cũng không bảo đảm độ lớn SD dự đoán chính xác. Vì vậy chỉ là diagnostic.

B1 cho cùng expression cùng prediction nên tất cả predicted SD = 0 → correlation NA, không phải code hỏng. B2 hiện ≈0,225 modifier / 0,181 head: liên hệ dương nhưng chưa đủ kết luận hiểu context tốt. Cần đọc cùng delta MAE và error.

## 7. Áp dụng vào pilot của bạn

| Output/model | Spearman ↑ | MAE ↓ | RMSE ↓ | Var. corr. ↑ | Delta MAE ↓ |
|---|---:|---:|---:|---:|---:|
| Modifier B1 | 0,446 | 0,676 | 0,899 | NA | 0,585 |
| Modifier B2 | 0,503 | 0,721 | 0,974 | 0,225 | 0,602 |
| Head B1 | 0,549 | 0,582 | 0,794 | NA | 0,546 |
| Head B2 | 0,552 | 0,602 | 0,808 | 0,181 | 0,601 |

Kết luận được phép: **một seed trên dev, B2 xếp thứ tự tốt hơn nhưng absolute error và delta error chưa tốt hơn B1**. Chưa được nói B2 luôn tốt hơn hoặc context vô ích. Xem tiếp lỗi dev và nhiều seeds theo [lộ trình](README.md).

## 8. Những chi tiết cần ghi trong paper

- Code tham chiếu: [metrics.py](../src/compartment/metrics.py). Trong JSON, `context_variance.value` tương ứng cột `variance_correlation`; `delta_mae_macro` tương ứng `delta_mae`.
- MAE hiện tính từ predictions/absolute errors trong phân tích và error groups, không có field overall MAE riêng trong `evaluate()` JSON. Không nhầm MAE theo từng nhóm với MAE toàn tập.
- Tính theo từng output, bỏ cặp gold/pred không hữu hạn và báo số mẫu hợp lệ; không âm thầm thay NA bằng 0. SD/delta bỏ expression chỉ có một context hợp lệ.
- Overall MAE/RMSE/Spearman tính trên rows; delta MAE macro tính đều theo expression. Khi báo macro target MAE hay CI phải ghi rõ cách gộp, không coi chúng là cùng một chỉ số.
- Mean ± SD qua training seeds đo độ dao động giữa lần train. Target-cluster bootstrap CI mô tả bất định khi lấy mẫu expression. Hai loại này không thay nhau.
- Các metric context ở đây là thiết kế phân tích nội bộ; official scorer parity chưa xác nhận ngày 23/10/2026.
