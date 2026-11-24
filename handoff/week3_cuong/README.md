# Cương — scorer và thí nghiệm bias

[Report](../../reports/week3/Cuong.md): B2 bias giảm RMSE nhưng không đạt điều kiện giữ vì Spearman head giảm quá 0,02. Đã thêm MAE vào scorer và đồng bộ các notebook baseline.

| File | Mục đích |
|---|---|
| `diagnose.py` | Chẩn đoán baseline trước thí nghiệm; cũng được dùng khi phân tích |
| `prepare_experiment.py` | Tạo hai notebook bias từ baseline khi cần tái lập |
| `analyze_bias.py` | Kiểm tra artifacts và tính lại bảng so sánh |
| `SAVED_VERSIONS.json` | URL hai phiên bản Kaggle, dùng trong hồ sơ kiểm chứng |

Chạy các lệnh dưới đây từ repo root, sau khi cài `requirement.txt`. Thư mục output phải chưa tồn tại. ZIP và kết quả phân tích nằm ngoài Git.

Kiểm tra local:

```bash
python -m pytest -q
```

Phân tích lại artifacts (sửa đường dẫn theo nơi lưu ZIP):

```bash
python handoff/week3_cuong/analyze_bias.py --baseline-dir /home/xcuon/dump/output/Bx --bias-dir /home/xcuon/dump/output/Bx_bias --output results/nn/cuong_bias_comparison_new
```

Kết quả đã chạy nằm ở `results/nn/cuong_bias_comparison/`: metrics, chênh lệch bias−baseline, diagnostics và verification. Script kiểm hash/CRC toàn bộ ZIP; không nạp weights.

Tạo lại notebook bias nếu cần:

```bash
python handoff/week3_cuong/prepare_experiment.py --decision 'B2 affine dev RMSE reductions: ModAvg=0.182733, HeadAvg=0.041887; both exceed 0.02. Choose train-mean bias; diagnostic only.' --output results/nn/cuong_bias_notebooks
```

Import hai notebook lên Kaggle, thêm dataset `cosense-kaggle-data`, bật GPU và Internet. Chạy `SMOKE=True` trước rồi `False` để chạy full; giữ split42, `RESUME=None`, không đánh giá test. Outputs nằm dưới `/kaggle/working/cuong_bias/`. Đây là bước tái lập tùy nhu cầu; hai run đã có kết quả.
