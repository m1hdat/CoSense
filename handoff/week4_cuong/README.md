# Cương tuần 4 — đối chứng cách chọn checkpoint

[Report hoàn chỉnh](../../reports/week4/Cuong.md): đã kiểm chứng đủ NN43/44 và PV43. Hai seed NN không đạt điều kiện giữ khi chọn mean RMSE; PV43 cùng chọn epoch1, predictions giống nhau. Giữ selector Spearman làm mốc. [Registry Kaggle](SAVED_VERSIONS.json) đã có đủ ba saved version.

## Kết quả và mã tái lập

| Nội dung | Vị trí |
|---|---|
| Audit bốn run tuần 3 seed42 | `results/nn/week4_cuong_audit/` |
| NN43/44: metrics, chênh lệch, summary, verification | `results/nn/week4_cuong_seed43/`, `results/nn/week4_cuong_seed44/` |
| PV43 | `results/pv/week4_cuong_seed43/` |
| Bảng gộp và kiểm tra giữa hai seed NN | `results/week4_cuong/combined_metrics.csv`, `verification_summary.json` cùng thư mục |
| Audit bằng Python chuẩn | `audit_rmse.py` |
| Kiểm artifacts và tính lại metrics | `compare_selectors.py` |
| Tạo notebook đối chứng từ baseline | `prepare_checkpoint_audit.py` |

Các đường dẫn results tính từ repo root và nằm ngoài Git. ZIP nguồn giữ nguyên tại `/home/xcuon/dump/output/week4/`. Scripts không nạp weights; link Kaggle do người dùng cung cấp, chưa xác minh trực tiếp với output trên web.

## Chạy lại kiểm chứng

Từ repo root, cài dependencies trong `requirement.txt`; dùng thư mục output mới:

```bash
python handoff/week4_cuong/compare_selectors.py --directory /home/xcuon/dump/output/week4 --task nn --seed 43 --output results/nn/week4_cuong_seed43_recheck
```

Đổi thành `--task nn --seed 44` hoặc `--task pv --seed 43`, đồng thời đổi output tương ứng. Input cần đủ complete/results/RMSE_results ZIP và ARTIFACTS/SELECTION_ARTIFACTS JSON. Script kiểm hash/CRC, dữ liệu ghép cặp, metadata, epoch và metrics; tự đưa saved version vào hồ sơ.

Audit lịch sử seed42 cũ:

```bash
python handoff/week4_cuong/audit_rmse.py --baseline-dir /home/xcuon/dump/output/Bx --bias-dir /home/xcuon/dump/output/Bx_bias --output results/nn/week4_cuong_audit_recheck
```

Centered RMSE dùng gold dev chỉ để chẩn đoán, không phải kết quả model mới.

## Tạo lại notebook nếu cần tái lập

```bash
python handoff/week4_cuong/prepare_checkpoint_audit.py --task nn --seed 43 --output results/week4_cuong_reproduce/NN_B2_seed43_checkpoint_audit.ipynb
```

Đổi task/seed cho NN44 hoặc PV43. Notebook không đổi loss, lr, khởi tạo baseline, split hoặc alignment; một trajectory lưu hai selector, **early stopping vẫn theo Spearman**. Không hỗ trợ resume hoặc test evaluation. Không cần chạy lại chỉ để hoàn thiện report.

Khi tái lập trên Kaggle: thêm `cosense-kaggle-data`, bật GPU/Internet; chạy smoke rồi full. Đợi **`SELECTION_OUTPUTS_READY` ở cell cuối**, tải hai results ZIP, complete ZIP và hai inventory JSON từ đúng saved version. Giữ `RESUME=None`, `EVALUATE_LOCAL_TEST=False`, lưu artifacts riêng để không ghi đè run cũ.

Kiểm tra local: `python -m pytest -q`.
