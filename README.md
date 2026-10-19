# SemEval 2027 Task 5

CoSense: audit **EN-NN/EN-PV tiếng Anh**, split và B0 cho **EN-NN**. B1/B2 chưa huấn luyện; các neural training run sẽ chạy Kaggle GPU.

- [Báo cáo dữ liệu và kết quả đã chạy](docs/DATA_REPORT.md)
- [Protocol split/model/evaluation](docs/RESEARCH_PROTOCOL.md)
- [Roadmap project](docs/PROJECT_ROADMAP.md)
- [Roadmap nghiên cứu](docs/RESEARCH_ROADMAP.md)
- [Hướng dẫn Kaggle chi tiết: upload Dataset, prepare, smoke và train](docs/KAGGLE_EN_NN_GUIDE.md)
- [Notebook Kaggle EN-NN tự chứa code](notebooks/kaggle_en_nn.ipynb)

Đã có notebook/code chuẩn bị EN-NN và tùy chọn fine-tune B1/B2; mặc định chỉ `prepare`, chưa chạy Kaggle GPU.
Tạo lại notebook và gói Dataset local (không upload, không train):

```powershell
python scripts/build_kaggle_notebook.py
python scripts/package_kaggle_en_nn.py
```

Gói được tạo tại `dist/kaggle/cosense-en-nn-data.zip`, bị Git ignore. Kiểm tra quyền sử dụng dữ liệu trước khi upload lên Kaggle.

Từ repository root, dùng Python ≥3.11. Core scripts dùng thư viện chuẩn; Matplotlib chỉ cần để vẽ PNG:

```powershell
python -m pip install -r requirements-audit.txt
python scripts/audit_data.py
python scripts/create_splits.py
python scripts/validate_splits.py
python scripts/run_mean_baseline.py
python scripts/build_data_report.py
python -m unittest discover -s tests -v
```

Nếu dùng dependency cài riêng trong `.runtime/`, dùng cùng phiên bản Python (hiện Python 3.11). Core audit có thể chạy với `--no-figures`; báo cáo có hình cần chạy audit đầy đủ. Ưu tiên môi trường riêng thay vì sửa Python base.

Nguồn đọc tường minh ở `configs/data_protocol.json`; trial đã gộp theo người dùng. Không nạp trial thêm hoặc sửa TSV. Split lưu source/config/file hash; B0 chỉ xuất validation, chưa có metric final. Đổi source/config phải audit và kiểm tra split lại, không tự đổi split giữa fine-tune runs.

`data/`, `reports/`, `results/`, checkpoint và runtime bị ignore vì quyền phân phối chưa xác minh. Chưa commit/upload dữ liệu nguồn hoặc dẫn xuất. Metric và template chính thức chưa xác minh; protocol phân biệt metric nội bộ với leaderboard.
