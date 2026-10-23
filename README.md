# CoSense — nghiên cứu và dự thi SemEval-2027 Task 5

Model dự đoán một cụm từ còn giữ nghĩa của các thành phần đến mức nào trong một câu. B1 chỉ đọc cụm từ; B2 đọc thêm context. Phạm vi hiện tại: English NN và PV, train riêng trên Kaggle.

**Bắt đầu ở [lộ trình dự án](docs/README.md).** Trang này cho biết đã làm gì, cần chạy gì tiếp và khi nào đủ để viết paper/nộp bài.

| Khi bạn cần… | Mở |
|---|---|
| Hiểu tổng thể và chọn việc tiếp theo | [Lộ trình](docs/README.md) |
| Hiểu điểm gold, công thức và lý do chọn metric | [Metrics](docs/METRICS.md) |
| Train Kaggle, mở web, giữ checkpoint, chuẩn bị submission | [Cách chạy và nộp bài](docs/RUN_GUIDE.md) |

## Mở web local

PowerShell tại thư mục repo. Cài lần đầu:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
```

Mỗi lần dùng:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Mở http://localhost:8501. Web hiện xem dữ liệu và results ZIP; chưa dự đoán câu mới.

## Repo gồm gì?

```text
notebooks/          Sửa B1/B2 trực tiếp, chạy trên Kaggle
app/ + src/         Web và logic đọc/so sánh kết quả
data/               Dữ liệu, split cố định, data ZIP — riêng tư
results/            Results tải từ Kaggle, baseline, phân tích — riêng tư
docs/               Ba trang hướng dẫn; audit local nằm trong audit/
literature-review/  Research notes và tài liệu paper
tests/              Kiểm tra logic local và notebook syntax
requirement.txt     Dependencies local; pyproject đọc cùng danh sách
```

Không có bước build notebook hoặc train local. Đổi logic chung thì cập nhật cả B1 và B2. Kiểm tra local bằng `.\.venv\Scripts\python.exe -m pytest -q`.

Dữ liệu, results, audit và archive được Git bỏ qua. Các bản trước khi dọn nằm trong `archive/`; docs trước lần viết lại này được lưu ở `archive/docs-before-rewrite-20261023.zip`.
