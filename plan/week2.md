# Tuần 2 (26/10–01/11): Dữ liệu, scorer, split, alignment

**Hạn việc: 20:00 Chủ nhật 01/11 · Họp: 21:00 Thứ Hai 02/11**

Cả tuần là thời gian làm, không chia việc theo ngày. Mỗi người tự xếp giờ, chỉ cần làm đúng thứ tự ở mục 5 và kịp bốn hạn ở đó.

## 1. Mục tiêu tuần 2

Cuối tuần này nhóm có bốn công cụ dùng chung **đã được đo và kiểm tra**, và hai quyết định thiết kế đã được chốt trong họp:

| # | Công cụ | Người làm | Đầu ra cuối |
|---|---|---|---|
| 1 | Bộ đọc dữ liệu + hiểu dữ liệu | Khang | `results/analysis/eda.md` |
| 2 | Scorer nội bộ | Cương | MAE tổng trong `evaluate`, test ví dụ tính tay, test chống lệch code (PR) |
| 3 | Split cố định | Đạt | `results/analysis/split_check.md` + đề xuất cho quyết định D1, D2 |
| 4 | Định vị cụm từ (alignment) | Phát | `results/analysis/alignment_report.md`, `alignment_failures.tsv`, test (PR) |

Cả bốn công cụ **đã có code** trong `src/compartment/`. Tuần này chủ yếu là đo đạc, bổ sung test và viết báo cáo, không viết hệ thống mới. Riêng `align` **chưa được sửa trong tuần 2** (xem mục 3.1).

Tuần 3 (Mốc A: chạy PV B1/B2) chỉ cần data ZIP và hai notebook, không phụ thuộc kết quả tuần 2. Trễ tuần 2 không chặn tuần 3.

## 2. Hiện trạng ngày 24/10

| Công cụ | Đã có | Còn thiếu |
|---|---|---|
| Bộ đọc dữ liệu | `load_records` (kiểm tra cột, ID trùng, Std âm, constituent không nhất quán), `audit`, `duplicates`; tab Data explorer trong web | Báo cáo EDA |
| Scorer | `evaluate` (Spearman, RMSE, delta MAE, variance correlation, coverage và lý do NA), `compare_predictions`, `error_groups` | MAE tổng (hiện chỉ có trong `error_groups`); test ví dụ tính tay; đối chiếu scorer chính thức (chưa biết scorer) |
| Split | `make_splits`, `validate_splits`, `overlap_report`, `seen_unseen_design`, manifest cố định | Ghi lại constituent dùng chung; quyết định RQ2 |
| Alignment | `align` bảo thủ và 1 test (biến hình, tách rời, mơ hồ) | Coverage theo task × split; danh sách thất bại; test các ca còn lại |
| Test | 13 test qua khi chạy thử ngày 24/10 (Python 3.11.9, môi trường mới cài từ `requirement.txt`) | Test mới ở mục 6 |

Việc tồn từ tuần 1 ảnh hưởng tuần 2: `plan/task_facts.md` chưa có. Hai câu cần cho tuần này là **câu 1** (official metric tính trong phạm vi nào, quyết định Cương có thêm `scope` hay không) và **câu 3** (thang điểm và chiều, Khang cần cho EDA).

Checkout hiện tại không có `data/`, `results/`, `.venv/` (bị `.gitignore`), nên ai cũng phải làm Bước 0 ở mục 4.

## 3. Bốn điều cần biết trước khi làm

### 3.1 Sửa `align` sẽ làm hỏng kiểm tra split cố định

`validate_splits` so `records_sha256` trong manifest với hash của records **đã qua `align`** (`fingerprint` băm cả `spans`, `alignment_status`, `candidates`). `load_frozen_task` trong cell "data.py" của cả hai notebook gọi `validate_splits`. Hệ quả: các thay đổi sau đều làm notebook (và code local) báo `Manifest does not match normalized records`:

- sửa hàm `align`, `forms`, `occurrences`;
- thêm dạng vào bảng `IRREGULAR`;
- dùng tham số `overrides` (span thủ công).

Đã thử ngày 24/10 trên dữ liệu giả: chỉ cần đổi một dạng trong `IRREGULAR` hoặc thêm một override là `validate_splits` raise.

**Quy tắc tuần 2: chỉ đo, không sửa `align`, `IRREGULAR`, `overrides`.** Thêm test thì được (test không đổi hash). Cách xử lý khi muốn cải thiện alignment được chốt ở họp (quyết định D2, mục 7).

### 3.2 Code dùng chung nằm ở ba chỗ

| Chỗ | Nội dung |
|---|---|
| `src/compartment/*.py` | Bản dùng cho web và test local |
| Cell 6 "data.py" của `B1_kaggle.ipynb` và `B2_kaggle.ipynb` | Bản copy của `data.py` + `splits.py` + loader frozen data |
| Cell 7 "metrics.py" của hai notebook | Bản copy của `metrics.py` + `artifacts.py` |

Cell 6, 7, 8 giống hệt nhau giữa B1 và B2. Ngày 24/10, 20 hàm dùng chung (`align`, `evaluate`, `make_splits`, `write_submission`...) so bằng AST **không lệch** giữa `src/` và hai notebook. `tests/test_notebooks.py` hiện chỉ kiểm tra notebook độc lập (không import `compartment`), **không bắt được lệch code**. Nên mọi sửa ở `src/` phải sửa tay ở cả hai notebook; Cương thêm test chống lệch (mục 6.2).

### 3.3 Sửa notebook bằng thay chuỗi, không ghi lại bằng `json.dump`

Notebook là file JSON, dùng CRLF trên Windows. Đọc rồi `json.dump` lại không cho ra file giống byte cũ (đã thử), nên cả file sẽ bị đổi trong `git diff`. Cách an toàn: đọc bytes, thay đúng một chuỗi, kiểm tra chuỗi chỉ xuất hiện một lần, ghi bytes lại (script ở mục 6.2). Sau đó `git diff --stat` phải cho thấy mỗi file chỉ đổi 1 dòng.

### 3.4 Câu context không lên Git

Mọi file có câu context (EDA, failures, extremes, ví dụ lỗi) để trong `results/analysis/` (bị `.gitignore`). PR chỉ chứa code, test và Markdown **không có câu lấy từ dữ liệu**. Test dùng câu tự viết. EDA mô tả cả file dữ liệu nhưng báo số theo split; không rút quyết định thiết kế từ phần test.

## 4. Bước 0 (mọi người, làm đầu tiên, khoảng 30–45 phút)

**a. Có dữ liệu.** Copy `data/` từ Drive `CompartMent/` hoặc từ máy cũ. Layout mà web và các đoạn code dưới đây dùng:

```text
data/train/en-nn-train.tsv
data/train/en-pv-train.tsv
data/splits/nn/split_manifest.json
data/splits/pv/split_manifest.json
```

Kiểm tra (PowerShell tại thư mục repo), cả bốn phải ra `True`:

```powershell
Test-Path data\train\en-nn-train.tsv, data\train\en-pv-train.tsv, data\splits\nn\split_manifest.json, data\splits\pv\split_manifest.json
```

Nếu chỉ có data ZIP (`cosense-kaggle-data.zip`): giải nén; ZIP để TSV và `split_manifest_nn.json` / `split_manifest_pv.json` chung một thư mục. Sửa đường dẫn trong các đoạn code cho khớp và ghi layout thật vào biên bản họp.

**b. Có môi trường và test qua.**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Kỳ vọng: `13 passed` (sau khi hai PR của tuần này merge sẽ là 17). Không qua thì báo trong nhóm trước khi làm tiếp.

**c. Tạo nhánh riêng** (tên theo quy ước `<tên>/<việc>`):

| Người | Nhánh |
|---|---|
| Khang | `khang/week2` (chỉ cần nếu có sửa file trong Git; EDA nằm ngoài Git) |
| Cương | `cuong/scorer` |
| Đạt | `dat/split-check` (có thể không cần nhánh) |
| Phát | `phat/alignment` |

**d. Cách chạy các đoạn code.** Lưu mỗi đoạn thành file `.py` trong `results/analysis/` (ngoài Git), chạy **từ thư mục gốc repo** để `sys.path.insert(0, 'src')` đúng:

```powershell
.\.venv\Scripts\python.exe -X utf8 results\analysis\<tên file>.py
```

Các đoạn code này đã được chạy thử trên dữ liệu giả ngày 24/10, chưa chạy trên dữ liệu thật. Nếu lỗi đường dẫn thì sửa đường dẫn, đừng sửa logic.

## 5. Thứ tự làm và các hạn

Mỗi người đi qua ba chặng dưới đây, lúc nào rảnh thì làm; số trong ngoặc là bước ở mục 6. Xong chặng nào thì sang chặng sau, không cần chờ ngày.

| Chặng | Khang (EDA) | Cương (scorer) | Đạt (split) | Phát (alignment) |
|---|---|---|---|---|
| **1. Làm trước** (không chờ ai) | Bước 0 → chạy đoạn EDA (2) → đối chiếu web (3) | Bước 0 → viết `test_scorer.py` (2) → thêm MAE (3) | Bước 0 → chạy đoạn split (2) → đối chiếu số (3) | Bước 0 → chạy đoạn coverage (2) |
| **2. Làm tiếp** (cần kết quả chặng 1 của chính mình) | Đọc 10 cao / 10 thấp, xác nhận chiều thang (4) → `eda.md` (5) | Test chống lệch (4) → `pytest` (5) → sửa docs (6) → **mở PR** (7) | Constituent dùng chung, thiết kế và chi phí RQ2 (4–6) → `split_check.md` (7) | Phân loại failures (3) → `test_alignment.py` (4) → **mở PR** (5) → `alignment_report.md` (6) |
| **3. Làm cuối** (cần người khác) | Chương trình và mẫu biên bản họp (6), cần trạng thái của ba người kia | Review PR của Phát (9); sửa PR theo review | Đề xuất D1; D2 soạn cùng Phát, cần `alignment_report.md` (8) | Review PR của Cương (7); sửa PR theo review |

Đạt bước 9 (kiểm tra Kaggle saved version) không phụ thuộc gì, làm lúc nào cũng được.

Bốn hạn cứng:

| Hạn | Việc |
|---|---|
| 20:00 Thứ Tư 28/10 | Mọi người xong Bước 0 (hoặc báo nhóm đang kẹt gì) |
| 20:00 Thứ Năm 29/10 | Cương và Phát đã mở PR, để người kia có 3 ngày review |
| 20:00 Chủ nhật 01/11 | Hai PR đã merge; `eda.md`, `split_check.md`, `alignment_report.md` và đề xuất D1, D2 đã gửi cả nhóm đọc trước họp |
| 21:00 Thứ Hai 02/11 | Họp (mục 8) |

Tuần 3 (28/10–03/11) chạy song song với tuần 2 và không phụ thuộc kết quả tuần 2 (mục 1). Ai có việc tuần 3 thì làm xen vào; thứ tự trên không đổi.

## 6. Việc từng người

### 6.1 Khang: Bộ đọc dữ liệu và EDA

**Mục tiêu:** có một bức tranh rõ ràng về dữ liệu, và xác nhận chiều thang để mọi diễn giải sau đó đúng hướng.

**Đã có:** `load_records`, `audit`, `duplicates`, Data explorer.

**Việc cần làm:**

1. **Bước 0** (mục 4).
2. **Chạy đoạn EDA** dưới đây cho cả `nn` và `pv`. Cần vẽ hình thì cài thêm matplotlib *chỉ trong `.venv` của bạn*: `.\.venv\Scripts\python.exe -m pip install matplotlib` (không thêm vào `requirement.txt`).

```python
import sys; sys.path.insert(0, 'src')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from compartment.data import load_records, SCHEMAS

out = Path('results/analysis'); out.mkdir(parents=True, exist_ok=True)
for task in ['nn', 'pv']:
    recs = load_records(f'data/train/en-{task}-train.tsv', task)
    labels, stds = SCHEMAS[task]['labels'], SCHEMAS[task]['std']
    df = pd.DataFrame(recs)
    for i, l in enumerate(labels): df[l] = [r['labels'][i] for r in recs]
    for i, s in enumerate(stds): df[s] = [r['annotation_std'][i] for r in recs]
    df['n_words'] = df['context'].str.split().str.len()
    n_ctx = df.groupby('target').size()
    sd_in = df.groupby('target')[labels].std(ddof=0)
    print(f'== {task}: {len(df)} dòng, {n_ctx.size} expression')
    print('câu/expression:', n_ctx.agg(['min', 'median', 'max']).to_dict())
    print(df[labels + stds + ['n_words']].describe().round(2))
    print('SD gold trong expression:'); print(sd_in.describe().round(2))
    for l in labels:
        cols = ['context_id', 'target', l, 'context']
        pd.concat([df.nlargest(10, l)[cols].assign(rank='cao'), df.nsmallest(10, l)[cols].assign(rank='thấp')]) \
          .to_csv(out / f'extremes_{task}_{l}.tsv', sep='\t', index=False)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for l in labels: ax[0].hist(df[l], bins=20, alpha=.6, label=l)
    ax[0].set_title(f'{task}: phân bố gold'); ax[0].legend()
    ax[1].hist(n_ctx, bins=range(1, n_ctx.max() + 2)); ax[1].set_title('số câu / expression')
    for l in labels: ax[2].hist(sd_in[l].dropna(), bins=20, alpha=.6, label=l)
    ax[2].set_title('SD gold trong expression')
    fig.tight_layout(); fig.savefig(out / f'eda_{task}.png', dpi=150)
```

   Đầu ra: bảng in ra màn hình, `results/analysis/eda_nn.png`, `eda_pv.png` (mỗi ảnh 3 ô), `extremes_{task}_{output}.tsv`.

   Lưu ý khi đọc số: SD ở đây dùng `ddof=0`, giống `metrics.py`. Trong `metrics.py`, expression chỉ có **1 câu bị loại** khỏi SD và delta; nếu dữ liệu có expression 1 câu thì chỉ đọc SD của expression có ≥2 câu (lọc bằng `sd_in = sd_in[n_ctx >= 2]` trước khi `describe`).
3. **Đối chiếu bằng web** (`streamlit run app/streamlit_app.py`): tab Data explorer, chọn task, "Đọc dữ liệu". Số dòng, số expression, phân bố nhãn phải khớp đoạn code. Bật "Chỉ alignment failures" xem thử vài expression (để biết Phát đang đo gì).
4. **Xác nhận chiều thang.** Mở `extremes_*.tsv`, đọc 10 câu điểm cao và 10 câu thấp mỗi output: điểm cao có đúng nghĩa là "thành phần giữ nghĩa / ghép nghĩa rõ" không? Đối chiếu ví dụ `flea market` (modifier 0,0, head 4,7) trong `docs/METRICS.md` mục 1. Ghi kết luận vào `task_facts.md` câu 3, kèm nguồn.
5. **Viết `results/analysis/eda.md`** (≤1 trang, kèm 2 ảnh ở trên) theo khung:

| Mục | Nội dung |
|---|---|
| 1. Quy mô | Bảng: task, số dòng, số expression, câu/expression (min, median, max), số từ/câu (median). Đối chiếu với NN 3.490 dòng/446 expression, PV 1.567/160 |
| 2. Phân bố gold | Hình ô 1: `ModAvg`, `HeadAvg`, `Avg`. Có lệch về một phía, có đỉnh nào bất thường không |
| 3. Chiều thang | Kết luận 1–2 câu, nêu nguồn xác nhận. Không trích dài câu từ dữ liệu |
| 4. Biến thiên trong expression | Hình ô 3: SD nhỏ nhất, 25%, median, lớn nhất. Expression có SD gần 0 thì khó xếp hạng nội bộ và khó đo delta MAE |
| 5. Annotation Std | Trung bình và median của `ModStd`, `HeadStd`, `Std` (độ phân tán người chấm, không phải đầu ra của model) |
| 6. Độ dài câu | Phân bố số từ; có câu quá ngắn không (liên quan truncation và RQ3) |
| 7. Điều rút ra | 3 gạch đầu dòng ảnh hưởng thiết kế, ví dụ "SD trong expression nhỏ nên delta MAE của B2 khó vượt B1" |

6. **Chuẩn bị họp 02/11** (mục 8): chương trình và mẫu biên bản. Khang là người ghi `plan/meetings.md`.

**Dễ sai:**

- NN có **hai output** (`ModAvg`, `HeadAvg`); báo riêng, không trung bình.
- `ContextID` là chuỗi; không để pandas đổi thành số khi đọc lại file khác.
- `ModStd`/`HeadStd`/`Std` là độ phân tán của người chấm, không phải SD qua context.
- Chưa có điểm từng người chấm, nên không tính được độ tin cậy giữa người chấm (việc của tuần 4).

**Hoàn thành khi:**

- [ ] Đoạn EDA chạy được với cả `nn` và `pv`, số khớp web
- [ ] Đã đọc 10 cao / 10 thấp mỗi output và ghi kết luận chiều thang vào `task_facts.md`
- [ ] `eda.md` đủ 7 mục và đã trình bày 5 phút trong họp

### 6.2 Cương: Scorer nội bộ

**Mục tiêu:** scorer đủ số (có MAE), có test kiểm chứng bằng tay, và có test bắt lệch code giữa `src/` và hai notebook.

**Đã có:** `evaluate`, `compare_predictions`, `error_groups`, các test trong `tests/test_pipeline.py`.

**Việc cần làm:**

1. **Bước 0** (mục 4) rồi `git checkout -b cuong/scorer`.
2. **Viết test ví dụ tính tay**, file mới `tests/test_scorer.py`. Gold [4,8; 4,5; 1,0; 1,3] và pred [0,91; 0,62; 0,40; 0,35]: hạng gold (4, 3, 1, 2), hạng pred (4, 3, 2, 1), Σd² = 2, nên ρ = 1 − 6×2/(4×15) = 0,8; đảo dấu dự đoán ra −0,8.

```python
import pandas as pd
import pytest
from compartment.metrics import evaluate


def test_spearman_and_mae_by_hand():
    f = pd.DataFrame(dict(target=['a', 'a', 'b', 'b'], gold_Avg=[4.8, 4.5, 1.0, 1.3], pred_Avg=[0.91, 0.62, 0.40, 0.35]))
    out = evaluate(f, 'pv')[0]['outputs']['Avg']
    assert out['spearman']['value'] == pytest.approx(0.8)
    assert out['mae'] == pytest.approx((3.89 + 3.88 + 0.60 + 0.95) / 4)
    f['pred_Avg'] = -f['pred_Avg']
    assert evaluate(f, 'pv')[0]['outputs']['Avg']['spearman']['value'] == pytest.approx(-0.8)
```

   Chạy test này ngay, trước khi thêm MAE: phải fail với `KeyError: 'mae'`; sau bước 3: phải qua. Đó là bằng chứng test thật sự kiểm tra MAE.
3. **Thêm MAE vào `evaluate` và hai notebook.** MAE tổng hiện chưa có trong JSON (`docs/METRICS.md` mục 8 cũng ghi điều này). Chạy script dưới đây từ thư mục gốc repo (lưu thành file trong `results/analysis/`). Nó thay đúng một dòng trong `src/compartment/metrics.py` và cell 7 của B1, B2 (mục 3.3), và dừng nếu chuỗi cần thay không xuất hiện đúng một lần:

```python
from pathlib import Path
old = "rmse=float(np.sqrt(np.mean((y[valid]-p[valid])**2))) if valid.any() else None,"
new = old + " mae=float(np.mean(np.abs(y[valid]-p[valid]))) if valid.any() else None,"
for path in ['src/compartment/metrics.py', 'notebooks/B1_kaggle.ipynb', 'notebooks/B2_kaggle.ipynb']:
    p = Path(path); text = p.read_bytes().decode('utf-8')
    assert text.count(old) == 1, (path, text.count(old))
    p.write_bytes(text.replace(old, new).encode('utf-8')); print('đã sửa', path)
```

   Kiểm tra: `git diff --stat` chỉ có đúng 3 file, mỗi file đổi 1 dòng. Bundle kết quả cũ (tạo trước thay đổi này) không có field `mae`; bundle mới có.
4. **Thêm test chống lệch code**, nối vào cuối `tests/test_notebooks.py` (thêm `import pytest` sau dòng `from pathlib import Path`). Test so cây cú pháp (AST) của 18 hàm dùng chung giữa `src/` và từng notebook:

```python
SHARED = {'data': ['occurrences', 'forms', 'align', 'load_records', 'audit', 'duplicates'],
          'splits': ['make_splits', 'validate_splits', 'fingerprint'],
          'metrics': ['correlation', 'evaluate', 'compare_predictions', 'error_groups'],
          'artifacts': ['prediction_frame', 'export_bundle', 'read_bundle', 'write_submission', 'validate_submission']}


def functions(code):
    return {n.name: ast.dump(n) for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize('variant', ['B1', 'B2'])
def test_notebook_shared_code_matches_src(variant):
    src = {}
    for module, names in SHARED.items():
        found = functions((ROOT / f'src/compartment/{module}.py').read_text(encoding='utf-8'))
        src.update({name: found[name] for name in names})
    nb = json.loads((ROOT / f'notebooks/{variant}_kaggle.ipynb').read_text(encoding='utf-8'))
    code = {}
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code.update(functions(''.join(cell['source'])))
    assert [name for name in src if code.get(name) != src[name]] == []
```

   Kiểm chứng: nếu chỉ sửa `src/` mà quên notebook thì cả hai test `[B1]`, `[B2]` fail.
5. **Chạy `pytest -q`.** Kỳ vọng 13 + 1 + 2 = **16 passed** (17 sau khi PR của Phát merge).
6. **Sửa `docs/METRICS.md`.** Mục 8, gạch đầu dòng thứ hai nói "không có field overall MAE riêng trong `evaluate()` JSON" sẽ **sai** sau bước 3: sửa lại cho đúng (field `mae` mỗi output, cùng tập dòng hợp lệ với RMSE). Thêm ví dụ gọi `evaluate` (không thêm trang docs mới):

```python
from compartment.metrics import evaluate
metrics, stats = evaluate(frame, 'nn')   # frame có cột target, gold_ModAvg, pred_ModAvg, gold_HeadAvg, pred_HeadAvg
metrics['outputs']['ModAvg']['spearman']['value'], metrics['outputs']['ModAvg']['mae']
```

7. **Mở PR trước 20:00 Thứ Năm 29/10.** File đổi: `src/compartment/metrics.py`, `notebooks/B1_kaggle.ipynb`, `notebooks/B2_kaggle.ipynb`, `tests/test_scorer.py`, `tests/test_notebooks.py`, `docs/METRICS.md`. Người review: Phát.
8. **Chưa làm (chờ thông tin):**
   - Thêm tham số `scope` vào `evaluate`: chỉ khi `task_facts.md` câu 1 cho biết official metric tính khác Spearman trên toàn dòng. Chưa biết thì không làm.
   - Đối chiếu scorer chính thức đến 4 chữ số thập phân: chỉ khi có scorer chính thức; ghi "chờ" vào biên bản.
9. Review PR của Phát.

**Hoàn thành khi:**

- [ ] `evaluate` trả `mae` cho từng output; `src/` và hai notebook khớp nhau
- [ ] `test_scorer.py` và test chống lệch đã qua; `pytest -q` không còn test fail
- [ ] `docs/METRICS.md` mục 8 không còn câu sai và có ví dụ gọi
- [ ] PR đã review và merge; scorer chính thức: ghi trạng thái "chờ" nếu chưa có

### 6.3 Đạt: Split cố định và quyết định RQ2

**Mục tiêu:** chắc split đang dùng đúng như docs mô tả, biết rõ những gì split không chứng minh, và đưa đề xuất cho hai quyết định D1, D2.

**Đã có:** frozen split theo expression (≈70/15/15, seed 42), exact-duplicate context giữ cùng partition, manifest và data ZIP; `overlap_report`, `seen_unseen_design`, test.

**Việc cần làm:**

1. **Bước 0** (mục 4).
2. **Chạy đoạn kiểm tra split** cho `nn` và `pv`:

```python
import sys, json
sys.path.insert(0, 'src')
from compartment.data import load_records
from compartment.splits import validate_splits, overlap_report, seen_unseen_design

for task in ['nn', 'pv']:
    recs = load_records(f'data/train/en-{task}-train.tsv', task)
    m = json.load(open(f'data/splits/{task}/split_manifest.json', encoding='utf-8-sig'))
    print(task, 'validate_splits:', validate_splits(recs, m))
    by_id = {r['context_id']: r for r in recs}
    for s, ids in m['partitions'].items():
        print(f'  {s:5} rows={len(ids):5} targets={len({by_id[i]["target"] for i in ids})}')
    ov = overlap_report(recs, m)
    for k in ['train_dev', 'train_test', 'dev_test']:
        print('  shared constituents', k, len(ov[k]['constituent_overlap']), ov[k]['constituent_overlap'][:10])
    d = seen_unseen_design(recs, m, support_count=1)
    print('  RQ2 support', len(d['support']), 'query', len(d['query']), 'excluded targets', len(d['excluded_targets']))
```

3. **Đối chiếu số.** Kỳ vọng theo docs:

| Task | train | dev | test (internal) |
|---|---|---|---|
| NN | 2.479 dòng | 504 dòng, **67 expression** (số expression dev của pilot NN) | 507 dòng |
| PV | 1.100 dòng | 231 dòng | 236 dòng |

   Tổng NN 446 expression, PV 160 expression. `validate_splits` phải in `True`. **Lệch số hoặc `validate_splits` lỗi thì dừng, báo nhóm ngay** (có thể data ZIP/manifest khác bản đã dùng cho pilot).
4. **Constituent dùng chung.** Ghi số lượng và vài ví dụ giữa train/dev, train/test, dev/test. Ý nghĩa: expression chưa gặp khi train không có nghĩa là thành phần (modifier/head/verb/particle) chưa gặp; đây là điều cần nêu trong paper.
5. **Hiểu thiết kế RQ2** (`seen_unseen_design`, `support_count=1`): với mỗi expression trong test, lấy 1 câu làm *support* (cho vào tập train ở điều kiện *seen*), các câu còn lại làm *query*. Điều kiện *unseen* thêm số dòng tương đương từ train làm đối chứng. Cả hai điều kiện chấm trên cùng query, dev giữ nguyên. Hàm tự ghi hai hạn chế: phân bố từ vựng/nhãn của dòng đối chứng có thể khác; chưa nhìn thấy trong task không có nghĩa chưa thấy khi pretrain BERT.
6. **Chi phí thật của RQ2 (cần biết trước khi quyết định D1).** Hai notebook hiện **từ chối** manifest `paired_seen_unseen` (`load_frozen_task` raise "Expected the main target-disjoint train/dev/test manifest"). Muốn chạy RQ2 phải: sửa loader trong cell 6 của cả hai notebook, đóng gói manifest chẩn đoán vào data bundle, và chạy thêm ít nhất 2 điều kiện × NN/PV × B1/B2 = 8 run (chưa tính nhiều seed).
7. **Viết `results/analysis/split_check.md`** (ngắn): bảng số khớp, constituent dùng chung, kích thước thiết kế RQ2, chi phí ở bước 6.
8. **Soạn đề xuất cho D1 và D2** (mục 7), mỗi đề xuất ≤5 dòng. D2 soạn cùng Phát vì cần số liệu từ báo cáo alignment.
9. *(Tùy chọn, 15 phút, mở đường cho tuần 3)* Mở Kaggle, kiểm tra saved version của hai run NN cũ: Output còn `best.pt`, tokenizer, config không? Ghi kết quả vào biên bản để tuần 3 biết có cần train lại NN hay không.

**Hoàn thành khi:**

- [ ] `validate_splits` in `True` và số dòng/expression khớp bảng (hoặc đã báo nhóm nếu lệch)
- [ ] `split_check.md` có đủ bảng số, constituent dùng chung và chi phí RQ2
- [ ] Đã có đề xuất D1, D2 để đưa vào họp

### 6.4 Phát: Định vị cụm từ (alignment)

**Mục tiêu:** biết chính xác bao nhiêu dòng bị fallback, vì sao, ở split nào, và chuẩn bị số liệu cho quyết định D2. **Không sửa `align`** (mục 3.1).

**Đã có:** `align` bảo thủ. Quy tắc (đọc từ code, để biết thất bại đến từ đâu):

| Loại | Quy tắc |
|---|---|
| NN | Hai thành phần đều được biến hình (số nhiều, -ed, -ing, bảng bất quy tắc ngắn `IRREGULAR`). Phần đầu phải đứng trước phần sau; giữa hai phần **chỉ** được có khoảng trắng hoặc gạch nối (-, –, —) |
| PV | Động từ được biến hình; tiểu từ **không** biến hình. Động từ phải đứng **trước** tiểu từ, khoảng cách giữa chúng không bị giới hạn |
| Kết quả | Đúng 1 cặp hợp lệ → có `spans`; trạng thái `exact` nếu chữ trong câu giống hệt thành phần, ngược lại `inflection`. 0 cặp → `missing`. ≥2 cặp → `ambiguous` |
| Fallback | `ambiguous` và `missing` vẫn giữ dòng, dùng expression thay cho spans khi tokenize (có log) |

**Việc cần làm:**

1. **Bước 0** (mục 4) rồi `git checkout -b phat/alignment`.
2. **Chạy đoạn coverage**, cho bảng task × split × trạng thái và file danh sách thất bại:

```python
import json, sys
from pathlib import Path
sys.path.insert(0, 'src')
import pandas as pd
from compartment.data import load_records

out = Path('results/analysis'); out.mkdir(parents=True, exist_ok=True)
rows, fails = [], []
for task in ['nn', 'pv']:
    recs = load_records(f'data/train/en-{task}-train.tsv', task)
    m = json.load(open(f'data/splits/{task}/split_manifest.json', encoding='utf-8-sig'))
    split_of = {i: s for s, ids in m['partitions'].items() for i in ids}
    for r in recs:
        s = split_of[r['context_id']]
        rows.append(dict(task=task, split=s, status=r['alignment_status']))
        if r['alignment_status'] in ('ambiguous', 'missing'):
            fails.append(dict(task=task, split=s, context_id=r['context_id'], target=r['target'],
                              status=r['alignment_status'], n_candidates=len(r['candidates']), context=r['context']))
tab = pd.DataFrame(rows).groupby(['task', 'split', 'status']).size().unstack(fill_value=0)
tab['n'] = tab.sum(axis=1)
tab['fallback_%'] = 100 * (tab.get('ambiguous', 0) + tab.get('missing', 0)) / tab['n']
print(tab.round(1))
pd.DataFrame(fails).to_csv(out / 'alignment_failures.tsv', sep='\t', index=False)
```

   So với ngưỡng đề xuất: `fallback_%` (ambiguous + missing) ≤ 5% mỗi task. Vượt ngưỡng thì chưa sửa, chỉ ghi vào báo cáo để họp quyết định D2.
3. **Phân loại thất bại.** Mở `alignment_failures.tsv`, thêm cột `loai` bằng tay (Excel/Sheets cũng được), dùng bộ nhãn dưới đây; nhãn nào chưa đủ thì thêm nhãn mới và ghi định nghĩa:

| Nhãn | Ý nghĩa | Trạng thái thường gặp |
|---|---|---|
| a. Tiểu từ trước động từ | PV, tiểu từ đứng trước động từ, `align` không bao giờ ghép | `missing` |
| b. Động từ bất quy tắc ngoài bảng | Dạng không nằm trong `IRREGULAR` | `missing` |
| c. Danh từ bất quy tắc / biến hình lạ | NN, ví dụ dạng số nhiều không theo quy tắc | `missing` |
| d. Lặp trong câu | Expression xuất hiện ≥2 lần, thật sự mơ hồ | `ambiguous` |
| e. Bị chen giữa | NN, dấu câu hoặc từ khác nằm giữa hai phần | `missing` |
| f. Câu không chứa expression | Nghi lỗi dữ liệu hoặc diễn đạt khác | `missing` |
| g. Khác | Ghi chú cụ thể | |

   Báo số dòng mỗi nhãn theo task, và theo split nếu chênh lệch rõ.
4. **Viết `test_alignment.py`** (file mới `tests/test_alignment.py`). Câu tự viết, không lấy từ dữ liệu. Ca "tiểu từ trước động từ" ghi lại *giới hạn đã biết* (kết quả hiện tại là `missing`); nếu sau này sửa `align` thì đổi test này có chủ đích:

```python
from compartment.data import align

PV = dict(task='pv', part1='take', part2='off')
NN = dict(task='nn', part1='flea', part2='market')


def status(base, text):
    return align({**base, 'context': text})['alignment_status']


def test_alignment_status_cases():
    assert status(PV, 'The plane will take off soon.') == 'exact'
    assert status(PV, 'She took her coat off.') == 'inflection'
    assert status(PV, 'He took it.') == 'missing'
    assert status(PV, 'Off he took.') == 'missing'  # tiểu từ đứng trước động từ: giới hạn đã biết, chưa xử lý
    assert status(NN, 'A flea-market opened.') == 'exact'
    assert status(NN, 'Two flea markets opened.') == 'inflection'
    assert status(NN, 'The flea, then the market.') == 'missing'  # dấu phẩy nằm giữa hai phần
    assert status(NN, 'A flea market and a flea market.') == 'ambiguous'
```

   Có thể thêm ca cho từng loại bẫy ở danh sách tuần 1 (miễn là test phản ánh đúng hành vi hiện tại). Test của Cương đã bảo đảm `src/` và notebook khớp nhau, nên test này đủ.
5. **Mở PR trước 20:00 Thứ Năm 29/10.** File đổi: chỉ `tests/test_alignment.py`. Người review: Cương.
6. **Viết `results/analysis/alignment_report.md`** (≤1 trang): bảng coverage, bảng phân loại, với **mỗi nhãn** một dòng đề xuất cách xử lý ở mức ý tưởng (ví dụ: xử lý ở bước tokenize thay vì ở `records`; chấp nhận; sửa dữ liệu) kèm số dòng bị ảnh hưởng. Nếu có `results/nn_B1_B2_seed42/*`, đối chiếu `run_metadata.json` mục `preprocessing` (số dòng fallback và truncated theo split) với số của bạn.
7. Review PR của Cương.

**Hoàn thành khi:**

- [ ] Có bảng coverage task × split × trạng thái và `alignment_failures.tsv`
- [ ] Mọi dòng thất bại đã có nhãn `loai`
- [ ] `test_alignment.py` qua; PR đã review và merge
- [ ] `alignment_report.md` có đề xuất cho từng nhãn, sẵn sàng cho D2

## 7. Quyết định cần chốt trong họp 02/11

Khang ghi quyết định, người chịu trách nhiệm và hạn vào `plan/meetings.md`.

### D1. Có chạy đối chứng seen/unseen (RQ2) không?

Chi phí đã nêu ở 6.3 bước 6.

| Phương án | Nội dung | Khi nào hợp lý |
|---|---|---|
| A (đề xuất) | Không chạy đối chứng seen/unseen. Báo kết quả trên expression chưa gặp (đúng cách RQ2 hiện được định nghĩa) và ghi vào Limitations rằng thiết kế không so được seen với unseen | Mặc định; không tốn thêm run |
| B | Làm đối chứng: sửa loader hai notebook, chạy ≥8 run bổ sung | Chỉ khi còn quota GPU sau Mốc C, và thiết kế (`support_count`, ngân sách dòng) được khóa **trước khi xem kết quả** |

Đề xuất: chọn A, xem lại ở G2 (24/11).

### D2. Cải thiện alignment mà không phá manifest

Chỉ cần quyết định nếu báo cáo của Phát cho thấy fallback > 5% hoặc fallback gây lỗi rõ.

| Phương án | Nội dung | Cái giá |
|---|---|---|
| P1 | Giữ nguyên `align` và `records`. Xử lý ca khó ở bước tokenize trong notebook (cell 8), sau `load_frozen_task` | Hash manifest không đổi; chỉ sửa tokenization, hai notebook |
| P2 | Đổi `fingerprint()` để chỉ băm các trường gốc (ID, task, target, thành phần, context, nhãn, Std), không băm spans/status. Sinh manifest mới: membership giữ nguyên từng ID (so bằng code), `records_sha256` mới; cập nhật `files_sha256` trong `dataset_info.json`; đóng gói lại data ZIP; đăng bản Dataset mới trên Kaggle | Tốn khoảng nửa ngày; các run đã có phải ghi chú manifest/dataset cũ; mọi người phải dùng bản mới |
| P3 | Giữ alignment hiện tại làm bản cuối cho paper, báo coverage/fallback như một Limitation | Không tốn công; chấp nhận mất điểm ở dòng bị fallback |

Đề xuất: **P3 mặc định**; nếu số liệu cho thấy phải cải thiện thì **P1** trước (ít rủi ro nhất); chỉ dùng P2 khi P1 không đủ. Dùng `overrides` thủ công cũng đổi hash nên thuộc P2.

### D3. Scorer chính thức

Nếu `task_facts.md` câu 1 vẫn chưa trả lời: giữ Spearman trên toàn dòng như hiện tại, không thêm `scope`. Hạn trả lời câu 1 và câu 3 trước G1 (03/11).

### D4. Chiều thang

Xác nhận kết luận từ EDA (Khang) và ghi vào `task_facts.md` câu 3.

### D5. Quy trình PR

Review chéo: Cương ↔ Phát. Kết quả/báo cáo có context ở `results/analysis/`, không đưa vào PR.

## 8. Họp 02/11 (21:00, 45 phút)

| Phút | Nội dung | Ai |
|---|---|---|
| 0–5 | EDA: bảng quy mô, chiều thang, SD trong expression | Khang |
| 5–10 | Scorer: MAE, test, test chống lệch, trạng thái scorer chính thức | Cương |
| 10–15 | Split: số khớp, constituent dùng chung, chi phí RQ2 | Đạt |
| 15–20 | Alignment: coverage, phân loại failures | Phát |
| 20–38 | Quyết định D1–D5 | Cả nhóm |
| 38–45 | Rà việc tuần 3, đã bắt đầu từ 28/10 (Đạt chạy PV B1/B2 là việc chính; Khang xác nhận format nộp; Cương E0; Phát đọc lỗi NN): ai còn thiếu gì cho G1 ngày 03/11 | Khang |

Biên bản mẫu cho `plan/meetings.md`:

- **Họp 02/11/2026 (tuần 2).** Có mặt: ...
- **Trạng thái:** EDA [xong/chưa], scorer [xong/chưa], split [xong/chưa], alignment [xong/chưa]. Việc chưa xong và lý do: ...
- **Quyết định:** D1: ... · D2: ... · D3: ... · D4: ... · D5: ...
- **Việc tuần 3:** (người) (việc) (hạn)
- **Vướng mắc / cần hỏi ban tổ chức:** ...
- **Layout dữ liệu thực tế đã dùng:** ...

## 9. Checklist tổng cuối tuần

- [ ] Cả bốn người qua Bước 0 và `pytest -q` (13 test) trước khi làm
- [ ] Khang: `eda.md` và chiều thang đã ghi vào `task_facts.md`
- [ ] Cương: PR scorer đã merge; `pytest -q` = 16 passed (17 với PR của Phát)
- [ ] Đạt: `split_check.md`; đề xuất D1, D2
- [ ] Phát: `alignment_report.md`, `alignment_failures.tsv`, PR test đã merge
- [ ] Họp 02/11 đã diễn ra, `plan/meetings.md` có quyết định D1–D5

## 10. Nếu trễ

Thứ tự ưu tiên khi không kịp hết (làm từ trên xuống):

1. **Phát: bảng coverage** và **Đạt: kiểm tra split** (số liệu cho D1, D2).
2. **Cương: MAE + test tính tay + test chống lệch** (nếu không kịp, chỉ làm bước 2 và 3).
3. **Khang: `eda.md`** (nếu không kịp, báo bảng quy mô và chiều thang trong họp, viết đầy đủ sau).
4. Báo cáo chi tiết (`alignment_report.md` phần đề xuất, `split_check.md`) có thể hoàn thiện đầu tuần 3.

Việc chưa xong ghi trong biên bản kèm người và hạn mới. Không chặn Mốc A của tuần 3.
