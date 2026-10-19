"""Generate the standalone notebook from reviewed Python sources; stdlib only."""
import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def cell(kind, source):
        source = dedent(source).strip() + '\n'
        item = {'cell_type': kind, 'id': f'cell-{len(cells):02d}', 'metadata': {},
                'source': source.splitlines(keepends=True)}
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', '''
    # CoSense EN-NN — chuẩn bị dữ liệu và fine-tune B1/B2 trên Kaggle

    **Đọc hướng dẫn `docs/KAGGLE_EN_NN_GUIDE.md` trước lần chạy đầu.**

    Notebook tự chứa code; không cần clone GitHub, GitHub token hay API key.
    Dùng Dataset tạo từ `dist/kaggle/cosense-en-nn-data.zip` và giữ notebook/Dataset private.
    Chỉ upload sau khi kiểm tra điều khoản sử dụng dữ liệu của BTC.

    Ba chế độ: `prepare` (mặc định, kiểm tra + tokenize + B0, không optimizer),
    `smoke` (tối đa 128 mẫu train, vài bước optimizer/GPU), `train` (fine-tune đầy đủ).
    Mọi đánh giá/chọn checkpoint chỉ dùng **validation**. Final chỉ được kiểm tra tính toàn vẹn;
    không xuất dữ liệu train-ready, prediction hoặc metric final.

    Lần đầu: import notebook → Add Input Dataset → Internet On → giữ `MODE="prepare"` → Run All.
    Không cần GPU cho `prepare`. Chọn GPU trước `smoke`/`train` và chạy lại từ đầu sau khi session restart.
    ''')
    cell('markdown', '''
    ## 1. Cấu hình — sửa tại cell này

    Lần đầu giữ nguyên. `DATA_ROOT=""` sẽ tự tìm manifest dù Kaggle đổi đường dẫn mount.
    Nếu gắn nhiều bản Dataset, điền đường dẫn thư mục chứa `bundle_manifest.json`.
    `unseen_compound` là split B chính; `seen_context` là split A bổ sung.
    Một GPU được dùng, kể cả khi Kaggle cấp T4 x2. Chưa chạy distributed/multi-GPU.
    ''')
    cell('code', '''
    from pathlib import Path
    from datetime import datetime, timezone
    import json
    import os
    import sys

    MODE = "prepare"                    # "prepare", "smoke", "train"
    SETTING = "unseen_compound"          # hoặc "seen_context"
    VARIANTS = ["B1", "B2"]
    SEED = 42
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    EPOCHS = 5
    PATIENCE = 2
    SMOKE_STEPS = 8
    MODEL_ID = "distilbert/distilbert-base-uncased"
    MODEL_REVISION = "main"              # lần đầu resolve SHA; pin SHA đó cho các lần sau
    INSTALL_DEPS = True
    DATA_ROOT = ""                      # để trống để tự tìm; không dùng đường dẫn Windows ở Kaggle
    RESUME_FROM = {"B1": "", "B2": ""}  # đường dẫn .../B1/last.pt; giữ cả best.safetensors cạnh nó
    RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    WORK = Path("/kaggle/working")
    INPUT = Path("/kaggle/input")

    if MODE not in {"prepare", "smoke", "train"}:
        raise ValueError("MODE không hợp lệ")
    if SETTING not in {"unseen_compound", "seen_context"}:
        raise ValueError("SETTING không hợp lệ")
    if not VARIANTS or len(set(VARIANTS)) != len(VARIANTS) or not set(VARIANTS) <= {"B1", "B2"}:
        raise ValueError("VARIANTS chỉ nhận B1/B2, không lặp")
    if min(BATCH_SIZE, MAX_LENGTH, EPOCHS, PATIENCE, SMOKE_STEPS) < 1 or LEARNING_RATE <= 0:
        raise ValueError("Cấu hình số phải dương")
    if not INPUT.is_dir():
        raise RuntimeError("Chạy notebook trên Kaggle. Local chỉ build/verify; không train trên CPU local.")
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT = WORK / "cosense_outputs" / f"{SETTING}_{MODE}_seed{SEED}_{RUN_TAG}"
    OUTPUT.mkdir(parents=True, exist_ok=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("Mode:", MODE, "| setting:", SETTING, "| output:", OUTPUT)
    ''')
    cell('markdown', '''
    ## 2. Cài dependency và kiểm tra runtime

    Giữ PyTorch CUDA có sẵn trên Kaggle. Cell chỉ cài ba dependency đã pin.
    Đây là cấu hình khởi đầu, không khẳng định tương thích mọi Kaggle image.
    Nếu đã import transformers/torch trước khi đổi dependency, restart session rồi Run All.
    Nếu Internet bị tắt thì pip/tải tokenizer lần đầu sẽ thất bại.
    ''')
    requirements = (ROOT / 'requirements-kaggle.txt').read_text(encoding='utf-8')
    cell('code', f'''
    import subprocess
    REQUIREMENTS = {requirements!r}
    requirements_path = WORK / "requirements-cosense.txt"
    requirements_path.write_text(REQUIREMENTS, encoding="utf-8")
    if INSTALL_DEPS:
        if any(name in sys.modules for name in ("transformers", "tokenizers", "huggingface_hub")):
            raise RuntimeError("Dependency đã import. Restart session trước khi cài/đổi phiên bản.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "-r", str(requirements_path)], check=True)
    import torch
    print("Python:", sys.version.split()[0], "| torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU 0:", torch.cuda.get_device_name(0))
    if MODE != "prepare" and not torch.cuda.is_available():
        raise RuntimeError("Bật Accelerator GPU rồi restart session trước smoke/train.")
    ''')
    cell('markdown', '''
    ## 3. Nạp code đi kèm notebook — không cần sửa cell này

    Nguồn được nhúng từ `scripts/data_utils.py` và `scripts/kaggle_en_nn.py` lúc build.
    Notebook ghi code vào thư mục working riêng, kiểm tra hash rồi import.
    Muốn cập nhật code: sửa source local, build lại notebook, import lại bản mới.
    ''')
    sources = {name: (ROOT / 'scripts' / name).read_text(encoding='utf-8')
               for name in ('data_utils.py', 'kaggle_en_nn.py')}
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    cell('code', 'import hashlib\nimport importlib\n'
         + 'EMBEDDED_SOURCES = ' + repr(sources) + '\n'
         + 'CODE_SHA256 = ' + repr(hashes) + '\n'
         + dedent('''
         CODE_DIR = OUTPUT / "code"
         CODE_DIR.mkdir()
         for name, source in EMBEDDED_SOURCES.items():
             if hashlib.sha256(source.encode("utf-8")).hexdigest() != CODE_SHA256[name]:
                 raise ValueError("Embedded code hash mismatch")
             (CODE_DIR / name).write_text(source, encoding="utf-8")
         sys.path.insert(0, str(CODE_DIR))
         for name in ("kaggle_en_nn", "data_utils"):
             sys.modules.pop(name, None)
         import kaggle_en_nn as pipeline
         pipeline.write_json(OUTPUT / "code_hashes.json", CODE_SHA256)
         print("Code loaded:", CODE_SHA256)
         ''').strip())
    cell('markdown', '''
    ## 4. Tìm Dataset, kiểm tra raw TSV và replay split đã khóa

    Hỗ trợ Dataset đã giải nén và Dataset còn ZIP. Không hard-code username/slug.
    Cell kiểm tra hash từng file, schema, nhãn hợp lệ, row keys, coverage,
    leakage context/Compound, và replay membership đúng thuật toán v2.
    Không gọi random split hoặc tạo membership mới. Mọi lỗi đều dừng pipeline.
    ''')
    cell('code', '''
    bundle_root = pipeline.discover_bundle(INPUT, WORK, DATA_ROOT)
    selected, integrity = pipeline.load_checked(bundle_root, SETTING)
    b0 = pipeline.save_prepared(selected, integrity, OUTPUT)
    pipeline.write_json(OUTPUT / "environment.json", pipeline.environment_report())
    print("Bundle:", bundle_root)
    print(json.dumps(integrity, indent=2, ensure_ascii=False))
    print("B0 validation E:", round(b0["metrics"]["compound_macro_two_output_mae"], 6))
    print("B1 inputs:", list(pipeline.model_inputs(selected["train"][0], "B1")))
    print("B2 inputs:", list(pipeline.model_inputs(selected["train"][0], "B2")))
    print("Prepared labels order: [ModAvg, HeadAvg]. ID/Avg/Std không nằm trong input.")
    ''')
    cell('markdown', '''
    ## 5. Tokenize B1/B2 và lưu dữ liệu train-ready

    B1: `Compound`; B2: text pair `(Compound, Context)`.
    Dùng nguyên văn nguồn; không xóa stopword, dấu câu, stemming hay sửa số nhiều.
    Tokenizer `uncased` tự xử lý chữ hoa/thường theo backbone.
    Không truyền token_type_ids vì DistilBERT không dùng chúng.
    Chưa padding ở bước lưu; DataLoader padding động trong từng batch.
    **Không tự truncate:** nếu mẫu vượt MAX_LENGTH, cell dừng để bạn xem lại cấu hình
    cho cả B1/B2. Cách này tránh cắt mất cách dùng mục tiêu.
    ''')
    cell('code', '''
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    resolved_revision = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=resolved_revision, use_fast=True)
    if MAX_LENGTH > tokenizer.model_max_length:
        raise ValueError("MAX_LENGTH vượt giới hạn backbone")
    tokenizer.save_pretrained(OUTPUT / "tokenizer")
    model_lock = {"model_id": MODEL_ID, "requested_revision": MODEL_REVISION,
                  "resolved_revision": resolved_revision, "max_length": MAX_LENGTH}
    pipeline.write_json(OUTPUT / "model_lock.json", model_lock)
    tokenized, token_lengths = pipeline.tokenize_all(selected, tokenizer, MAX_LENGTH, OUTPUT)
    print("PIN SHA này cho các lần train sau:", resolved_revision)
    print(json.dumps(token_lengths, ensure_ascii=False, indent=2))
    print("Prepare hoàn tất. MODE=prepare sẽ bỏ qua optimizer ở cell kế tiếp.")
    ''')
    cell('markdown', '''
    ## 6. Tùy chọn: smoke hoặc train trên GPU

    `prepare`: bỏ qua toàn bộ training. `smoke`: tối đa128 mẫu train, tối đa8 bước,
    một epoch; validation vẫn dùng toàn bộ membership, kết quả smoke không dùng làm headline.
    `train`: tối đa5 epochs, early stopping patience2; chọn best bằng MAE trung bình
    theo Compound và hai output, sau clipping prediction về [0,5].
    Backbone + masked mean pooling (bao gồm special tokens, loại padding) + linear2;
    loss MSE, AdamW, weight decay0.01, warmup10%, linear decay, grad clipping1.0, FP32.
    Hai model dùng cùng budget/config/seed và cùng pretrained revision.

    Checkpoint: `best.safetensors` để inference; `last.pt` gồm model/optimizer/scheduler/RNG
    để resume sau epoch đã hoàn thành. Chỉ load last.pt do chính notebook này tạo.
    Interrupt giữa epoch thì phải chạy lại epoch đó; interrupt trước epoch đầu chưa có last.pt.
    ''')
    cell('code', '''
    results = []
    if MODE == "prepare":
        print("Không train: đổi MODE thành smoke khi bạn sẵn sàng chạy thử GPU.")
    else:
        for variant in VARIANTS:
            cfg = {"mode": MODE, "setting": SETTING, "variant": variant, "seed": SEED,
                   "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                   "max_length": MAX_LENGTH, "epochs": EPOCHS, "patience": PATIENCE,
                   "smoke_steps": SMOKE_STEPS, "model_id": MODEL_ID,
                   "model_revision": resolved_revision, "code_sha256": CODE_SHA256,
                   "pooling": "masked_mean_including_special_tokens", "head": "linear_two_outputs",
                   "loss": "MSE", "weight_decay": 0.01, "warmup_ratio": 0.1,
                   "precision": "float32", "gradient_clip_norm": 1.0}
            result = pipeline.train_variant(selected, tokenized[variant], tokenizer, cfg,
                                            OUTPUT / "runs" / variant, integrity,
                                            resume_from=RESUME_FROM.get(variant, ""))
            results.append(result)
    summary = [{"model": "B0", "validation_E": b0["metrics"]["compound_macro_two_output_mae"]}]
    summary.extend({"model": r["variant"], "validation_E": r["metrics"]["compound_macro_two_output_mae"],
                    "mode": MODE, "seed": SEED} for r in results)
    pipeline.write_json(OUTPUT / "comparison_validation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(results) == 2:
        by_model = {r["variant"]: r for r in results}
        delta = (by_model["B1"]["metrics"]["compound_macro_two_output_mae"]
                 - by_model["B2"]["metrics"]["compound_macro_two_output_mae"])
        print("Delta E(B1)-E(B2):", delta, "| dương = B2 ít sai hơn; một seed chưa đủ kết luận ổn định")
    ''')
    cell('markdown', '''
    ## 7. Kiểm tra load lại checkpoint và inference

    Nạp lại đầy đủ custom regression head, config và tokenizer.
    So ba prediction validation đầu với prediction đã lưu; không đụng final.
    Nếu chỉ prepare thì cell bỏ qua. Web sau này có thể dùng `predict_texts` cùng checkpoint này.
    ''')
    cell('code', '''
    if results:
        import csv
        for result in results:
            run_dir = OUTPUT / "runs" / result["variant"]
            probe = selected["validation"][:3]
            fresh = pipeline.predict_texts(run_dir, [r["Compound"] for r in probe],
                                          [r["Context"] for r in probe], device="cuda:0")
            with (run_dir / "validation_predictions.tsv").open(encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle, delimiter="\\t"))[:3]
            for actual, expected in zip(fresh, saved):
                for value, column in zip(actual["raw"], ["raw_ModAvg", "raw_HeadAvg"]):
                    if abs(value - float(expected[column])) > 1e-4:
                        raise ValueError("Reload prediction mismatch")
            print(result["variant"], "reload/inference PASS")
        pipeline.write_json(OUTPUT / "reload_check.json", {"passed": True, "final_evaluated": False})
    else:
        print("Prepare only: chưa có checkpoint để reload.")
    ''')
    cell('markdown', '''
    ## 8. Đóng gói kết quả để tải về

    ZIP chỉ chứa thư mục output của lần chạy này. Không gom toàn bộ /kaggle/working,
    không kèm credentials, input Dataset hoặc cache tải model.
    Prepared data và prediction gold là dữ liệu dẫn xuất: tiếp tục giữ private.
    Checkpoint có thể lớn vì last.pt chứa optimizer. Tải về trước khi hết session;
    lưu một private Notebook Version để giữ output lâu dài.
    ''')
    cell('code', '''
    import shutil
    from IPython.display import FileLink, display
    archive = Path(shutil.make_archive(str(WORK / OUTPUT.name), "zip", root_dir=OUTPUT.parent, base_dir=OUTPUT.name))
    print("Output:", OUTPUT)
    print("ZIP:", archive, "| bytes:", archive.stat().st_size)
    print("SHA256:", pipeline.digest(archive))
    # Link tương đối để Jupyter/Kaggle phục vụ file từ working directory.
    os.chdir(WORK)
    display(FileLink(archive.name))
    ''')
    cell('markdown', '''
    ## Cách đọc kết quả và lượt chạy tiếp theo

    B0 tham chiếu (validation E): B≈0.670054; A≈0.745341. E càng thấp càng tốt,
    so B0/B1/B2 **trong cùng setting**, không dùng chênh lệch A/B như một hiệu ứng nhân quả.
    Đây là metric nội bộ, chưa xác minh là metric leaderboard.

    Sau prepare → smoke → train B1/B2 seed42 trên B → đọc lỗi validation.
    Sau đó chạy A và lặp seed13,2027 cho các so sánh quan trọng.
    Tuning chỉ trên validation: grid1e-5,2e-5,3e-5, tối đa3 trial/model/setting,
    lưu mọi trial và áp dụng budget công bằng. Không chỉ báo best seed.
    Chốt lựa chọn chung A/B trước khi mở bất kỳ final nào.

    Tài liệu chính thức được dùng để soạn hướng dẫn (đối chiếu17/10/2026):
    [Kaggle Datasets](https://www.kaggle.com/docs/datasets),
    [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks),
    [HF padding/truncation4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/pad_truncation),
    [Auto Classes4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/auto).
    ''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'},
                             'cosense': {'mode_default': 'prepare', 'embedded_sha256': hashes,
                                         'validation_status': 'local checks only; Kaggle GPU execution pending'}},
                'cells': cells}
    path = ROOT / 'notebooks/kaggle_en_nn.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Wrote {path.relative_to(ROOT)} ({len(cells)} cells)')
    return path


if __name__ == '__main__':
    build()
