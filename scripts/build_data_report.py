"""Render Vietnamese Markdown from saved audit, split and B0 artifacts."""
from data_utils import *

def f(v):return 'NA' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)

def main():
    cfg=config(); nn=read_json(ROOT/'reports/audit/en_nn_summary.json'); pv=read_json(ROOT/'reports/audit/en_pv_summary.json')
    manifest=read_json(ROOT/'reports/audit/manifest.json'); rr={b:load(b,cfg)[0] for b in cfg['sources']}
    splits={s:read_json(ROOT/f'reports/audit/split_validation_{s}.json') for s in ['seen_context','unseen_compound']}
    baseline={s:read_json(ROOT/f'results/b0/{s}/metrics.json') for s in splits}
    assert all(s['passed'] for s in splits.values()) and not manifest['fatal_issues']
    text=['# CoSense — Báo cáo dữ liệu tiếng Anh và split nghiên cứu',
      '\nNgày khảo sát: 14/10/2026. Sinh từ script `scripts/build_data_report.py`, dựa trên artifact đã chạy thật.',
      '\n## 1. Tóm tắt và mức sẵn sàng',
      f'\nEN-NN có **{nn["records"]:,} mẫu / {nn["unique_expressions"]} cụm**; EN-PV có **{pv["records"]:,} mẫu / {pv["unique_expressions"]} cụm**. Không có lỗi parse/schema, ô thiếu, ID lặp, mẫu trùng toàn bộ hoặc nhãn không hợp lệ. Có câu dùng chung giữa các mục tiêu: bốn nhóm EN-NN và một nhóm EN-PV, đã giữ lại vì là các mục tiêu khác nhau.',
      '\nHai split EN-NN đã được tạo và kiểm tra; B0 đã chạy trên **validation**, chưa tạo prediction/metric final. Có thể chuẩn bị notebook Kaggle B1/B2 với input Compound hoặc Compound+Context. Span heuristic còn cần rà soát nếu muốn dùng target markers/span pooling; chưa đủ để triển khai phương pháp cần span tự động đáng tin cậy.',
      '\n**Quan sát:** nhãn EN-NN thiên về điểm cao và có biến thiên trong cùng cụm. **Giả thuyết cần kiểm chứng:** context có thể giúp dự đoán cách dùng cụ thể. **Đề xuất:** so B1/B2 theo [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), không kết luận mô hình hiểu nghĩa từ EDA.',
      '\n## 2. Nguồn dữ liệu và từ điển cột',
      '\nChỉ đọc `data/train/en-nn-train.tsv` và `data/train/en-pv-train.tsv`. Hai file tiếng Đức hiện có được liệt kê trong manifest nhưng không nạp. Không tìm thấy bản train tiếng Anh thứ hai hay trial/ZIP riêng trong cây khảo sát. Theo người dùng, trial đã gộp; chưa có manifest gộp để xác minh từng dòng. Không nạp lại trial, không xem trial là evaluation độc lập. Giữ raw tại vị trí hiện có; mọi flag, chuẩn hóa và membership đều lưu riêng.',
      '\nMột **mẫu** là một mục tiêu trong một context; số mẫu không phải số cụm. **Schema** là danh sách trường của file. **Metadata** phục vụ truy vết/phân tích, không tự trở thành feature. Std là độ phân tán đánh giá của người, không phải độ tự tin mô hình. Không có năm/nguồn corpus/thời kỳ; không suy nhãn lịch sử/hiện đại từ ID hoặc văn phong.',
      '\n| Nhánh | File | Byte | SHA-256 |\n| --- | --- | ---: | --- |']
    for b,m in manifest['sources'].items():text.append(f'| {b} | `{m["path"]}` | {m["bytes"]} | `{m["sha256"]}` |')
    meanings={'ContextID':'ID dùng nối dòng nguồn và prediction','Compound':'Danh từ ghép mục tiêu','Mod':'Modifier, thành phần bổ nghĩa','Head':'Head, thành phần trung tâm',
      'ModAvg':'Điểm trung bình cho modifier','HeadAvg':'Điểm trung bình cho head','ModStd':'Độ phân tán đánh giá modifier','HeadStd':'Độ phân tán đánh giá head',
      'Context':'Câu sử dụng mục tiêu','ParticleVerb':'Động từ-tiểu từ mục tiêu','Base':'Động từ cơ sở','Particle':'Tiểu từ','Avg':'Điểm trung bình PV','Std':'Độ phân tán đánh giá PV'}
    text+=['\n| Nhánh | Cột | Ý nghĩa | Ví dụ thật từ dòng đầu | Vai trò |','| --- | --- | --- | --- | --- |']
    for b,s in [('en_nn',nn),('en_pv',pv)]:
        for c in s['schema']:
            role='target' if c in labels(b) else 'input' if c in ['Compound','Context','ParticleVerb'] else 'metadata (không vào B1/B2)'
            value=rr[b][0][c].replace('|','\\|')
            text.append(f'| {b} | `{c}` | {meanings[c]} | {value} | {role} |')
    text+=['\nMod/Head và Base/Particle được audit để hiểu cấu trúc; protocol hiện tại B1 **chỉ đọc Compound**, B2 **Compound + Context**, không bổ sung Mod/Head riêng. Không có gold/Std/ID trong input mô hình.',
      '\n## 3. Thống kê tổng quan',
      '\n| Nhánh | Mẫu / ID duy nhất | Cụm | Thành phần duy nhất | Context/cụm min / median / mean / max |',
      '| --- | --- | ---: | --- | --- |']
    for b,s in [('EN-NN',nn),('EN-PV',pv)]:
        v=s['contexts_per_expression']; text.append(f'| {b} | {s["records"]} / {s["unique_ids"]} | {s["unique_expressions"]} | {s["unique_components"]} | {v["min"]} / {v["median"]} / {v["mean"]:.3f} / {v["max"]} |')
    text+=['\n| Nhánh/cột | Min | Q05 | Q25 | Median | Mean | SD | Q75 | Q95 | Max |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for b,s in [('NN',nn),('PV',pv)]:
        for c,v in s['numeric'].items():
            q=v['quantiles']; vals=[v['min'],q['0.05'],q['0.25'],v['median'],v['mean'],v['std_population'],q['0.75'],q['0.95'],v['max']]
            text.append('| '+b+'/'+c+' | '+' | '.join(f(x) for x in vals)+' |')
    text+=['\nSD ở bảng là độ lệch chuẩn population **giữa các dòng của cột**, khác với giá trị Std trong từng dòng. Quantile dùng nội suy tuyến tính tại vị trí `(n−1)×p`. Không có Avg ngoài [0,5], Std âm hoặc số không hữu hạn.',
      '\n| Nhánh | Độ dài | Min | Median | Mean | Max |','| --- | --- | ---: | ---: | ---: | ---: |']
    for b,s in [('EN-NN',nn),('EN-PV',pv)]:
        for k,label in [('sentence_words','token regex'),('sentence_characters','ký tự Unicode')]:
            v=s[k];text.append(f'| {b} | {label} | {v["min"]} | {v["median"]} | {v["mean"]:.2f} | {v["max"]} |')
    text+=['\nĐếm “từ” ở audit bằng regex Unicode `\\w+`: tách dấu nối/dấu nháy, khác cách đếm whitespace trong khảo sát ban đầu và khác subword tokenizer. Chưa tải tokenizer nên chưa kết luận max length 256 giữ hết mọi câu.',
      '\n## 4. Trùng và gần trùng',
      '\n| Loại | EN-NN: nhóm / dòng dư | EN-PV: nhóm / dòng dư |','| --- | --- | --- |']
    for k in nn['duplicates']:
        v=nn['duplicates'][k];w=pv['duplicates'][k];text.append(f'| {k} | {v["groups"]} / {v["extra_records"]} | {w["groups"]} / {w["extra_records"]} |')
    text+=['\n“Dòng dư” chỉ là số đếm n−1 trong mỗi nhóm, **không phải yêu cầu xóa**. Bốn cặp NN: `11382/13476`, `13134/13458`, `13142/13457`, `13316/13333`. Cặp PV: `30312/31455`. Không có cùng Compound+context với nhãn xung đột.',
      f'\nNear-duplicate là **ứng viên gần trùng**, không phải xác nhận cùng nghĩa. Dùng hai chỉ mục độc lập, token-set Jaccard ≥{cfg["near_duplicate"]["token_jaccard"]} **hoặc** character-5gram Jaccard ≥{cfg["near_duplicate"]["char_5gram_jaccard"]}; bỏ cặp context chuẩn hóa bằng nhau vì đã audit riêng. Tìm được {nn["near_duplicate_candidates"]} ứng viên NN và {pv["near_duplicate_candidates"]} ứng viên PV. File candidate vẫn được xuất với header dù rỗng. Thuật toán bao phủ cặp đạt hai ngưỡng lexical này nhưng không bảo đảm phát hiện paraphrase/ngữ nghĩa tương đương.',
      '\nPolicy bảo thủ: mọi ứng viên phát hiện được nằm cùng group trước split, vẫn giữ trạng thái chưa xác nhận ngữ nghĩa. Trong lần chạy này không có ứng viên cần giải quyết; **không tuyên bố loại hết rò rỉ near-duplicate**. Group context lớn nhất là 2 dòng; group B nối thêm Compound lớn nhất 20 dòng/2 cụm, do câu dùng chung; không có chuỗi near-duplicate tạo group lớn.',
      '\n## 5. Mục tiêu xuất hiện trong câu',
      '\n| Loại theo thuật toán | EN-NN | EN-PV |','| --- | ---: | ---: |']
    kinds=sorted(set(nn['alignment_counts'])|set(pv['alignment_counts']))
    for k in kinds:text.append(f'| {k} | {nn["alignment_counts"].get(k,0)} | {pv["alignment_counts"].get(k,0)} |')
    text+=['\nExact và case-insensitive cần biên từ, trả offset chỉ khi đúng một vị trí surface. “Inflection or punctuation candidate” dùng quy tắc số nhiều/đuôi động từ và danh sách bất quy tắc nhỏ; **không phải lemmatizer hoặc xác nhận hình thái**. Heuristic không gán selected span cho variant, nhiều ứng viên hoặc thành phần rời. PV xét base biến hình đứng trước particle trong khoảng ≤32 token regex; có thể nhầm quan hệ cú pháp hoặc bỏ lỡ thứ tự khác.',
      '\n`unresolved` nghĩa thuật toán chưa xác định được, không phải cụm vắng mặt. Các mẫu variant/ambiguous/discontinuous/unresolved vẫn eligible cho Compound+Context; span review bắt buộc nếu chọn phương pháp cần span. Bảng đầy đủ có cả context gốc/chuẩn hóa và candidate offsets trong `reports/audit/*_manual_alignment.csv`.',
      '\n### Ví dụ thật, chọn theo ID định trước',
      '\n| Nhánh / ID | Mục tiêu | Điểm thật | Context gốc |','| --- | --- | --- | --- |']
    examples={'en_nn':['10001','11382','13476'],'en_pv':['30002','30020','30312','31455']}
    for b,ids in examples.items():
        lookup={r['ContextID']:r for r in rr[b]}
        for id_ in ids:
            r=lookup[id_];text.append(f'| {b}/{id_} | {r[target(b)]} | '+', '.join(c+'='+r[c] for c in labels(b))+' | '+r['Context'].replace('|','\\|')+' |')
    text+=['\n## 6. Các hình và cách đọc',
      '\n### 6.1 Nhãn modifier/head', '\n![Phân bố NN](../reports/figures/nn_scores.png)',
      '\nHai cột dùng cùng bin rộng 0,25 điểm; màu xanh là modifier, cam là head. Nhãn tập trung về phía cao nên baseline trung bình là mốc cần thiết; biểu đồ chỉ mô tả gold, chưa nói mô hình tốt hay context hữu ích.',
      '\n### 6.2 Nhãn PV', '\n![Phân bố PV](../reports/figures/pv_scores.png)',
      '\nPV dùng cùng thang 0–5 và bin 0,25 như NN, nhưng phân bố khác. Vì khác target và dữ liệu, không dùng trực tiếp điểm B0 NN làm chuẩn cho PV; hiện chưa split hoặc chạy B0 PV.',
      '\n### 6.3 Số context mỗi cụm', '\n![Số context](../reports/figures/context_counts.png)',
      '\nNN có nhiều cụm 5 hoặc 10 câu, PV phần lớn 10 câu. Cụm `viscosity work` chỉ có một mẫu, được train-only ở A; số câu không chứng minh số nghĩa hay số quan sát độc lập.',
      '\n### 6.4 Độ dài', '\n![Độ dài](../reports/figures/sentence_lengths.png)',
      '\nHai panel đo token regex và ký tự Unicode, thể hiện chi phí input khác nhau. Cần đo lại bằng tokenizer Kaggle trước truncation; dữ liệu ngắn theo từ vẫn có thể dài theo subword.',
      '\n### 6.5 Quan hệ modifier/head', '\n![Mod và head](../reports/figures/nn_mod_head.png)',
      f'\nĐường chéo là hai điểm bằng nhau, độ đậm ô là số mẫu. Có {nn["nn_gap"]["absolute_ge_2"]}/{nn["records"]} mẫu với gap tuyệt đối ≥2, đủ lý do báo riêng hai output; chưa chứng minh hai đầu ra model sẽ phân biệt được.',
      '\n### 6.6 Biến thiên nội cụm', '\n![Biến thiên theo context](../reports/figures/nn_context_variation.png)',
      '\nChọn sáu Compound đầu theo thứ tự alphabet có ≥5 context, không chọn theo điểm/lỗi. Trục ngang là thứ tự ID, **không phải thời gian**; đường nối chỉ giúp nhìn điểm giữa câu, không chứng minh chuyển nghĩa hoặc xu hướng lịch sử.',
      '\n## 7. Phân tích EN-NN phục vụ nghiên cứu',
      f'\nTrung bình ModAvg−HeadAvg = {nn["nn_gap"]["signed"]["mean"]:.4f}; mean gap tuyệt đối = {nn["nn_gap"]["absolute"]["mean"]:.4f}. Đây là mô tả annotation, không phải độ khó hoặc hiệu quả mô hình.',
      '\n| Output | Tổng variance | Trong cụm | Giữa cụm | Tỷ lệ trong cụm |','| --- | ---: | ---: | ---: | ---: |']
    for c,v in nn['variance_decomposition'].items():text.append(f'| {c} | {v["total_population"]:.4f} | {v["within"]:.4f} | {v["between"]:.4f} | {v["within_fraction"]:.1%} |')
    text+=['\nVariance phân rã có trọng số theo số câu: tổng = mean bình phương độ lệch trong cụm + variance mean cụm theo trọng số. Khác biệt giữa các cụm chiếm phần lớn biến thiên, nhưng biến thiên nội cụm vẫn đáng phân tích; nó có thể gồm nghĩa, nguồn/chủ đề và bất đồng annotation, không chỉ context gây đổi nghĩa.',
      '\n| Output | Pearson độ dài–điểm | Pearson Std–điểm | Pearson độ dài–Std |','| --- | ---: | ---: | ---: |']
    for c,v in nn['descriptive_correlations'].items():text.append('| '+c+' | '+' | '.join(f(v[k]) for k in ['length_score_pearson','std_score_pearson','length_std_pearson'])+' |')
    text+=['\nĐộ dài liên hệ tuyến tính rất yếu với điểm trong dữ liệu này; Std và điểm có tương quan âm. Thang bị chặn 0–5 và cách phân phối rating có thể góp phần tạo quan hệ này; không kết luận nhân quả và không đưa Std vào model. Không tính reliability/CI của annotation từ Std khi thiếu raw rating/số người.',
      '\n## 8. Policy eligible và kết quả split',
      '\nGiữ toàn bộ 3.490 NN và 1.567 PV eligible; không loại vì alignment heuristic chưa chắc. Không chỉnh điểm, không lấy trung bình duplicate nhãn. Lỗi schema/parse/ID/nhãn nghiêm trọng sẽ ngăn split và B0; audit flags ghi từng row key, eligible và exclusion reason. Không có mẫu bị loại trong lần chạy này.',
      '\nA=`seen_context`: group context trùng/gần trùng bảo thủ, đưa đủ train coverage trước rồi cân bằng số dòng về 70/15/15. Cụm có ≤2 context group là train-only. B=`unseen_compound`: nối cả Compound chuẩn hóa và context group; giữ nguyên connected component. Cả hai dùng seed 42, tie order giả ngẫu nhiên, không dùng nhãn/metric chọn seed. Tái tạo được membership y hệt.',
      '\n| Thiết lập | Tập | Mẫu | Cụm | Tỷ lệ mẫu | Context/component group |','| --- | --- | ---: | ---: | ---: | ---: |']
    for setting,s in splits.items():
        for part,c in s['counts'].items():text.append(f'| {setting} | {part} | {c["records"]} | {c["compounds"]} | {c["fraction"]:.3%} | {c["context_groups"]} |')
    text+=['\nTỷ lệ xấp xỉ70/15/15, số thực tế ở bảng; lệch làm tròn vì record/group nguyên. A không ép mọi cụm có cả validation và final; mọi cụm được đánh giá có trong train. B không chung Compound nhưng được chung modifier/head; chưa phải held-out cả hai thành phần hoặc unseen pretrain.',
      '\n| Thiết lập/tập | Context/cụm: min / median / mean / max |','| --- | --- |']
    for setting,s in splits.items():
        for part,c in s['counts'].items():
            v=c['contexts_per_compound'];text.append(f'| {setting}/{part} | {v["min"]} / {v["median"]} / {v["mean"]:.3f} / {v["max"]} |')
    text+=[
      '\n| Thiết lập/tập | Mod mean / min / max | Head mean / min / max |','| --- | --- | --- |']
    for setting,s in splits.items():
        for part,c in s['counts'].items():
            vv=[c['labels'][k] for k in ['ModAvg','HeadAvg']];text.append(f'| {setting}/{part} | '+' | '.join(f'{v["mean"]:.4f} / {v["min"]} / {v["max"]}' for v in vv)+' |')
    a_final={r['row_key'] for r in read_members('seen_context','final')};b_train={r['row_key'] for r in read_members('unseen_compound','train')}
    b_final={r['row_key'] for r in read_members('unseen_compound','final')};a_train={r['row_key'] for r in read_members('seen_context','train')}
    text+=[f'\nCó {len(a_final&b_train)} mẫu final A nằm trong train B, và {len(b_final&a_train)} mẫu final B nằm trong train A. A/B là protocol tách biệt **nhưng không độc lập thống kê**. Khóa lựa chọn phương pháp cho cả hai trước khi xem bất kỳ final; không sửa method theo final A rồi gọi final B là untouched. Không quy chênh lệch MAE A/B hoàn toàn cho generalization.',
      '\n| Thiết lập | Assertion đạt | Trạng thái |','| --- | ---: | --- |']
    for setting,s in splits.items():text.append(f'| {setting} | {sum(c["passed"] for c in s["assertions"])} / {len(s["assertions"])} | PASS |')
    text+=['\nCác assertion gồm coverage, ID uniqueness/disjointness, source hash/schema/score, câu chuẩn hóa, A train coverage, B Compound disjointness, candidate near grouping và deterministic replay. Báo cáo JSON chứa kết quả từng assertion. Không còn candidate near-duplicate phát hiện được chưa được xử lý, nhưng paraphrase chưa phát hiện vẫn là giới hạn.',
      '\n## 9. B0 trên validation, chưa đánh giá final',
      '\nB0 tính hai mean chỉ từ train của từng setting, dự đoán cùng hai giá trị cho mọi mẫu validation. Đây là mốc tham chiếu đơn giản; model học cần chứng minh lợi ích vượt nó, không mặc định sẽ thắng.',
      '\n| Thiết lập | Train mean Mod / Head | Val MAE Mod | Val MAE Head | MAE hai đầu theo dòng | E theo cụm (primary) | RMSE Mod / Head |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for setting,b in baseline.items():
        m=b['metrics'];text.append(f'| {setting} | {b["train_means"]["ModAvg"]:.4f} / {b["train_means"]["HeadAvg"]:.4f} | {m["outputs"]["ModAvg"]["mae"]:.4f} | {m["outputs"]["HeadAvg"]["mae"]:.4f} | {m["row_macro_two_output_mae"]:.4f} | {m["compound_macro_two_output_mae"]:.4f} | {m["outputs"]["ModAvg"]["rmse"]:.4f} / {m["outputs"]["HeadAvg"]["rmse"]:.4f} |')
    text+=['\nPearson và Spearman là **NA** vì prediction hằng, không đổi thành 0. E là mean lỗi hai thành phần trong mỗi Compound rồi mean các Compound; MAE theo dòng là diagnostic để thấy khác biệt trọng số. Không có B0 prediction/metric final; phân bố nhãn final ở mục 8 chỉ phục vụ kiểm tra dữ liệu.',
      '\n## 10. Lệnh chạy và bước tiếp theo',
      '\nCore audit/split/B0 dùng Python ≥3.11 và thư viện chuẩn; biểu đồ cần Matplotlib. Không dùng torch, transformers hoặc tải backbone trong giai đoạn này. Matplotlib 3.11.2 đã được cài vào `.runtime/` riêng bằng Python 3.11.9; core scripts cũng được chạy dưới Python 3.14.6 trong sandbox. Không dùng binary wheel 3.11 với interpreter 3.14. Tám integrity tests đã chạy đạt, gồm quoted TSV, lỗi nhãn, word boundary/ambiguity, gần trùng, metric tính tay, membership bị sửa sai và CLI nonzero/B0 bị chặn.',
      '\nTừ repository root, trong môi trường Python tương thích:',
      '\n```powershell\npython -m pip install -r requirements-audit.txt\npython scripts/audit_data.py\npython scripts/create_splits.py\npython scripts/validate_splits.py\npython scripts/run_mean_baseline.py\npython scripts/build_data_report.py\npython -m unittest discover -s tests -v\n```',
      '\nNếu chỉ kiểm tra core không có Matplotlib: `python scripts/audit_data.py --no-figures`; sáu PNG cần chạy audit đầy đủ trước khi render report. Các script có `--config` cho config thay thế; báo cáo mặc định dùng config repository. `validate_splits.py` trả exit code khác 0 khi fail; B0 gọi validation lại trước khi chạy.',
      '\nTiếp theo: rà soát alignment nếu định dùng span; chuẩn bị notebook Kaggle đọc membership đã khóa và chạy smoke B1/B2. Không train lại split, không đưa ID/Avg/Std vào input. Chưa upload/commit dữ liệu dẫn xuất khi license chưa xác minh.',
      '\n## Phụ lục: thuật toán và artifact',
      '\nParser: `csv.DictReader`, UTF-8-sig, tab delimiter, quotechar dấu nháy kép, strict; newline do csv xử lý. Record index khác số dòng vật lý khi có newline trong quoted cell. Chuẩn hóa NFC/casefold/gộp whitespace, giữ punctuation và accent; không sửa context gốc.',
      '\nSplit: groups được sắp thứ tự gốc, shuffle bằng `random.Random(42)`, bảo đảm coverage A, rồi phân bổ theo thứ tự shuffle để giảm tổng squared row deficit. Phiên bản `seed42_shuffled_group_squared_row_deficit_v2`; config/seed/python/source hashes/file hashes trong manifest. Không stratify theo nhãn hoặc chọn seed theo model scores. Bản development v1 dùng largest-first làm train thiên về cụm nhiều context nên đã thay v2 trước B1/B2; quyết định dựa số context/cụm, không theo metric B0/final. Chỉ v2 là membership hiện hành.',
      '\nArtifact chính: `reports/audit/*_summary.json`, `*_flags.csv`, `*_duplicates.csv`, `*_near_candidates.csv/json`, `*_manual_alignment.csv`, `en_nn_compound_variation.csv`, `split_validation_*.json`; sáu PNG + manifest; `data/splits/en_nn/{seen_context,unseen_compound}/{train,validation,final}.csv`; `results/b0/*/{metrics.json,validation_predictions.tsv}`. Files dẫn xuất bị `.gitignore` chặn commit mặc định.',
      '\n[Trang BTC](https://sites.google.com/view/semeval-compartment) được đối chiếu ngày 14/10/2026. Liên kết [CodaBench 17971](https://www.codabench.org/competitions/17971/) trả nội dung không liên quan trong lần truy cập, nên metric/template/rules/license vẫn **chưa xác minh**. Không nhận metric nội bộ là leaderboard metric. Thống kê file source được kiểm tra hash trước/sau audit; không có sửa source.']
    (ROOT/'docs/DATA_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Created docs/DATA_REPORT.md from verified artifacts')

if __name__=='__main__':main()
