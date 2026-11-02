# RPDI-EE trên vLLM + chạy thử 200 câu — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cài đặt lại RPDI-EE đúng theo Algorithm 1 của bài gốc (Guan et al., NeurIPS 2026) trên vLLM. Kèm theo là bản sao chính xác cài đặt HF hiện tại, script phân tích, và notebook Colab chạy thử 200 câu (ids 100–299) ở ngân sách 16.384 token.

**Architecture:**
- **Lõi Python thuần** (`rpdi_core.py`, không dùng torch) giữ toàn bộ logic quyết định phanh. Nhờ đó test được trên máy không có GPU.
- **Lớp bọc mỏng** trong `run_vllm.py` tính entropy trên GPU rồi gọi lõi. Lớp này là *logits processor theo từng request* của vLLM V0.
- **Mỗi cặp (câu hỏi, phương pháp) chỉ sinh một lần ở 16k.** Kết quả ở mọi ngân sách B nhỏ hơn có được bằng cách cắt `output_ids[:B]`. Cách này hợp lệ vì giải mã greedy và quyết định phanh chỉ phụ thuộc phần tiền tố.

**Tech Stack:** Python 3.12, vLLM V0 (ghim `0.6.3.post1`; Python ≥ 3.13 dùng `0.8.5.post1` với `VLLM_USE_V1=0`), transformers (chỉ dùng tokenizer), pytest cho test CPU, Colab T4 fp16.

**Spec:** bài RPDI gốc (arXiv:2603.14251) gồm §4.1–4.3, Algorithm 1 và Appendix B. Các quyết định đã chốt trong hội thoại được ghi ở phần Global Constraints bên dưới.

## Global Constraints

- **Prompt:** giữ đúng `common.model_loader.format_reasoning_prompt`, tức `"User: {q}\nAssistant: <think>"`. Token hóa bằng `AutoTokenizer(prompt)["input_ids"]` (có BOS), giống hệt các lần chạy HF.
- **Giải mã:** greedy (`temperature=0`), `dtype=float16` trên T4, `max_tokens=16384` là tổng số token sinh ra (gồm cả pha suy nghĩ và pha trả lời, giống `max_new_tokens`).
- **Entropy:** Shannon trên toàn bộ vocab của logits thô (trước temperature), tính bằng float32, với quy ước `0·log0 = 0`.
- **Chế độ `paper`, bám đúng Algorithm 1:**
  - Tại bước i, sau khi đã có t_i, kiểm tra `i ≥ W và t_i ∈ B`.
  - Công thức: `GTF = S_global/i`, `LTF = S_local/W` (với S_local là tổng W entropy gần nhất), `RPDI = LTF/GTF`. Phanh khi `RPDI > λ`.
  - Kiểm tra được thực hiện *trước* khi nạp entropy của token kế tiếp.
  - Khi phanh: chỉ chèn đúng 1 token `</think>`, sau đó mô hình tự sinh câu trả lời.
- **Chế độ `legacy`, sao chép chính xác `method_c_rpdi_ee/logits_processor.py::RPDILogitsProcessor`:**
  - Entropy của phân phối kế tiếp được nạp *trước*, sau đó mới kiểm tra token biên ở token liền trước (`step ≥ W`).
  - Khi phanh: ép chuỗi `"</think>\n"`.
  - Bộ biên: dấu câu `{".", "\n", ";", "?", "!", ","}` cùng biến thể có khoảng trắng đứng trước.
- **Bộ biên B ở chế độ `paper`:** bài gốc chỉ ghi "giống hệt DEER", nên có hai cách hiểu và chạy cả hai.
  - `wait` (mặc định): token có `decode([id]).strip() == "Wait"`, tức id 14190 và 13824. Đây là điểm chuyển ý (ATP) của DEER. Khi phanh, `</think>` *thay thế* token "Wait"; cách này khớp Hình 4 của bài.
  - `newline`: mọi token có `"\n"` khi decode.
- **Ngân sách được suy ra:** `512,768,1024,1536,2048,3072,4096,8192,16384`.
- **Tiêu chí pipeline đạt:** trên 200 câu, accuracy của vLLM@768 lệch so với HF@768 không quá **2 điểm**, cho cả `standard` và `rpdi_legacy_W16_l1.5`.
- **Biên δ:** phải chốt với thầy *trước khi* xem bảng so sánh. Script bắt buộc truyền `--delta` mới chạy phần so sánh.
- **Định dạng kết quả:** cùng các khóa với `run_controls.py` (`sample_id, method, max_new_tokens, ground_truth, predicted_answer, is_correct, total_tokens, think_tokens, answer_tokens, early_exited, exit_step`).

## Review Focus

1. **Mô hình tự đóng `</think>` trước khi bị phanh:** từ đó trở đi lõi phải ngừng giám sát và không phanh nhầm trong pha trả lời. Test: `test_paper_natural_close_stops_monitoring`, `test_legacy_natural_close_stops_monitoring`.
2. **Chuỗi phanh nhiều token của chế độ legacy bị cắt ngang bởi ngân sách:** `early_exited` ở ngân sách B chỉ đúng khi `</think>` nằm trong B token đầu. Test: `test_expand_exit_after_budget_not_counted`.
3. **Output kết thúc bằng EOS trước B, hoặc dài đúng bằng B:** `hit_budget` phải giống quy ước `total_tokens >= B` của HF. Test: `test_expand_short_output_all_budgets`.
4. **Logit bằng `-inf`:** entropy không được thành NaN. Đây là đoạn torch duy nhất; được kiểm tra trong cell smoke test của notebook (in entropy đầu tiên của câu đầu tiên và assert là số hữu hạn).
5. **Hai mảng kết quả giống hệt nhau, hoặc n nhỏ:** bootstrap phải cho CI là `(0, 0)` và kết luận là "tương đương" chứ không phải lỗi. Test: `test_bootstrap_identical_is_zero`.

---

### Task 1: Lõi RPDI thuần Python

**Files:**
- Modify: `method_c_rpdi_ee/__init__.py`. Chuyển sang import lười bằng `__getattr__` theo PEP 562, để `method_c_rpdi_ee.vllm_engine.*` import được trên máy không có torch. Các import cũ phải tiếp tục hoạt động.
- Create: `method_c_rpdi_ee/vllm_engine/__init__.py` (rỗng), `method_c_rpdi_ee/vllm_engine/rpdi_core.py`
- Test: `tests/conftest.py` (thêm thư mục gốc repo vào `sys.path`), `tests/test_rpdi_core.py`

**Interfaces (Produces):**
- `build_boundary_ids(tokenizer, preset: str) -> frozenset[int]`, với preset ∈ `{"wait","newline","punct"}`. Tokenizer chỉ cần có `encode(text, add_special_tokens=False)`, `decode(ids)` và `__len__`.
- `MethodSpec(key, kind, W, lam, boundary)`: dataclass frozen. `kind` ∈ `{"standard","rpdi","rpdi_legacy"}`.
- `parse_method(spec: str) -> MethodSpec`:
  - `"standard"` → key `standard`.
  - `"rpdi:512:2.0[:newline]"` → key `rpdi_W512_l2.0_wait` (hoặc `rpdi_W512_l2.0_newline`).
  - `"rpdi_legacy:16:1.5"` → key `rpdi_legacy_W16_l1.5`, boundary `punct`.
  - Chuỗi sai → `ValueError`.
- `RPDIMonitor(W, lam: float|None, mode: "paper"|"legacy", boundary: str, boundary_ids, think_close_id, close_ids)`:
  - Phương thức `step(prompt_ids, output_ids, stats: Callable[[], tuple[float,int]]) -> int|None` trả về id token bị ép, hoặc `None` nếu giữ nguyên logits.
  - `stats()` trả về `(entropy, argmax)` của phân phối kế tiếp và chỉ được gọi khi đang giám sát.
  - Thuộc tính: `entropies: list[float]`, `early_exited`, `exit_step` (số token suy nghĩ trước `</think>` bị ép; -1 nếu không phanh), `rpdi_at_exit`, `closed`.
  - `lam=None` nghĩa là chỉ ghi entropy, không bao giờ phanh (dùng cho `standard`).
- `monitor_for(spec: MethodSpec, tokenizer, boundary_cache: dict) -> RPDIMonitor`.

- [ ] **Step 1: Viết test (thất bại)** trong `tests/test_rpdi_core.py`, dùng `FakeTokenizer` (mỗi ký tự là một token, cộng thêm vài token đặc biệt):
  - `test_parse_method_keys`: kiểm tra 3 dạng key ở trên; `"rpdi:16"` phải ném `ValueError`.
  - `test_paper_checks_before_appending`:
    - Với W=2, λ=1.5 và entropy `[1,1,5,5]`, biên ở token thứ 4: tại lần gọi thứ 5 thì `S_global=12, i=4, LTF=5, GTF=3, RPDI=1.667 > 1.5` → trả về `think_close_id`, `exit_step == 4`.
    - Với cùng chuỗi nhưng λ=1.7 thì không phanh.
  - `test_paper_wait_replaces_token`: ở chế độ `wait`, phanh khi `argmax ∈ B` (token sắp sinh), không xét token liền trước.
  - `test_paper_natural_close_stops_monitoring` và `test_legacy_natural_close_stops_monitoring`: sau khi `output_ids[-1] == think_close_id`, `step` luôn trả `None` và `stats` không còn bị gọi.
  - `test_legacy_matches_hf_semantics`:
    - Entropy được nạp trước, rồi xét biên ở token liền trước (lần gọi đầu xét `prompt_ids[-1]`). Đặt `step ≥ W`.
    - Khi phanh, trả về `close_ids[0]`, lần gọi kế trả `close_ids[1]`, lần sau nữa trả `None`.
    - `exit_step == step − 1`.
  - `test_lam_none_never_exits`.
  - `test_boundary_presets`: `wait` lấy đúng các id decode ra "Wait"; `newline` lấy mọi id chứa `\n`; `punct` gồm cả biến thể có khoảng trắng.
- [ ] **Step 2:** `python -m pytest tests/test_rpdi_core.py -q` → FAIL (chưa có module).
- [ ] **Step 3:** Cài đặt `rpdi_core.py` và sửa `__init__.py` lười. Tổng S_global và S_local được cộng dồn bằng float của Python, giống bản HF.
- [ ] **Step 4:** Chạy lại pytest → PASS.

### Task 2: Suy ra kết quả theo ngân sách

**Files:**
- Create: `method_c_rpdi_ee/vllm_engine/budgets.py`
- Test: `tests/test_budgets.py`

**Interfaces:**
- Consumes: `common.metrics.extract_answer_from_response`, `is_answer_correct`.
- Produces: `expand_to_budgets(raw: dict, decode: Callable[[list[int]], str], think_close_id: int, budgets: list[int]) -> list[dict]`.
  - `raw` có các trường: `sample_id, method, max_tokens, ground_truth, prompt_ids, output_ids, close_index (int|None), forced_close (bool), exit_step, rpdi_at_exit`.
  - Bỏ qua các B lớn hơn `raw["max_tokens"]`.
  - Mỗi bản ghi ra có các khóa của `run_controls.py`, cộng thêm `hit_budget` (`total_tokens >= B`), `response_sha1` (sha1 của `decode(output_ids[:B])`), `engine: "vllm"` và `source_max_tokens`.
  - Đáp án được trích từ `decode(prompt_ids + output_ids[:B])`, giống `run_controls.py`.

- [ ] **Step 1: Viết test (thất bại):**
  - `test_expand_truncates_and_scores`: đáp án `\boxed{7}` nằm sau `</think>` ở vị trí 6. Với B=4 phải sai và `think_tokens==4`; với B=12 phải đúng và `think_tokens==6`.
  - `test_expand_exit_after_budget_not_counted`: `forced_close` ở vị trí 6. Với B=4 thì `early_exited False` và `exit_step -1`; với B=8 thì `True`.
  - `test_expand_short_output_all_budgets`: output dài 5 token. Với B=5 thì `hit_budget True`; với B=8 thì `False` và `total_tokens 5`.
  - `test_expand_skips_budgets_above_run`.
- [ ] **Step 2:** pytest → FAIL.
- [ ] **Step 3:** Cài đặt.
- [ ] **Step 4:** pytest → PASS.

### Task 3: Runner vLLM (GPU)

**Files:**
- Create: `method_c_rpdi_ee/vllm_engine/run_vllm.py`

**Interfaces:**
- Consumes: Task 1 (`parse_method`, `monitor_for`) và Task 2 (`expand_to_budgets`).
- Produces: CLI ghi 3 file vào `--output_dir`: `raw.jsonl` (mỗi dòng một cặp câu × phương pháp, có `output_ids` và `entropies` của pha suy nghĩ làm tròn 4 chữ số), `budgets.jsonl` và `run_info.json` (phiên bản thư viện, tên GPU, tham số, thời gian chạy).
  - Tham số: `--start 100 --end 300 --methods ... --max_tokens 16384 --budgets ... --chunk_size 50 --max_num_seqs 128 --gpu_memory_utilization 0.90 --dtype float16 --output_dir`, cùng `--expand_only` (tạo lại `budgets.jsonl` từ `raw.jsonl` chỉ bằng tokenizer, không cần GPU).
  - Bỏ qua các cặp `(sample_id, method, max_tokens)` đã có trong `raw.jsonl`, để chạy tiếp được khi Colab ngắt.
- Logits processor dùng chữ ký 3 tham số `(prompt_ids, output_ids, logits)`. Entropy tính theo `-(where(p>0, p*logp, 0)).sum()` trên `logits.float()`. Entropy và argmax lấy về trong một lần `tolist()`.
- Đặt `os.environ.setdefault("VLLM_USE_V1", "0")` trước khi import vllm.
- Prompt truyền dưới dạng `{"prompt_token_ids": ids}`. Gửi theo từng chunk; ghi file và flush sau mỗi chunk.

- [ ] **Step 1:** Viết script.
- [ ] **Step 2:** Kiểm tra cục bộ: `python method_c_rpdi_ee/vllm_engine/run_vllm.py --help` chạy được dù không có vllm/torch (chỉ import các thư viện này bên trong hàm).
- [ ] **Step 3:** Kiểm tra trên GPU: notebook chạy smoke test (Task 5).

### Task 4: Phân tích (CPU)

**Files:**
- Create: `method_c_rpdi_ee/vllm_engine/analyze_budgets.py`
- Test: `tests/test_analyze_budgets.py`

**Interfaces:**
- Consumes: `wilson_ci` và `mcnemar_exact_p` từ `method_c_rpdi_ee/analyze_results.py`.
- Produces:
  - `paired_bootstrap_ci(a: list[int], b: list[int], n_boot: int, seed: int) -> dict` trả về `{"diff","ci90","ci95"}`, tính theo điểm phần trăm, phương pháp percentile.
  - `classify(ci90, ci95, delta) -> str` trả về một khóa, xét theo thứ tự. Nhãn tiếng Việt nằm ở `VERDICT_TEXT` và `VERDICT_SHORT`.
    1. `better` nếu `lo95 > 0`.
    2. `slightly_worse` nếu `hi95 < 0` và `lo95 > −δ`.
    3. `worse` nếu `hi95 < 0`.
    4. `equivalent` nếu `lo90 > −δ` và `hi90 < δ`.
    5. `noninferior` nếu `lo95 > −δ`.
    6. Còn lại: `inconclusive`.
  - CLI nhận `--budgets_file ... --hf_reference ... --validate vllm_key=hf_key ... --compare target@B:ref@B ... --delta δ --output_md --output_json`.
  - Báo cáo gồm:
    - Bảng accuracy và số token trung bình theo ngân sách.
    - Kiểm tra pipeline: accuracy hai bên, tỷ lệ đồng thuận đúng/sai, tỷ lệ trùng khớp văn bản theo `response_sha1`, ĐẠT/KHÔNG ĐẠT theo ngưỡng 2 điểm.
    - So sánh chính: Δ, CI90, CI95, McNemar, kết luận, tỷ lệ token, tỷ lệ phục hồi, và ngân sách nhỏ nhất để `standard` đạt bằng accuracy của target.
  - Thiếu `--delta` thì bỏ qua phần so sánh và in cảnh báo.

- [ ] **Step 1: Viết test (thất bại):**
  - `test_bootstrap_identical_is_zero`.
  - `test_bootstrap_deterministic`: cùng seed thì cùng kết quả.
  - `test_classify_cases`: mỗi nhánh kết luận có ít nhất một ca.
  - `test_compare_end_to_end`: dữ liệu giả 10 câu, kiểm tra số cặp bất đồng b/c và Δ.
- [ ] **Step 2:** pytest → FAIL.
- [ ] **Step 3:** Cài đặt.
- [ ] **Step 4:** pytest → PASS. Chạy thêm `--help`.

### Task 5: Notebook Colab chạy thử 200 câu

**Files:**
- Create: `rpdi_vllm_colab.ipynb` (ở thư mục gốc, giống các notebook hiện có)

Các cell theo thứ tự:
1. Kiểm tra GPU và nhận diện môi trường (Colab hay Kaggle) để chọn `OUT_DIR`.
2. Clone hoặc pull repo.
3. Cài vLLM theo bản đã ghim tùy phiên bản Python, rồi in phiên bản. Kernel không import torch nên không cần khởi động lại.
4. Gắn Drive. Khai báo `DELTA`, có chú thích phải chốt với thầy trước khi xem kết quả.
5. Smoke test: ids 100–102, `max_tokens=1024`, 3 phương pháp. Assert: đủ 9 dòng raw, entropy là số hữu hạn, `rpdi_legacy` có phanh ở ít nhất một câu hoặc in cảnh báo.
6. Chạy thử 200 câu bằng `nohup`, với 4 phương pháp: `standard, rpdi_legacy:16:1.5, rpdi:512:2.0, rpdi:512:2.0:newline`.
7. Theo dõi tiến độ: tail log và đếm số dòng trong raw.
8. Phân tích: `--validate standard=standard rpdi_legacy_W16_l1.5=rpdi_W16_l1.5 --compare rpdi_legacy_W16_l1.5@768:rpdi_W512_l2.0_wait@16384 rpdi_legacy_W16_l1.5@768:standard@16384`, rồi hiển thị file markdown.
9. Hướng dẫn những file cần gửi lại.

- [ ] **Step 1:** Sinh notebook bằng script trong scratchpad để JSON hợp lệ.
- [ ] **Step 2:** Kiểm tra bằng `python -c "import json; json.load(open('rpdi_vllm_colab.ipynb'))"` và đọc lại các lệnh trong từng cell.
- [ ] **Step 3:** Chạy toàn bộ test: `python -m pytest tests -q` → PASS.
