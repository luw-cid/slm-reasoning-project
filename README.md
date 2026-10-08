# Đồ Án Tốt Nghiệp: Nâng Cao Năng Lực Suy Luận Của Small Language Models (SLM)

> **Mô hình nghiên cứu xuyên suốt**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
> **Môi trường thực nghiệm chính**: Google Colab (GPU T4 / A100 / L4) & Môi trường cục bộ  
> **Tập dữ liệu chuẩn**: GSM8K (100 câu kiểm thử cố định + tập con 300–500 câu huấn luyện)

---

## 📌 1. Bối Cảnh Và Mục Tiêu Đề Tài

Đề tài tốt nghiệp (6 tín chỉ) tập trung nghiên cứu, hiện thực và đánh giá các phương pháp cải thiện khả năng suy luận logic/toán học (reasoning) trên dòng mô hình ngôn ngữ nhỏ (Small Language Models - SLM).

Dựa trên taxonomy 5 nhánh phương pháp từ giảng viên hướng dẫn:
- **Nhánh A**: Supervised transfer từ mô hình mạnh hơn (Distillation).
- **Nhánh B**: Self-generated experience / Reinforcement Learning (RLVR).
- **Nhánh C**: Inference-time compute (Giám sát và can thiệp trong lúc sinh).
- **Nhánh D**: Architecture / Latent-space reasoning.
- **Nhánh E**: External knowledge / Tools / Ensemble.

Đồ án lựa chọn và thực hiện nghiên cứu chuyên sâu **2 phương pháp bổ trợ lẫn nhau**:
1. **Phương pháp 1 (Nhánh C - Inference-time)**: **RPDI-EE** (*Reasoning Path Deviation Index - Early Exit*). Giám sát độ bất định entropy để phát hiện suy luận lệch hướng và phanh sớm.
2. **Phương pháp 2 (Nhánh B - Training-time)**: **RLVR** (*Reinforcement Learning with Verifiable Rewards*). Sử dụng thuật toán **GRPO** kết hợp **LoRA** để mô hình tự học kinh nghiệm từ phần thưởng đúng/sai.
3. **Thực nghiệm kết hợp (Combined Evaluation)**: Kiểm chứng 4 kịch bản trên cùng 100 câu test GSM8K:
   - Kịch bản 1: **Baseline gốc** (Không can thiệp).
   - Kịch bản 2: **Baseline + RPDI-EE** (Phanh suy luận từ bên ngoài).
   - Kịch bản 3: **Mô hình sau RLVR** (Tự điều chỉnh suy luận từ bên trong).
   - Kịch bản 4: **Mô hình sau RLVR + RPDI-EE** (Kết hợp cả 2 phương pháp).

---

## 📂 2. Cấu Trúc Thư Mục Dự Án

```
slm-reasoning-project/
├── common/                  # Thư viện tiện ích nền tảng dùng chung cho toàn bộ dự án
│   ├── __init__.py          # Đóng gói package common
│   ├── seed.py              # Đảm bảo tính tất định và tái lập (reproducibility)
│   ├── eval_dataset.py      # Trình nạp dữ liệu kiểm thử GSM8K ngoại tuyến (offline)
│   ├── model_loader.py      # Tự động nạp mô hình DeepSeek-R1-1.5B tối ưu GPU/CPU
│   ├── metrics.py           # Bộ trích xuất đáp án cuối và đo lường đa chỉ số
│   └── data/
│       └── gsm8k_test_100.json # 100 mẫu kiểm thử GSM8K cố định kèm đáp án chuẩn
│
├── method_c_rpdi_ee/        # Phương pháp 1: RPDI-EE (Inference-time Compute)
│   ├── __init__.py          # Export class RPDILogitsProcessor và các hàm runner
│   ├── logits_processor.py  # Thuật toán giám sát entropy O(1) và ép đóng </think>
│   ├── runner.py            # Bộ điều khiển sinh so sánh Standard vs RPDI-EE
│   └── evaluate.py          # Script thực nghiệm đối chứng đa chiều trên 100 câu GSM8K
│
├── method_b_rlvr/           # Phương pháp 2: RLVR qua GRPO + LoRA (Đang chuẩn bị code)
│   ├── prepare_train_data.py # Trích xuất 300-500 câu GSM8K train (Zero Data Leakage)
│   ├── reward_funcs.py      # Hệ hàm phần thưởng: Accuracy Reward + Format Reward
│   ├── train_grpo.py        # Huấn luyện GRPO kết hợp LoRA bằng thư viện TRL
│   └── evaluate.py          # Đánh giá mô hình sau khi qua huấn luyện RLVR
│
├── combined/                # Thử nghiệm kết hợp 4 kịch bản đối chứng
│   └── run_4_scenarios.py   # Chạy và tổng hợp so sánh 4 kịch bản
│
├── results/                 # Lưu trữ dữ liệu số liệu thực nghiệm (JSON, CSV, Log)
│   └── method_c_results.json # Kết quả thực nghiệm đối chứng RPDI-EE
│
├── report/                  # Tài liệu báo cáo, bảng biểu và slide thuyết minh đồ án
├── fine_tune_deepseek_r1.ipynb # Notebook lưu trữ các vòng thử nghiệm kỹ thuật ban đầu
├── requirements.txt         # Danh sách thư viện cài đặt 1 lệnh trên Colab/Local
└── README.md                # Tài liệu tổng quan dự án
```

---

## 🔍 3. Vai Trò Và Tác Dụng Chi Tiết Từng Thành Phần

### 3.1. Thư mục `common/` (Tiện ích dùng chung)
Đảm bảo tính nhất quán (consistency) và chuẩn hóa khoa học giữa mọi phương pháp:
* **[seed.py](file:///b:/GIT/slm-reasoning-project/common/seed.py)**: Khóa seed tất định cho Python `random`, `numpy`, và `torch` (cả CPU lẫn cuDNN). Chứa hàm `seed_worker` bảo vệ DataLoader không bị trùng lặp seed giữa các tiến trình con.
* **[data/gsm8k_test_100.json](file:///b:/GIT/slm-reasoning-project/common/data/gsm8k_test_100.json)**: Tập hợp đúng 100 câu hỏi GSM8K cố định (kèm ID, đề bài, lời giải đầy đủ và `ground_truth` số nguyên bản), giúp chạy thực nghiệm hoàn toàn ngoại tuyến không phụ thuộc mạng.
* **[eval_dataset.py](file:///b:/GIT/slm-reasoning-project/common/eval_dataset.py)**: Cung cấp hàm `load_gsm8k_test(limit=...)` và `get_sample_by_id(id)` để lấy các mẫu dữ liệu kiểm thử.
* **[model_loader.py](file:///b:/GIT/slm-reasoning-project/common/model_loader.py)**: Tự động phát hiện năng lực phần cứng GPU để chọn độ chính xác tối ưu (`bfloat16` trên Ampere/Hopper hoặc `float16` trên T4), chuẩn hóa định dạng prompt tư duy:
  ```text
  User: {question}
  Assistant: <think>
  ```
* **[metrics.py](file:///b:/GIT/slm-reasoning-project/common/metrics.py)**: 
  - Hàm `normalize_numeric_str`: Xử lý triệt để các ký tự định dạng và escape của LaTeX (`\text{}`, `\$`, `\!`, dấu phẩy, khoảng trắng).
  - Hàm `extract_answer_from_response`: Trích xuất đáp số cuối theo 4 cấp độ ưu tiên (`\boxed{}`, `#### <số>`, mẫu câu kết luận, số cuối cùng).
  - Hàm `split_think_and_answer`: Tách riêng chuỗi tư duy `<think>...</think>` và câu trả lời.
  - Hàm `compute_aggregate_metrics`: Tính toán Độ chính xác (%), Thời gian (s), Tốc độ (TPS), Token suy nghĩ và Token trả lời.

---

### 3.2. Thư mục `method_c_rpdi_ee/` (Phương pháp RPDI-EE)
Hiện thực hóa thuật toán phanh suy luận sớm dựa trên chỉ số lệch hướng tư duy (**RPDI**):
$$\text{RPDI} = \frac{\text{LTF}}{\text{GTF}} = \frac{\frac{1}{W}\sum_{i=t-W+1}^{t} H(t_i)}{\frac{1}{t}\sum_{i=1}^{t} H(t_i)}$$

* **Quá trình tiến hóa kỹ thuật**:
  - **V1 (Vòng lặp Python thủ công)**: Quá chậm do overhead CPU-GPU (25.74s vs 4.97s).
  - **V2 (StoppingCriteria + 2 lệnh generate)**: Vẫn chậm vì mất KV Cache giữa pha suy nghĩ và pha trả lời (22.53s vs 21.60s).
  - **V3 (LogitsProcessor trong 1 lệnh generate duy nhất)**: Giữ nguyên 100% KV Cache, lọc token biên câu (`.`, `\n`, `;`, `?`, `!`, `,`) trực tiếp bằng PyTorch GPU tensor. Nhanh hơn Standard rõ rệt.
* **Các file thành phần**:
  - **[logits_processor.py](file:///b:/GIT/slm-reasoning-project/method_c_rpdi_ee/logits_processor.py)**: Lớp `RPDILogitsProcessor` kế thừa `transformers.LogitsProcessor`. Khi phát hiện $\text{RPDI} > \lambda$, cưỡng bức sinh chuỗi `</think>\n` ngay lập tức để chuyển sang pha kết luận.
  - **[runner.py](file:///b:/GIT/slm-reasoning-project/method_c_rpdi_ee/runner.py)**: Hàm `generate_standard()` và `generate_rpdi()`. Hỗ trợ **Greedy Decoding (`do_sample=False`)** để loại bỏ hoàn toàn nhiễu ngẫu nhiên.
  - **[evaluate.py](file:///b:/GIT/slm-reasoning-project/method_c_rpdi_ee/evaluate.py)**: Script dòng lệnh đánh giá so sánh tự động, xuất bảng so sánh Markdown và lưu chi tiết ra JSON.

---

### 3.3. Thư mục `method_b_rlvr/` (Phương pháp RLVR)
Tập trung vào nhánh huấn luyện học tăng cường từ kinh nghiệm tự sinh:
* **Thuật toán cốt lõi**: **GRPO** (*Group Relative Policy Optimization*), thuật toán RL không cần Value Model (tiết kiệm bộ nhớ vượt trội so với PPO), kết hợp **LoRA** (*Low-Rank Adaptation*).
* **Mục tiêu nghiên cứu**: Khảo sát xem việc huấn luyện RLVR với phần thưởng đúng/sai (Verifiable Reward) có giúp mô hình tự điều chỉnh làm ngắn chuỗi suy luận lại hay không (so sánh với việc bị ép cắt từ bên ngoài của RPDI-EE).

---

### 3.4. Thư mục `combined/` (Kết hợp 4 kịch bản)
Nơi chạy thực nghiệm tổng hợp cuối cùng để đưa số liệu vào báo cáo luận văn tốt nghiệp:
1. `Baseline` (Mô hình gốc 1.5B chưa can thiệp).
2. `Baseline + RPDI-EE` (Cắt suy luận lúc suy luận).
3. `RLVR` (Mô hình đã fine-tune GRPO).
4. `RLVR + RPDI-EE` (Mô hình sau RLVR được trang bị thêm bộ phanh RPDI-EE).

---

## 🚀 4. Hướng Dẫn Sử Dụng Và Chạy Thực Nghiệm (Quick Start)

### 4.1. Cài đặt môi trường (Google Colab hoặc Local)
```bash
git clone <repo_url>
cd slm-reasoning-project
pip install -r requirements.txt
```

### 4.2. Chạy đánh giá Phương pháp RPDI-EE (Method C)
Chạy thử nghiệm trên tập GSM8K với giải mã **Greedy (`do_sample=False`)**:

```bash
# 1. Chạy test nhanh trên 5 câu đầu tiên
python method_c_rpdi_ee/evaluate.py --limit 5

# 2. Chạy thực nghiệm với cấu hình tối ưu (W=8, lambda=1.2, min_steps=64)
python method_c_rpdi_ee/evaluate.py \
    --W 8 \
    --lambda_th 1.2 \
    --min_steps 64 \
    --output_file results/method_c_greedy_W8_min64.json

# 3. Chạy thực nghiệm đối chứng theo bài báo arXiv:2603.14251 (W=512, lambda=2.0)
python method_c_rpdi_ee/evaluate.py \
    --W 512 \
    --lambda_th 2.0 \
    --output_file results/method_c_greedy_paper_W512.json
```
### 4.3. Chạy Ablation Study quét lưới (W × λ) & Dựng đường biên Pareto
Để chọn siêu tham số tối ưu dựa trên số liệu thực nghiệm thay vì cảm tính:

```bash
# Chạy toàn bộ pipeline 3 bước (test nhanh 10 mẫu hoặc bỏ --limit để chạy full)
python method_c_rpdi_ee/ablation/run_ablation_pipeline.py --limit 10

# Hoặc chạy từng bước độc lập:
# Bước 1: Ghi vết Baseline (Greedy + Entropy)
python method_c_rpdi_ee/ablation/step1_trace_baseline.py --limit 10

# Bước 2: Mô phỏng CPU offline 16 cấu hình (W x lambda) + Sinh tiếp phần trả lời trên GPU
python method_c_rpdi_ee/ablation/step2_ablate_grid.py

# Bước 3: Phân tích đường biên Pareto và xuất đồ thị PNG/PDF
python method_c_rpdi_ee/ablation/step3_plot_pareto.py
```

---

## 📊 5. Phát Hiện Thực Nghiệm Đáng Chú Ý (Key Findings)

Qua thực nghiệm kiểm chứng đối chứng ban đầu trên mô hình `DeepSeek-R1-Distill-Qwen-1.5B`:

1. **Hiện tượng Cạn kiệt Ngân sách Token (Token Exhaustion) ở Standard**:
   - Ở chế độ Standard, có tới **45% số câu hỏi** mô hình sinh phần `<think>` kéo dài đến tận kịch trần 511/512 tokens mà **không kịp đóng thẻ `</think>`**.
   - Do dùng chung ngân sách `max_new_tokens=512`, mô hình cạn kiệt token trước khi kịp viết câu trả lời cuối cùng, dẫn đến câu trả lời bị cắt cụt và bị tính là sai.
   - **RPDI-EE đã giải quyết triệt để lỗi này**: Khi phát hiện lệch hướng, phanh ép đóng thẻ sớm, chừa lại hơn 400 token cho pha trả lời để kết luận chính xác.
2. **Hiệu ứng biên của `min_steps=64` và Chuyển dịch sang Quét Lưới Hệ thống (Ablation Study)**:
   - Trong cấu hình thử nghiệm ban đầu ($W=8, \lambda=1.2$), do cửa sổ quá hẹp và ngưỡng quá nhạy, cấm thoát trước 64 bước (`min_steps=64`) được đưa vào nhằm chống phanh non.
   - Tuy nhiên, phân tích dữ liệu cho thấy `min(exit_step) = 64` và có tới 17 câu thoát ngay trong khoảng 64–70 bước. Điều này chứng minh nhiều câu đã thỏa điều kiện dừng từ trước và bị dồn lại thoát ngay khi `min_steps` vừa hết.
   - **Giải pháp khoa học chuẩn xác**: Bỏ rào cản nhân tạo `min_steps` (mặc định cho phép giám sát ngay khi $i \ge W$ theo bài báo), mở rộng không gian tìm kiếm sang $W \in \{16, 32, 64, 128\} \times \lambda \in \{1.5, 2.0, 2.5, 3.0\}$, sau đó dựng đường biên **Pareto Frontier (Accuracy vs. Average Tokens)** để chọn điểm cân bằng tối ưu ("Sweet Spot") dựa trên số liệu khách quan.

---

## 📜 6. Liên Hệ Học Thuật & Bài Báo Tham Chiếu
- DeepSeek-AI: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025).
- arXiv:2603.14251: *Mitigating Overthinking in LRLMs via Reasoning Path Deviation Monitoring* (2026).