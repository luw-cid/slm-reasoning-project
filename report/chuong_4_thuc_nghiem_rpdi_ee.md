# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ CHUYÊN SÂU PHƯƠNG PHÁP RPDI-EE

---

## 4.1. Thiết Lập Thực Nghiệm (Experimental Setup)

### 4.1.1. Mô hình và Các tập dữ liệu nghiên cứu
* **Mô hình nghiên cứu cốt lõi**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — mô hình ngôn ngữ nhỏ (Small Language Model - SLM) với 1.5 tỷ tham số, được chưng cất tri thức suy luận chuỗi tư duy (Chain-of-Thought - CoT) từ mô hình nền tảng DeepSeek-R1 lớn.
* **Quy trình thực nghiệm 2 giai đoạn (Two-Stage Experimental Pipeline)**:
  1. **Giai đoạn 1 — Quét lưới siêu tham số (Ablation Study trên 100 câu)**: Sử dụng tập con 100 câu hỏi toán học độc lập từ **GSM8K** (*Grade School Math 8K*), lưu trữ ngoại tuyến tại [common/data/gsm8k_test_100.json](file:///b:/GIT/slm-reasoning-project/common/data/gsm8k_test_100.json). Tập này đóng vai trò tập thẩm định (Validation Subset) nhằm quét lưới 16 cấu hình RPDI và các baseline đối chứng. **Lưu ý**: 100 câu này chính là câu 0–99 của tập test GSM8K, đồng thời cũng là 100 câu đầu của lượt chạy 705 câu ở Giai đoạn 2. Vì vậy, kết quả đánh giá chính của chương được báo cáo trên **605 câu held-out (câu 100–704)**, là phần không tham gia vào việc chọn siêu tham số (Mục 4.6.4). Các thực nghiệm tiếp theo chọn siêu tham số trên tập validation 200 câu trích từ GSM8K train ([common/data/gsm8k_val_200.json](../common/data/gsm8k_val_200.json)).
  2. **Giai đoạn 2 — Kiểm chứng mở rộng quy mô lớn (Scale-Up Verification trên 705 câu)**: Sau khi tìm được cấu hình tối ưu $W^*=16, \lambda^*=1.5$, nghiên cứu triển khai đánh giá đối chứng toàn diện quy mô lớn trên tập GSM8K mở rộng. Tiến trình chạy thực tế thực hiện liên tục **705 câu hỏi độc lập** (chiếm **53.45%** toàn bộ tập kiểm thử 1,319 câu của GSM8K), tiêu tốn **11.66 giờ GPU**, dữ liệu log chi tiết được lưu trữ tại [log-705.txt](../log-705.txt) và trích xuất cấu trúc thành [results/method_c_results_705.json](file:///b:/GIT/slm-reasoning-project/results/method_c_results_705.json).
* **Môi trường phần cứng**: Thực nghiệm thực thi trên môi trường GPU NVIDIA, hỗ trợ định dạng số học dấu phẩy động `bfloat16`.

### 4.1.2. Chiến lược giải mã và Ngân sách Token
* **Chiến lược giải mã**: **Greedy Decoding (`do_sample=False`)**. Việc sử dụng giải mã tham lam là bắt buộc trong thực nghiệm khoa học nhằm loại bỏ phương sai ngẫu nhiên của quá trình sampling nhiệt độ, đảm bảo tính tất định và khả năng tái lập kết quả (reproducibility) 100%.
* **Ngân sách sinh tối đa (`max_new_tokens`)**: Thiết lập cố định ở mức **768 tokens** cho mỗi câu hỏi.

### 4.1.3. Hệ thống chỉ số đo lường (Evaluation Metrics)
1. **Độ chính xác (Accuracy - %)**: Tỷ lệ phần trăm câu hỏi mà mô hình trích xuất được đáp án số cuối cùng khớp chính xác với nhãn chuẩn `ground_truth`.
2. **Tổng số Token trung bình (`avg_total_tokens`)**: Đo lường tổng chi phí tính toán (Inference Cost) trên mỗi câu hỏi.
3. **Số Token suy nghĩ trung bình (`avg_think_tokens`)**: Số lượng token được sinh ra bên trong khối `<think>...</think>`.
4. **Số Token phản hồi trung bình (`avg_answer_tokens`)**: Số lượng token dành cho việc viết lời giải chính thức và đáp số sau thẻ `</think>`.
5. **Tỷ lệ dừng sớm (Early Exit Rate - %)**: Tỷ lệ phần trăm số câu hỏi được thuật toán RPDI-EE kích hoạt cơ chế phanh cưỡng bức.
6. **Vị trí bước dừng trung bình (`avg_exit_step`)**: Chỉ số bước token trung bình mà tại đó thẻ `</think>\n` được ép chèn vào chuỗi sinh.
7. **Tỷ lệ cạn kiệt ngân sách (Token Exhaustion Rate - %)**: Tỷ lệ câu hỏi chạm trần tối đa 768 tokens dẫn đến cụt lời giải hoặc không kịp xuất đáp án.
8. **Độ trễ thời gian (`avg_latency_s`)**: Thời gian suy luận tính bằng giây trên mỗi câu hỏi.

---

## 4.2. Phân Tích Hạn Chế Của Cấu Hình Ban Đầu & Hiện Tượng "Kẹt Đập Nhân Tạo"

Trong các thử nghiệm sơ khởi của đề tài, phương pháp RPDI-EE được cấu hình với các tham số:
$$W = 8, \quad \lambda = 1.2, \quad \text{min\_steps} = 64$$

### 4.2.1. Phát hiện thực nghiệm bất thường
Khi phân tích phân bố của `exit_step` trên tập dữ liệu kiểm thử thực tế, nghiên cứu ghi nhận hiện tượng bất thường:
* Giá trị bước dừng nhỏ nhất chính xác bằng 64: $\min(\text{exit\_step}) = 64$.
* Có tới **17 câu hỏi (chiếm ~18% tổng số ca phanh)** bị dừng sớm trong khoảng hẹp từ **64 đến 70 bước**.
* Hơn 37% số ca dừng sớm tập trung trước bước 80.

```
Số lượng mẫu phanh
   ▲
   │         ████ (17 câu dồn ứ tại 64–70 steps)
   │         ████
   │         ████
   │         ████
   │         ████    ██
   │         ████    ██    ██
   │_________████____██____██________________► Bước token
   0        64  70  100   150
             ▲
             └── [Rào cản nhân tạo: min_steps=64]
```

### 4.2.2. Bản chất toán học của hiện tượng
1. **Phương sai lớn do cửa sổ trượt quá hẹp ($W=8$)**: Cửa sổ 8 token là quá ngắn đối với một SLM. Chỉ cần một cụm từ phức tạp hoặc ký hiệu toán học có phân bố xác suất phân tán, giá trị độ trôi chảy cục bộ ($LTF$) lập tức tăng vọt, đẩy chỉ số $\text{RPDI} = LTF / GTF > 1.2$ ngay từ những bước đầu tiên (bước 10–20).
2. **Hiệu ứng "Vỡ đập" do `min_steps=64`**: Rào cản `min_steps=64` không có trong bài báo gốc (arXiv:2603.14251) mà vốn là một heuristic cảm tính nhằm ngăn hiện tượng phanh non. Rào cản này đóng vai trò như một "con đập" ngăn chặn thuật toán kích hoạt. Ngay khi bước sinh chạm ngưỡng 64, gặp dấu ngắt câu đầu tiên (khoảng 64–70), con đập bị dỡ bỏ và thuật toán lập tức phanh ồ ạt.
3. **Kết luận**: Các ca phanh tại 64–70 bước xảy ra do **hiệu ứng biên của rào cản nhân tạo**, chứ không phản ánh đúng thời điểm mạch suy luận của mô hình thực sự thoái hóa hay lạc hướng. Do đó, cần loại bỏ hoàn toàn `min_steps=64`, cho phép thuật toán tự do giám sát ngay khi đủ cửa sổ trượt ($i \ge W$) và tiến hành quét lưới siêu tham số để tìm điểm hoạt động tự nhiên.

---

## 4.3. Thiết Kế Thực Nghiệm Quét Lưới Hệ Thống (Ablation Study)

Để tìm bộ tham số tối ưu dựa trên cơ sở định lượng thay vì trực giác, nghiên cứu thiết lập không gian quét lưới 2 chiều gồm **16 cấu hình RPDI** cùng **6 cấu hình tham chiếu đối chứng** trên 100 câu GSM8K:

### 4.3.1. Không gian tìm kiếm tham số
* **Kích thước cửa sổ trượt $W$**: $W \in \{16, 32, 64, 128\}$ (Loại bỏ $W=8$ vì quá nhiễu, mở rộng để làm mượt phương sai cục bộ).
* **Ngưỡng độ lệch hướng $\lambda$**: $\lambda \in \{1.5, 2.0, 2.5, 3.0\}$ (Tiệm cận ngưỡng khuyến nghị của bài báo gốc $\lambda=2.0$).
* **Điều kiện kích hoạt**: $i \ge W$ (Tuân thủ nguyên lý bài báo: bắt đầu giám sát ngay khi cửa sổ $W$ được lấp đầy).

### 4.3.2. Nhóm Baselines đối chứng (Reference Baselines)
1. **Standard Baseline (No Exit)**: Mô hình tự do suy nghĩ đến khi tự đóng thẻ `</think>` hoặc cạn kiệt ngân sách 768 tokens.
2. **Fixed Hard Truncation Baselines ($N \in \{32, 64, 96, 128, 256\}$)**: Cắt cứng mù mạch suy nghĩ tại đúng $N$ tokens bất kể trạng thái entropy. Đây là cơ sở đối chứng quan trọng nhằm chứng minh: *RPDI phanh có nhận thức vượt trội hơn việc cắt dừng ngẫu nhiên*.

### 4.3.3. Tối ưu hóa tính toán: "Offline Exit Simulation + Unique Answer Resumption"
Chạy đầy đủ 16 cấu hình $\times$ 100 câu với tốc độ ~30s/câu sẽ tiêu tốn hơn **13 giờ GPU**. Đề tài đã đề xuất và hiện thực hóa giải thuật tối ưu hóa 3 pha:
* **Pha 1 (Ghi vết trên GPU)**: Chạy Standard Baseline 1 lần duy nhất, thu thập mảng entropy $H(t_i)$, ID token và vị trí ranh giới câu vào [results/baseline_traces.json](file:///b:/GIT/slm-reasoning-project/results/baseline_traces.json).
* **Pha 2 (Mô phỏng Offline trên CPU)**: Do giải mã Greedy là tất định, chuỗi token trước điểm dừng ở mọi cấu hình về nguyên tắc trùng khớp với Baseline. Việc tính toán điểm dừng cho 16 cấu hình chỉ là duyệt mảng 1D, thực thi trong **0.003 giây** trên CPU.
  * *Kiểm chứng*: Đối chiếu với lượt chạy thật (Giai đoạn 2) trên cùng 100 câu, điểm phanh mô phỏng trùng với điểm phanh thật ở **93/100 câu**. 7 câu lệch do lỗi của `RPDILogitsProcessor` (đã sửa, xem Mục 4.6.5). Tuy nhiên, accuracy mô phỏng là 78% trong khi chạy thật cho **73%** trên cùng 100 câu. Phần chênh lệch nằm ở pha sinh câu trả lời (prefill lại prefix so với sinh liên tục trong một lệnh `generate()`), và đang được kiểm chứng bằng `method_c_rpdi_ee/ablation/diagnose_sim_vs_online.py`. Do đó, các con số trong Bảng 4.1 nên được hiểu là kết quả mô phỏng.
* **Pha 3 (Unique Exit Cache trên GPU)**: Tập hợp các cặp phân biệt `(sample_id, think_tokens)` thực sự cần sinh câu trả lời. Các cấu hình không phanh hoặc có cùng điểm phanh được tái sử dụng kết quả, **tiết kiệm hơn 85% thời gian GPU**.

---

## 4.4. Kết Quả Định Lượng & Đường Biên Pareto (Tập Thăm Dò 100 Câu GSM8K)

Bảng 4.1 tổng hợp toàn diện kết quả thực nghiệm của 16 cấu hình RPDI và các Baseline tham chiếu trên 100 câu hỏi GSM8K kiểm thử:

### Bảng 4.1: Bảng tổng hợp hiệu năng quét lưới Ablation Study trên 100 câu GSM8K

| Nhóm | Cấu hình | Độ chính xác Acc (%) | Tổng Token TB | Think Token TB | Tỷ lệ phanh (%) | Bước phanh TB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | **Standard (No Exit)** | **59.00%** | **615.2** | **461.3** | **0.0%** | — |
| **RPDI-EE** | **RPDI(W=16, λ=1.5)** | **78.00%** | **518.1** | **121.9** | **75.0%** | **122.4** |
| | RPDI(W=16, λ=2.0) | 76.00% | 563.2 | 272.6 | 56.0% | 316.9 |
| | RPDI(W=16, λ=2.5) | 66.00% | 593.3 | 403.1 | 33.0% | 497.6 |
| | RPDI(W=16, λ=3.0) | 61.00% | 610.2 | 441.7 | 12.0% | 515.4 |
| | RPDI(W=32, λ=1.5) | 74.00% | 547.5 | 212.4 | 61.0% | 247.2 |
| | RPDI(W=32, λ=2.0) | 62.00% | 587.4 | 405.8 | 32.0% | 506.0 |
| | RPDI(W=32, λ=2.5) | 61.00% | 612.4 | 453.4 | 11.0% | 594.5 |
| | RPDI(W=32, λ=3.0) | 59.00% | 615.0 | 460.4 | 2.0% | 638.0 |
| | RPDI(W=64, λ=1.5) | 68.00% | 577.3 | 359.3 | 38.0% | 436.2 |
| | RPDI(W=64, λ=2.0) | 60.00% | 610.4 | 453.9 | 9.0% | 621.9 |
| | RPDI(W=64, λ=2.5) | 59.00% | 615.2 | 460.4 | 2.0% | 723.0 |
| | RPDI(W=64, λ=3.0) | 59.00% | 615.2 | 461.3 | 0.0% | — |
| | RPDI(W=128, λ=1.5) | 60.00% | 602.3 | 431.5 | 19.0% | 531.4 |
| | RPDI(W=128, λ=2.0) | 59.00% | 615.2 | 461.3 | 1.0% | 765.0 |
| | RPDI(W=128, λ=2.5) | 59.00% | 615.2 | 461.3 | 0.0% | — |
| | RPDI(W=128, λ=3.0) | 59.00% | 615.2 | 461.3 | 0.0% | — |
| **Cắt cứng** | HardTrunc(N=32) | 69.00% | 549.3 | 32.0 | 100.0% | 34.0 |
| | HardTrunc(N=64) | 68.00% | 550.7 | 64.0 | 100.0% | 66.0 |
| | HardTrunc(N=96) | 65.00% | 536.2 | 95.1 | 94.0% | 98.0 |
| | HardTrunc(N=128) | 75.00% | 520.5 | 123.7 | 85.0% | 130.0 |
| | HardTrunc(N=256) | 73.00% | 546.1 | 214.2 | 61.0% | 258.0 |

---

## 4.5. Trực Quan Hóa Đường Biên Pareto Frontier

Đồ thị hình 4.1 biểu diễn không gian đánh đổi 2 chiều giữa Độ chính xác và Chi phí Token được xuất tại [report/pareto_accuracy_vs_tokens.png](file:///b:/GIT/slm-reasoning-project/report/pareto_accuracy_vs_tokens.png):

```
Độ chính xác Acc (%)
   ▲
80%│   * RPDI(W=16, λ=1.5) [78.0%, 518.1 tokens]  <-- [PARETO CHAMPION]
   │    \
75%│     \-- ■ HardTrunc(N=128) [75.0%]
   │      \
70%│       \
   │        \-- ■ HardTrunc(N=32) [69.0%]
65%│
   │
60%│                                               ● Standard Baseline [59.0%, 615.2 tokens]
   │
55%│
   └─────────────────────────────────────────────────────────────────────────────►
  500                       550                       600                    Tổng Tokens
```

### Phân tích đường biên Pareto:
1. **Điểm cực trị Pareto Frontier**: Điểm cấu hình **`RPDI(W=16, λ=1.5)`** nằm ở vị trí cao nhất và xa nhất về bên trái, áp đảo 15 cấu hình RPDI còn lại. Có hai giới hạn cần lưu ý:
   * **So với cắt cứng**: `HardTrunc(N=128)` đạt 75% với lượng token gần tương đương. Chênh lệch 3 câu trên 100 câu **không có ý nghĩa thống kê**, nên tập 100 câu chưa đủ để kết luận RPDI vượt trội cắt cứng. Phép so sánh này được làm lại trên 605 câu held-out (Mục 4.7.3).
   * **Điểm tối ưu nằm ở góc lưới**: $W=16$ và $\lambda=1.5$ đều là giá trị nhỏ nhất đã quét. Cần mở rộng lưới ($W \in \{4, 8\}$, $\lambda \in \{1.1, 1.25\}$) trên tập validation để xác nhận đây là điểm tối ưu thật.
2. **Xu hướng phân kỳ theo $W$ và $\lambda$**:
   * Khi cố định $W=16$, việc tăng $\lambda$ từ 1.5 lên 3.0 kéo Accuracy giảm đơn điệu từ $78\% \rightarrow 76\% \rightarrow 66\% \rightarrow 61\%$, đồng thời tổng token tăng dần tiệm cận Baseline.
   * Khi $W \ge 64$ và $\lambda \ge 2.5$, tất cả các điểm co cụm quanh tọa độ của Standard Baseline (59% Accuracy, 615 tokens), chứng minh hệ thống hoàn toàn vô hiệu khi cửa sổ quá lớn hoặc ngưỡng quá cao.

---

## 4.6. Thực Nghiệm Mở Rộng Quy Mô Lớn (Scale-Up Verification trên 705 câu GSM8K)

Để kiểm chứng câu hỏi khoa học mang tính quyết định: **"Liệu mức tăng vọt Accuracy (+19%) ghi nhận trên 100 câu có phải là hiện tượng ngẫu nhiên do phương sai mẫu nhỏ, hay là quy luật bản chất có tính ổn định thống kê?"**, đề tài triển khai đánh giá mở rộng cấu hình tối ưu `RPDI(W=16, λ=1.5)` trên quy mô lớn của bộ đề GSM8K.

### 4.6.1. Tiến trình thực thi thực tế
* Thực nghiệm đã chạy liên tục **705 câu hỏi độc lập** (chiếm **53.45%** toàn bộ tập GSM8K gồm 1,319 câu) trong suốt **11.66 giờ tính toán GPU** liên tục trước khi dừng lại ở câu 706 do kết thúc phiên làm việc.
* Với $N=705$, khoảng tin cậy 95% của accuracy có độ rộng khoảng ±3.5 điểm, đủ để phân biệt mức chênh lệch khoảng 19 điểm giữa hai phương pháp (kiểm định chi tiết ở Mục 4.6.4).

### Bảng 4.2: Kết quả đánh giá đối chứng toàn diện Standard vs RPDI-EE trên 705 câu GSM8K

| Chỉ số đo lường | Standard Baseline | RPDI-EE ($W=16, \lambda=1.5$) | Biến thiên / Hiệu quả |
| :--- | :---: | :---: | :---: |
| **Số lượng câu hỏi đánh giá ($N$)** | 705 câu | 705 câu | — |
| **Số câu giải đúng** | 375 câu | **506 câu** | **+131 câu đúng** |
| **Độ chính xác (Accuracy - %)** | **53.19%** | **71.77%** | **+18.58% tuyệt đối** *(Tăng tương đối +34.93%)* |
| **Token suy nghĩ trung bình (`<think>`)** | **464.53** tokens | **131.85** tokens | **Cắt giảm 71.62%** |
| **Token phản hồi trung bình (`</think>`)** | 147.70 tokens | **412.74** tokens | +265.04 tokens *(Đủ chỗ cho lời giải)* |
| **Tổng số Token trung bình / câu** | **612.22** tokens | **544.59** tokens | **Tiết kiệm 11.05%** tổng tokens |
| **Độ trễ trung bình / câu (Latency)** | 31.44 giây | 28.11 giây | **Nhanh hơn 10.61%** |
| **Tốc độ sinh token (Throughput)** | 19.47 tok/s | 19.37 tok/s | Giữ nguyên hiệu năng phần cứng |
| **Tỷ lệ phanh trong pha suy nghĩ** | 0.00% | **71.49%** (504/705 câu) | Vị trí phanh TB: **131.3 steps** *(xem Mục 4.6.5)* |
| **Tỷ lệ cạn ngân sách (`max_tokens=768`)** | **47.09%** (332/705 câu) | **21.56%** (152/705 câu) | **Giảm 54.2% số ca bị cụt đáp án** |
| **Tổng thời gian GPU thực thi** | 6.16 giờ | 5.50 giờ | Tiết kiệm 0.66 giờ GPU |

### 4.6.2. Phân tích ma trận đối đầu trực tiếp (Head-to-Head Matrix)

Bảng 4.3 phân tách hành vi dự đoán của Standard Baseline và RPDI-EE trên từng câu hỏi cụ thể trong số 705 câu:

### Bảng 4.3: Ma trận tương quan dự đoán giữa Standard Baseline và RPDI-EE (N = 705)

| Phân loại trường hợp | Số câu | Tỷ lệ (%) | Ý nghĩa bản chất |
| :--- | :---: | :---: | :--- |
| **Cả hai cùng ĐÚNG** | 343 | 48.65% | Bài toán cơ bản, cả hai cơ chế đều trích xuất đúng đáp số. |
| **Cả hai cùng SAI** | 167 | 23.69% | Bài toán khó vượt quá năng lực tri thức nội vi của SLM 1.5B. |
| **RPDI-EE CỨU NGUY (RPDI Đúng, Std Sai)** | **163** | **23.12%** | Phanh kịp thời, dập tắt luẩn quẩn và giải phóng ngân sách viết đáp án đúng. |
| **RPDI-EE Phanh nhầm (Std Đúng, RPDI Sai)** | **32** | **4.54%** | Hiện tượng phanh non khiến mô hình chưa kịp lập luận hoàn chỉnh. |

> **Nhận định quan trọng**: Tỷ số giữa số ca cứu nguy và số ca phanh nhầm là:
> $$\frac{\text{Số ca cứu nguy}}{\text{Số ca phanh nhầm}} = \frac{163}{32} \approx \mathbf{5.09 \text{ lần}}$$
> Cứ mỗi 1 câu bị ảnh hưởng tiêu cực do phanh non, RPDI-EE mang lại lợi ích giải cứu thành công cho hơn **5 câu hỏi khác**, giải thích lý do vì sao độ chính xác tăng vọt từ 53.19% lên 71.77%.

---

### 4.6.3. Đối chiếu tính nhất quán giữa Tập 100 câu và Tập 705 câu

Bảng 4.4 so sánh các chỉ số cốt lõi giữa hai quy mô thử nghiệm nhằm đánh giá tính tái lập và độ bền vững thống kê của phương pháp:

### Bảng 4.4: So sánh tính nhất quán giữa tập thăm dò (100 câu) và tập mở rộng (705 câu)

| Chỉ số | Tập thăm dò (100 câu) | Tập mở rộng (705 câu) | Độ sai lệch ($\Delta$) / Nhận xét |
| :--- | :---: | :---: | :--- |
| **Accuracy Standard Baseline** | 59.00% | 53.19% | Phản ánh độ khó đa dạng hơn khi mở rộng mẫu |
| **Accuracy RPDI-EE ($W=16, \lambda=1.5$)** | 78.00% *(mô phỏng)* | 71.77% | Chạy thật trên chính 100 câu này chỉ đạt **73.00%** (Mục 4.3.3) |
| **Mức cải thiện Accuracy ($\Delta \text{Acc}$)** | +19.00% *(mô phỏng)* | +18.58% | Trên 605 câu held-out: **+19.34%** (Mục 4.6.4) |
| **Mức tăng tương đối (Relative Gain)** | +32.20% | +34.93% | Đồng nhất ở mức tăng ~1/3 hiệu năng |
| **Tỷ lệ cắt giảm Think Tokens** | **73.6%** | **71.6%** | Đồng nhất ở mức ~72% |
| **Vị trí bước phanh trung bình** | 122.4 steps | 131.3 steps | Chỉ tính phanh trong pha suy nghĩ |
| **Tỷ lệ phanh trong pha suy nghĩ** | 75.0% | 71.5% | Ổn định ở mức khoảng 3/4 số câu |
| **Tỷ lệ cạn ngân sách ở Standard** | 45.0% | 47.09% | Khẳng định bản chất nghẽn token cố hữu của SLM |

### 4.6.4. Kết quả trên tập held-out 605 câu (kết quả đánh giá chính)

Vì 100 câu dùng để chọn $W, \lambda$ nằm trong 705 câu, kết quả khách quan được tính trên **605 câu còn lại (câu 100–704)**. Số liệu sinh bởi `method_c_rpdi_ee/analyze_results.py`; chi tiết tại [report/phan_tich_heldout.md](phan_tich_heldout.md).

### Bảng 4.5: Kết quả trên 605 câu held-out (khoảng tin cậy Wilson 95%)

| Chỉ số | Standard Baseline | RPDI-EE ($W=16, \lambda=1.5$) |
| :--- | :---: | :---: |
| Số câu đúng | 316/605 | **433/605** |
| Accuracy | 52.23% [48.2, 56.2] | **71.57% [67.8, 75.0]** |
| Token suy nghĩ TB | 465.1 | 133.6 |
| Tỷ lệ cạn ngân sách | 47.60% | 22.64% |

* **So sánh cặp**: RPDI-EE đúng mà baseline sai ở **141** câu; ngược lại ở **24** câu. Chênh lệch **+19.34 điểm**, khoảng tin cậy 95% **[+15.5, +23.2]**, kiểm định McNemar chính xác $p \approx 2.4 \times 10^{-21}$.
* Kết luận: mức tăng so với baseline **giữ nguyên trên dữ liệu không dùng để chọn tham số** và có ý nghĩa thống kê rất mạnh.

### Bảng 4.6: Phân tách theo trạng thái cạn ngân sách của baseline (605 câu)

| Nhóm câu | Số câu | Standard | RPDI-EE |
| :--- | :---: | :---: | :---: |
| Baseline tự đóng `</think>` trong 768 token | 317 | 80.44% | 79.18% |
| Baseline cạn 768 token | 288 | 21.18% | **63.19%** |

Toàn bộ mức tăng đến từ nhóm câu mà baseline bị cạn ngân sách. Ở nhóm baseline tự kết thúc, RPDI-EE không cải thiện (giảm nhẹ 1.3 điểm). Nói cách khác, dưới ngân sách 768 token, cơ chế tác động chính của RPDI-EE là **phân bổ lại ngân sách** từ pha suy nghĩ sang pha trả lời, chứ không phải sửa mạch suy luận sai.

### 4.6.5. Lỗi phanh trong pha trả lời và hiệu chỉnh số liệu

Bản `RPDILogitsProcessor` dùng trong lượt chạy 705 câu **vẫn tiếp tục giám sát entropy sau khi mô hình đã tự đóng `</think>`**, và có thể chèn thêm một thẻ `</think>` vào giữa phần trả lời. Có **61/705 câu** rơi vào trường hợp này. Hệ quả:
* Cờ `early_exited` gốc báo 80.14% (565 câu), nhưng chỉ **71.49% (504 câu)** là phanh thật trong pha suy nghĩ. Các bảng trong chương đã dùng con số hiệu chỉnh.
* Ảnh hưởng tới accuracy không đáng kể: trên 61 câu này, baseline đúng 52 câu và RPDI-EE đúng 51 câu.
* Lỗi đã được sửa trong `method_c_rpdi_ee/logits_processor.py`: bộ xử lý ngừng giám sát ngay khi mô hình tự sinh `</think>`. Các thực nghiệm mới dùng bản đã sửa.

---

## 4.7. Thảo Luận Chuyên Sâu (In-depth Discussion)

### 4.7.1. Cơ chế chính: Giải phóng ngân sách và hóa giải hiện tượng Token Exhaustion
* Ở chế độ Standard Baseline, **47.09% số câu hỏi (332/705 câu)** mô hình sinh `<think>` kéo dài quá mức, chạm trần 768 tokens mà không kịp đóng thẻ `</think>`, dẫn đến câu trả lời bị cắt cụt và bị chấm sai.
* Cấu hình `RPDI(W=16, λ=1.5)` phanh trong pha suy nghĩ ở 71.5% số câu, tại bước trung bình **131.3 tokens**, nhường hơn 400 token cho pha trả lời.
* Bảng 4.6 cho thấy toàn bộ mức tăng accuracy đến từ nhóm câu mà baseline bị cạn ngân sách (21.18% → 63.19%). Ở nhóm baseline tự kết thúc, RPDI-EE không cải thiện. Đây là bằng chứng trực tiếp rằng, dưới ngân sách 768 token, RPDI-EE hoạt động chủ yếu như một **cơ chế phân bổ lại ngân sách**.

### 4.7.2. Phanh nhầm và các trường hợp vẫn cạn ngân sách
* Trên 705 câu có **32 câu phanh nhầm** (baseline đúng, RPDI-EE sai), trong đó 31 câu là do phanh trong pha suy nghĩ. Tỷ số giải cứu/phanh nhầm là 163/32 ≈ 5.1 trên 705 câu, và 141/24 ≈ 5.9 trên 605 câu held-out.
* Có **152 câu (21.56%)** vẫn cạn ngân sách dù đã phanh; accuracy của nhóm này chỉ 36.8%. Sau khi bị chèn `</think>`, mô hình tiếp tục suy luận lại trong pha trả lời và lại dùng hết ngân sách. Đây là hướng cải tiến: ép tiền tố trả lời (ví dụ `The final answer is \boxed{`) hoặc dành riêng một phần ngân sách cho pha trả lời.

### 4.7.3. Câu hỏi còn mở: Tín hiệu entropy có đóng góp gì so với cắt cứng?
* Trên 100 câu (mô phỏng), `HardTrunc(N=128)` đạt 75% so với 78% của RPDI-EE; `HardTrunc(N=32)` cũng đạt 69%, cao hơn baseline 10 điểm. Chênh lệch giữa RPDI-EE và cắt cứng **chưa có ý nghĩa thống kê** trên cỡ mẫu này.
* Vì mức tăng chủ yếu đến từ việc giải phóng ngân sách (Mục 4.7.1), cần tách riêng phần đóng góp của tín hiệu entropy. Hai thực nghiệm đối chứng được thực hiện trên cùng 605 câu held-out bằng `method_c_rpdi_ee/run_controls.py`:
  1. **Cắt cứng tại 130 token** (xấp xỉ số token suy nghĩ trung bình của RPDI-EE).
  2. **Phanh ngẫu nhiên**: phanh tại biên câu đầu tiên sau một bước $N$ lấy ngẫu nhiên từ phân phối bước phanh thực tế của RPDI-EE. Baseline này giữ nguyên thời điểm phanh "trung bình" và cơ chế chèn thẻ, chỉ bỏ đi tín hiệu entropy.
* *[Kết quả sẽ được bổ sung sau khi chạy.]* Nếu RPDI-EE vượt cả hai baseline với ý nghĩa thống kê, tín hiệu entropy có giá trị thực trên SLM. Nếu không, kết luận đúng là dưới ngân sách hẹp, nút thắt của SLM là ngân sách, và lợi thế của RPDI-EE là tự điều chỉnh điểm dừng mà không cần chọn trước $N$.

---

## 4.8. Đối Chiếu Với Bài Báo Gốc RPDI-EE (arXiv:2603.14251)

### 4.8.1. Điều Kiện Thực Nghiệm

| Yếu tố | **Bài báo gốc** (Li et al., 2026) | **Thực nghiệm này** |
| :--- | :--- | :--- |
| **Mô hình** | 8 mô hình: DeepSeek-R1-Distill-Qwen **1.5B** / 7B / 14B / 32B, DeepSeek-R1-Distill-Llama 8B / 70B, Qwen3-30B-A3B-Thinking, Qwen3-235B-A22B-Thinking | DeepSeek-R1-Distill-Qwen **1.5B** |
| **Chiến lược giải mã** | Greedy | Greedy (`do_sample=False`) |
| **Ngân sách token** | **16,384** (32,768 cho Qwen3-Thinking) | **768** |
| **Benchmark** | GSM8K, AMC23, MATH500 (gộp thành "Math Easy"); AIME2024, AIME2025, OlympiadBench ("Math Hard"); GPQA-Diamond | GSM8K |
| **Baseline đối chứng** | Vanilla CoT; NoThinking, ThinkLess; DEER, Dynasor-CoT (trên 1.5B chỉ có Vanilla CoT và DEER) | Vanilla CoT; cắt cứng $N \in \{32, …, 256\}$ |
| **Tham số** | $W=512$, $\lambda=2.0$ | $W=16$, $\lambda=1.5$ |
| **Kết quả trên 1.5B** | Math Easy: 65.7% → 69.4% (**+3.7**); trung bình 7 tập: 40.0% → 43.4% (+3.4) | GSM8K (605 câu held-out): 52.23% → 71.57% (**+19.34**) |

*Ghi chú*: Bài báo không báo cáo riêng GSM8K mà gộp vào nhóm Math Easy, nên không so sánh trực tiếp được con số accuracy tuyệt đối.

### 4.8.2. Điểm Tương Đồng
1. **Cùng mô hình, cùng chiến lược giải mã**: Bài báo gốc đã đánh giá RPDI-EE trên chính DeepSeek-R1-Distill-Qwen-1.5B với giải mã Greedy. Thực nghiệm này **không phải lần đầu áp dụng RPDI-EE cho mô hình 1.5B**.
2. **Cùng chiều tác động**: Cả hai đều ghi nhận RPDI-EE cải thiện accuracy so với Vanilla CoT trên mô hình 1.5B.

### 4.8.3. Điểm Khác Biệt — Đóng Góp Của Nghiên Cứu
* **Chế độ ngân sách cực hẹp**: Điểm khác biệt cốt lõi là ngân sách 768 token so với 16,384 token. Ở 16k token, mô hình 1.5B gần như luôn tự kết thúc pha suy nghĩ, nên RPDI-EE chỉ tác động lên chất lượng suy luận (+3.7). Ở 768 token, gần một nửa số câu bị cạn ngân sách, và RPDI-EE chủ yếu giải cứu các câu này (Bảng 4.6). **Mức +19.34 và +3.7 đo hai hiệu ứng khác nhau, không nên so sánh trực tiếp theo kiểu "lớn hơn gấp nhiều lần".**
* **Siêu tham số dịch chuyển theo ngân sách**: Dưới ngân sách 768 token, cấu hình tối ưu dịch từ $W=512, \lambda=2.0$ sang $W=16, \lambda=1.5$. Điều này hợp lý vì với 768 token, cửa sổ $W=512$ gần như không bao giờ đủ dữ liệu để kích hoạt (Bảng 4.1: $W=128$ đã gần như vô hiệu). Nguyên nhân chính là ngân sách, không phải kích thước mô hình, vì bài báo gốc dùng $W=512$ cho chính mô hình 1.5B.
* **Phân tích cơ chế và baseline cắt cứng**: Bài báo gốc không so sánh với cắt cứng. Thực nghiệm này bổ sung phép đối chứng với cắt cứng và phanh ngẫu nhiên (Mục 4.7.3).
* **Kế hoạch nối kết quả với bài gốc**: Khảo sát accuracy theo ngân sách (768 → 1,536 → 4,096 → 16,384 token) trên một tập con, với cả cấu hình $W=16, \lambda=1.5$ và $W=512, \lambda=2.0$. Kỳ vọng mức tăng giảm dần về khoảng +3–4 điểm khi ngân sách đủ rộng.

---

## 4.9. Kết Luận Chương 4

1. **Cấu hình tốt nhất trong lưới đã quét**: $W = 16, \lambda = 1.5, \text{min\_steps} = W$. Đây là điểm ở góc lưới nên cần mở rộng lưới trên tập validation để xác nhận.
2. **Hiệu quả trên dữ liệu held-out**: Trên 605 câu không dùng để chọn tham số, accuracy tăng từ 52.23% lên 71.57% (**+19.34 điểm**, KTC 95% [+15.5, +23.2], McNemar $p \approx 2.4 \times 10^{-21}$), giảm khoảng 71% token suy nghĩ.
3. **Cơ chế**: Dưới ngân sách 768 token, mức tăng đến gần như hoàn toàn từ việc giải cứu các câu bị cạn ngân sách. RPDI-EE không cải thiện các câu mà mô hình tự kết thúc được.
4. **Câu hỏi còn mở**: Phần đóng góp riêng của tín hiệu entropy so với cắt cứng và phanh ngẫu nhiên đang được kiểm chứng trên 605 câu held-out (Mục 4.7.3).
5. **Ý nghĩa học thuật**: Đề tài mở rộng kết quả của bài báo gốc sang chế độ ngân sách token cực hẹp, mô phỏng điều kiện triển khai trên thiết bị biên. Đề tài chỉ ra rằng siêu tham số tối ưu và bản chất cơ chế tác động của RPDI-EE thay đổi đáng kể theo ngân sách.
