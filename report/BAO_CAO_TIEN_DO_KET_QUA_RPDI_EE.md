# BÁO CÁO TIẾN ĐỘ THỰC NGHIỆM ĐỀ TÀI
## ĐÁNH GIÁ PHƯƠNG PHÁP SUY LUẬN NHẬN THỨC SỚM (RPDI-EE) TRÊN MÔ HÌNH NGÔN NGỮ NHỎ (SLM)

---

- **Đề tài**: Nghiên cứu tối ưu hóa hiệu quả suy luận chuỗi tư duy (Chain-of-Thought) trên mô hình ngôn ngữ nhỏ (Small Language Model - SLM).
- **Mô hình nghiên cứu**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (1.5 tỷ tham số).
- **Phương pháp trọng tâm**: RPDI-EE (*Relative Pattern Drift Indicator - Early Exit*).
- **Tập dữ liệu đánh giá**: GSM8K (*Grade School Math 8K*).
- **Giảng viên hướng dẫn**: Thầy Vũ Hồng.
- **Thời gian báo cáo**: Tuần 2, Tháng 10/2026.

---

## 1. Tóm Tắt Các Công Việc Đã Hoàn Thành Trong Tuần

Trong tuần làm việc vừa qua, nhóm nghiên cứu đã tập trung thực hiện và hoàn thành các nội dung thực nghiệm trọng tâm sau:

1. **Phân tích và khắc phục hạn chế của thuật toán ban đầu**:
   - Chỉ ra bản chất của hiện tượng "kẹt đập nhân tạo" (artificial dam effect) khi sử dụng heuristic cố định $min\_steps=64$ và cửa sổ hẹp $W=8$.
   - Loại bỏ rào cản nhân tạo, đưa thuật toán về đúng nguyên lý tự nhiên: bắt đầu giám sát entropy động ngay khi chuỗi sinh đạt đủ kích thước cửa sổ ($i \ge W$).

2. **Hoàn thành thực nghiệm quét lưới Ablation Study (Tập 100 câu GSM8K)**:
   - Khảo sát toàn diện 16 cấu hình RPDI ($W \in \{16, 32, 64, 128\}$ và $\lambda \in \{1.5, 2.0, 2.5, 3.0\}$) cùng 6 cấu hình tham chiếu đối chứng (Standard Baseline và 5 mốc Hard Truncation).
   - Thiết lập đồ thị đường biên Pareto (*Pareto Frontier*), xác định điểm hoạt động tối ưu toàn cục của đề tài là:
     $$W^* = 16, \quad \lambda^* = 1.5, \quad min\_steps = W^*$$

3. **Triển khai thực nghiệm kiểm chứng quy mô lớn (Scale-Up Verification)**:
   - Mở rộng quy mô thực nghiệm từ tập 100 câu lên toàn bộ tập kiểm thử GSM8K với cấu hình tối ưu $W=16, \lambda=1.5$.
   - Tiến trình thực nghiệm thực thi liên tục trên phần cứng GPU trong suốt **11.66 giờ**, hoàn thành trọn vẹn **705 / 1,319 câu hỏi độc lập** (đạt **53.45%** tập kiểm thử chuẩn GSM8K).
   - Ghi nhận và cấu trúc hóa toàn bộ nhật ký thực nghiệm (log terminal `kq.txt` và file dữ liệu `results/method_c_results_705.json`).

4. **Soạn thảo báo cáo chuyên sâu Chương 4**:
   - Hoàn thiện tài liệu thuyết minh thực nghiệm chi tiết tại `report/chuong_4_thuc_nghiem_rpdi_ee.md` theo chuẩn học thuật.

---

## 2. Kết Quả Thực Nghiệm Định Lượng

### 2.1. Kết quả kiểm chứng quy mô lớn trên 705 câu GSM8K

Bảng dưới đây tổng hợp so sánh trực tiếp giữa phương pháp gốc (Standard Baseline) và phương pháp đề xuất (RPDI-EE) trên 705 câu hỏi:

| Chỉ số đánh giá | Standard Baseline | RPDI-EE ($W=16, \lambda=1.5$) | Hiệu quả cải thiện |
| :--- | :---: | :---: | :---: |
| **Số lượng mẫu kiểm thử ($N$)** | 705 câu | 705 câu | 53.45% toàn bộ GSM8K |
| **Số câu giải đúng** | 375 câu | **506 câu** | **Tăng thêm 131 câu đúng** |
| **Độ chính xác (Accuracy)** | **53.19%** | **71.77%** | **+18.58% tuyệt đối** *(Tăng tương đối +34.93%)* |
| **Số token suy nghĩ trung bình (`<think>`)** | 464.53 tokens | **131.85 tokens** | **Tiết kiệm 71.62%** token suy nghĩ |
| **Số token phản hồi trung bình (`</think>`)** | 147.70 tokens | **412.74 tokens** | Tăng +265.04 tokens cho phần lập luận |
| **Tổng số token trung bình / câu** | 612.22 tokens | **544.59 tokens** | **Tiết kiệm 11.05%** tổng chi phí sinh |
| **Độ trễ trung bình / câu (Latency)** | 31.44 giây | **28.11 giây** | **Nhanh hơn 10.61%** |
| **Tỷ lệ kích hoạt phanh sớm (Early Exit Rate)** | 0.00% | **80.14%** (565/705 câu) | Vị trí phanh trung bình: 143.54 steps |
| **Tỷ lệ cạn kiệt ngân sách (`max_tokens=768`)** | **47.09%** (332/705 câu) | **21.56%** (152/705 câu) | **Giảm 54.2% số ca bị cụt đáp án** |
| **Tổng thời gian GPU thực thi** | 6.16 giờ | 5.50 giờ | Tiết kiệm 0.66 giờ GPU |

---

### 2.2. Phân tích ma trận đối đầu (Head-to-Head Comparison)

Phân loại 705 câu hỏi theo ma trận kết quả giữa Standard Baseline và RPDI-EE:

| Nhóm kết quả | Số câu | Tỷ lệ (%) | Ý nghĩa bản chất |
| :--- | :---: | :---: | :--- |
| **Cả 2 phương pháp cùng ĐÚNG** | 343 | 48.65% | Bài toán cơ bản, mô hình suy luận tốt ở cả 2 chế độ. |
| **Cả 2 phương pháp cùng SAI** | 167 | 23.69% | Bài toán phức tạp vượt quá tri thức nội vi của SLM 1.5B. |
| **RPDI-EE giải cứu thành công (RPDI Đúng, Std Sai)** | **163** | **23.12%** | Phanh kịp thời vòng lặp suy nghĩ, giải phóng token để trả lời đúng. |
| **RPDI-EE phanh nhầm (Std Đúng, RPDI Sai)** | **32** | **4.54%** | Phanh non khiến mô hình chưa kịp hoàn tất thiết lập phương trình. |

Tỷ số giữa số ca giải cứu thành công và số ca phanh nhầm:
$$\frac{\text{Số ca giải cứu}}{\text{Số ca phanh nhầm}} = \frac{163}{32} \approx \mathbf{5.09 \text{ lần}}$$
Điều này giải thích vì sao độ chính xác tổng thể tăng trưởng vượt bậc từ 53.19% lên 71.77%.

---

### 2.3. Tính nhất quán giữa tập 100 câu và tập 705 câu

| Tiêu chí so sánh | Tập 100 câu (Ablation) | Tập 705 câu (Mở rộng) | Độ lệch ($\Delta$) / Nhận xét |
| :--- | :---: | :---: | :--- |
| **Mức tăng Accuracy ($\Delta \text{Acc}$)** | **+19.00%** (59.0% -> 78.0%) | **+18.58%** (53.2% -> 71.8%) | Độ ổn định gần như tuyệt đối ($\Delta = 0.42\%$) |
| **Tỷ lệ cắt giảm Token suy nghĩ** | **73.6%** | **71.6%** | Đồng nhất ở mức ~72% |
| **Vị trí bước phanh trung bình** | 122.4 tokens | 143.5 tokens | Chênh lệch nhỏ (~21 tokens) do bài toán phong phú hơn |
| **Tỷ lệ kích hoạt phanh** | 75.0% | 80.1% | Ổn định ở khoảng 3/4 đến 4/5 câu hỏi |
| **Tỷ lệ cạn token ở Standard** | 45.0% | 47.1% | Khẳng định bản chất nghẽn token cố hữu của SLM |

---

### 2.4. So sánh thực nghiệm của đề tài với bài báo gốc (arXiv:2603.14251)

| Tiêu chí đối chứng | Bài báo gốc (Li et al., 2026) | Thực nghiệm của đề tài | Ý nghĩa khoa học |
| :--- | :--- | :--- | :--- |
| **Quy mô mô hình** | DeepSeek-R1-Distill-Qwen 7B/14B/32B, Qwen3-30B/235B | DeepSeek-R1-Distill-Qwen **1.5B** | Đề tài tiên phong kiểm chứng trên dòng SLM 1.5B (chưa được bài báo gốc khảo sát). |
| **Chiến lược giải mã** | Sampling ($T \ne 0$) | **Greedy Decoding** (`do_sample=False`) | Đảm bảo tính tất định và khả năng tái lập 100% trong môi trường nghiên cứu chuẩn. |
| **Ngân sách Token** | 32,768 tokens (rất lớn) | **768 tokens** (tài nguyên giới hạn) | Phù hợp với kịch bản triển khai thực tế trên thiết bị biên (edge devices). |
| **Cấu hình tối ưu** | $W = 512, \lambda = 2.0$ | **$W^* = 16, \lambda^* = 1.5$** | Với SLM, cửa sổ ngắn hơn và ngưỡng nhạy hơn giúp phát hiện suy luận lạc hướng sớm hơn. |
| **Mức tăng Accuracy** | Trung bình **+3.9%** (distillation: **+5.1%**) | **+18.58% ~ +19.00%** (53.2% lên 71.8%) | SLM hưởng lợi lớn hơn gấp nhiều lần nhờ khắc phục triệt để hiện tượng cạn ngân sách token. |
| **Xu hướng cốt lõi** | RPDI-EE vượt trội hơn Vanilla CoT và Hard Truncation | Tái xác nhận nhất quán trên cả 100 câu và 705 câu | Khẳng định tính đúng đắn của nguyên lý toán học RPDI khi chuyển dịch sang mô hình nhỏ. |

---

## 3. Các Phát Hiện Khoa Học Đóng Góp

1. **Khắc phục điểm yếu "Nghẽn suy nghĩ" (Overthinking Trapping) của SLM**:
   - Ở mô hình nhỏ 1.5B, gần một nửa số câu hỏi (47.09%) bị rơi vào trạng thái suy nghĩ luẩn quẩn cho đến khi cạn sạch ngân sách 768 tokens, dẫn đến việc không kịp xuất thẻ đóng `</think>` và bị chấm sai.
   - RPDI-EE nhận diện biến động entropy tương đối bất thường và cưỡng bức chuyển trạng thái, giúp mô hình giữ lại hơn 400 token để trình bày lời giải tường minh.

2. **Sự dịch chuyển của siêu tham số tối ưu theo quy mô mô hình**:
   - Trong bài báo gốc (arXiv:2603.14251), các tác giả đề xuất cấu hình $W=512, \lambda=2.0$ cho các mô hình lớn (7B đến 235B) với ngân sách 32k tokens.
   - Thực nghiệm của đề tài chứng minh rằng đối với dòng SLM 1.5B và ngân sách 768 tokens, cấu hình tối ưu dịch chuyển về vùng cửa sổ ngắn hơn và nhạy hơn ($W^*=16, \lambda^*=1.5$).

3. **Vượt trội hơn cơ chế cắt cứng cố định (Hard Truncation)**:
   - RPDI phanh có nhận thức dựa trên trạng thái entropy thực tế của chuỗi sinh, đạt độ chính xác 78.00% (trên 100 câu) so với mốc cắt cứng tốt nhất chỉ đạt 75.00% (tại $N=128$).

---

## 4. Danh Mục Tài Liệu Và Dữ Liệu Bàn Giao Trong File Zip

Thư mục nộp kèm chứa các thành phần sau:

```
slm-reasoning-project/
├── report/
│   ├── BAO_CAO_TIEN_DO_KET_QUA_RPDI_EE.md   # Báo cáo tiến độ tổng hợp (file này)
│   ├── chuong_4_thuc_nghiem_rpdi_ee.md       # Báo cáo chi tiết Chương 4 (262 dòng học thuật)
│   ├── ablation_summary.md                   # Tóm tắt số liệu 16 cấu hình quét lưới
│   └── pareto_accuracy_vs_tokens.png         # Đồ thị đường biên Pareto
├── results/
│   ├── method_c_results_705.json             # Dữ liệu JSON chi tiết 705 câu hỏi (Standard & RPDI-EE)
│   ├── method_c_results.json                 # Dữ liệu JSON tập 100 câu Ablation Study
│   └── baseline_traces.json                  # Trace entropy phục vụ mô phỏng nhanh
├── method_c_rpdi_ee/
│   ├── logits_processor.py                   # Cài đặt giải thuật giám sát RPDI LogitsProcessor
│   ├── evaluate.py                           # Script thực nghiệm đánh giá chuẩn hóa
│   └── ablation/                             # Toàn bộ mã nguồn quét lưới và mô phỏng Pareto
└── kq.txt                                    # Nhật ký chạy thực nghiệm terminal (11.66 giờ GPU)
```

---

## 5. Kế Hoạch Triển Khai Trong Tuần Tới

1. **Hoàn tất 614 câu hỏi còn lại của GSM8K**:
   - Chạy tiếp nối (resume) từ câu 706 đến câu 1,319 để hoàn thiện 100% tập dữ liệu chuẩn hóa GSM8K.
2. **Mở rộng đánh giá trên tập toán MATH500 / SVAMP**:
   - Kiểm tra tính tổng quát hóa của tham số tối ưu $W^*=16, \lambda^*=1.5$ trên các dạng toán đố có mức độ phân nhánh tư duy khác nhau.
3. **Chuẩn bị tích hợp so sánh đối chứng với Nhánh B (RLVR - Học tăng cường nội vi)**:
   - Xây dựng ma trận so sánh giữa can thiệp lúc giải mã (Inference-time Intervention - RPDI-EE) và can thiệp trong huấn luyện (Training-time Alignment - RLVR).
