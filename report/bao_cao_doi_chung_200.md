# Báo cáo thực nghiệm đối chứng RPDI-EE: cắt cứng và phanh ngẫu nhiên (200 câu)

> Ngày chạy: 19/10/2026 · Môi trường: Google Colab, Tesla T4, float16 · Phân tích: `method_c_rpdi_ee/analyze_results.py`
>
> Dữ liệu gốc: `results/rpdi_results/controls_*.jsonl` · Bảng thống kê: `report/phan_tich_200.md`, `report/phan_tich_200_vs_rpdi.md`

## 1. Mục tiêu

Trên 605 câu held-out, RPDI-EE đã tăng độ chính xác từ 52.23% lên 71.57% ở ngân sách 768 token. Câu hỏi còn bỏ ngỏ (Gap 2 trong `tong_quan_tai_lieu_nhanh_c.md`) là:

> **Mức tăng này đến từ tín hiệu entropy (RPDI), hay chỉ đơn giản vì pha suy nghĩ được kết thúc sớm để còn ngân sách cho pha trả lời?**

Bài báo gốc (Guan et al., 2026) không có đối chứng cắt cứng hay phanh ngẫu nhiên, nên không trả lời được câu hỏi này. Thực nghiệm dưới đây bổ sung hai đối chứng đó.

## 2. Thiết lập

| Thành phần | Giá trị |
| :--- | :--- |
| Mô hình | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Dữ liệu | 200 câu GSM8K test, id 100–299 (không trùng 100 câu id 0–99 đã dùng để chọn W, λ) |
| Ngân sách | `max_new_tokens = 768` |
| Giải mã | Greedy (`do_sample=False`) |
| Phần cứng | Tesla T4, float16; 4 tiến trình chạy song song trên cùng GPU |
| Code | commit `6b4bff9` (đã sửa lỗi phanh trong pha trả lời của `RPDILogitsProcessor`) |

Bốn phương pháp được chạy lại cùng lúc, cùng code, cùng phần cứng:

| Phương pháp | Mô tả | Vai trò |
| :--- | :--- | :--- |
| `standard` | Sinh tự nhiên, không can thiệp | Baseline |
| `hardtrunc_N130` | Ép đóng `</think>` đúng sau 130 token suy nghĩ | Đối chứng "dừng mù" ở cùng mức token (RPDI-EE trung bình ≈132 token suy nghĩ) |
| `random` | Phanh tại biên câu đầu tiên sau một bước N ngẫu nhiên. N bốc từ phân phối bước phanh thật của RPDI-EE, seed cố định theo từng câu | Đối chứng chặt nhất: cùng phân phối thời điểm và cùng cách chèn thẻ như RPDI-EE, chỉ khác là không dùng entropy |
| `rpdi_W16_l1.5` | RPDI-EE với W=16, λ=1.5 | Phương pháp nghiên cứu |

Thống kê: khoảng tin cậy Wilson 95% cho accuracy; kiểm định McNemar chính xác (hai phía) và KTC 95% của chênh lệch cho so sánh cặp.

## 3. Kết quả

### 3.1. Kết quả từng phương pháp

| Phương pháp | Đúng | Accuracy | KTC 95% | Token TB | Token suy nghĩ TB | Token trả lời TB | Cạn ngân sách | Tỷ lệ phanh |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | 105/200 | 52.50% | [45.6, 59.3] | 619.3 | 492.2 | 127.1 | 50.50% | 0% |
| `hardtrunc_N130` | **144/200** | **72.00%** | [65.4, 77.8] | 551.0 | 126.2 | 424.8 | 27.50% | 85.0% |
| `random` | 137/200 | 68.50% | [61.8, 74.5] | 552.1 | 126.8 | 425.2 | 26.00% | 85.0% |
| `rpdi_W16_l1.5` | **143/200** | **71.50%** | [64.9, 77.3] | 564.8 | 133.1 | 431.7 | 27.00% | 70.5% |

### 3.2. So sánh cặp với Standard

| Phương pháp | Chỉ phương pháp đúng | Chỉ Standard đúng | Δ Accuracy | KTC 95% của Δ | McNemar p |
| :--- | ---: | ---: | ---: | :---: | ---: |
| `hardtrunc_N130` | 48 | 9 | +19.50 | [+12.6, +26.4] | 1.52e-07 |
| `random` | 40 | 8 | +16.00 | [+9.6, +22.4] | 3.31e-06 |
| `rpdi_W16_l1.5` | 43 | 5 | +19.00 | [+12.7, +25.3] | 1.37e-08 |

Cả ba cách phanh đều vượt Standard rất rõ (p < 1e-5).

### 3.3. So sánh cặp với RPDI-EE (phép so chính)

| Phương pháp | Chỉ phương pháp đúng | Chỉ RPDI-EE đúng | Δ Accuracy | KTC 95% của Δ | McNemar p |
| :--- | ---: | ---: | ---: | :---: | ---: |
| `hardtrunc_N130` | 18 | 17 | +0.50 | [−5.3, +6.3] | 1.000 |
| `random` | 13 | 19 | −3.00 | [−8.5, +2.5] | 0.377 |

- **RPDI-EE và cắt cứng N=130 tương đương.** Hai bên hơn thua gần như cân bằng (17 câu so với 18 câu). Với KTC [−5.3, +6.3], nếu RPDI-EE có lợi thế thật thì lợi thế đó **không vượt quá khoảng 5 điểm**.
- **RPDI-EE cao hơn phanh ngẫu nhiên 3 điểm nhưng không có ý nghĩa thống kê** (p = 0.38). Đây chỉ là gợi ý yếu rằng thời điểm phanh do entropy chọn tốt hơn bốc ngẫu nhiên một chút.

### 3.4. Phân tách theo trạng thái của baseline

| Nhóm | Số câu | Standard | Cắt cứng N=130 | Phanh ngẫu nhiên | RPDI-EE |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline tự đóng `</think>` | 99 | 80.81% | 81.82% | 78.79% | 82.83% |
| Baseline cạn 768 token | 101 | 24.75% | 62.38% | 58.42% | 60.40% |

| Chỉ số | Cắt cứng N=130 | Phanh ngẫu nhiên | RPDI-EE |
| :--- | ---: | ---: | ---: |
| Câu được giải cứu (baseline cạn ngân sách và sai → phương pháp đúng) | 43 | 37 | 41 |
| Câu bị làm hỏng (baseline tự kết thúc và đúng → phương pháp sai) | 4 | 5 | **0** |

### 3.5. Hành vi phanh

| Chỉ số | Cắt cứng N=130 | Phanh ngẫu nhiên | RPDI-EE |
| :--- | :---: | :---: | :---: |
| Số câu bị phanh | 170 | 170 | 141 |
| Vị trí phanh: trung vị | 131 | 109 | 103 |
| Vị trí phanh: P10–P90 | 131–131 | 68–220 | 60–239 |

- RPDI-EE **thực sự thích nghi theo từng câu**: điểm phanh trải rộng từ khoảng 60 đến 239 token (độ lệch chuẩn của số token suy nghĩ là 79). Trong 59 câu RPDI-EE không phanh, có 58 câu là câu baseline tự kết thúc.
- Dù vậy, độ chính xác tổng thể vẫn không khác cách cắt cố định ở 130 token.

### 3.6. Tính tái lập giữa hai phần cứng

So với lần chạy 705 câu trước (GPU hỗ trợ bf16) trên cùng 200 câu id 100–299:

| Phương pháp | Lần trước (bf16) | Lần này (T4, float16) | Số câu có kết quả đúng/sai trùng nhau |
| :--- | ---: | ---: | ---: |
| Standard | 105/200 | 105/200 | 200/200 |
| RPDI-EE | 143/200 | 143/200 | 200/200 |

Kết quả đúng/sai **trùng 100% trên từng câu**, dù RPDI-EE lần trước chạy bằng code chưa sửa lỗi phanh trong pha trả lời. Điều này cho thấy:
- Kết quả không phụ thuộc phần cứng hay kiểu số;
- Lỗi phanh trong pha trả lời trước đây chỉ làm sai lệch thống kê *tỷ lệ phanh*, không làm thay đổi accuracy.

## 4. Thảo luận

### 4.1. Kết luận chính

Ở ngân sách 768 token, **nguồn gốc mức tăng khoảng 19 điểm là việc kết thúc pha suy nghĩ sớm**, để mô hình còn ngân sách viết lời giải và đáp số. Với cỡ mẫu 200 câu, **chưa thấy tín hiệu entropy của RPDI tạo thêm khác biệt có ý nghĩa** so với cắt cố định ở cùng mức token.

Điều này nhất quán với cơ chế đã ghi nhận trên 605 câu: toàn bộ mức tăng nằm ở nhóm câu mà baseline bị cạn ngân sách. Trong nhóm này, cả ba cách phanh đều cứu được khoảng 37–43 câu.

### 4.2. Điểm khác biệt nhỏ nhưng đáng ghi nhận: RPDI-EE "an toàn" hơn

Trên nhóm câu baseline tự kết thúc, RPDI-EE **không làm hỏng câu nào** (0 câu). Cắt cứng làm hỏng 4 câu và phanh ngẫu nhiên làm hỏng 5 câu. Lý do là RPDI-EE không phanh trên phần lớn các câu này (58/59 câu không phanh thuộc nhóm này), trong khi cắt cứng cắt mọi câu suy nghĩ dài hơn 130 token.

Đổi lại, RPDI-EE giải cứu ít hơn cắt cứng 2 câu ở nhóm cạn ngân sách (41 so với 43), nên hai bên hòa nhau về tổng thể. Với số lượng nhỏ như vậy (0 so với 4), khác biệt này **chưa có ý nghĩa thống kê** và chỉ nên nêu như một quan sát.

### 4.3. Phát hiện phụ: cạn ngân sách chuyển từ pha suy nghĩ sang pha trả lời

Sau khi phanh, mô hình viết trung bình khoảng 425–432 token trong pha trả lời (Standard chỉ 127). Kết quả là khoảng 26–27% số câu **vẫn cạn ngân sách, nhưng lần này cạn trong pha trả lời** (RPDI-EE: 54 câu, cắt cứng: 55 câu).

Như vậy, sau khi bị ép đóng `</think>`, mô hình 1.5B **tiếp tục suy luận trong phần trả lời** thay vì viết ngay đáp số. Đây là giới hạn trên của mọi phương pháp phanh ở ngân sách này, và là hướng cải thiện trực tiếp (Gap 8): chèn thêm câu dẫn kiểu "Final Answer:" như s1, hoặc dành ngân sách riêng cho pha trả lời như Elastic Reasoning.

### 4.4. Đối chiếu với tài liệu

- **Phù hợp với** ThinkLess (Li et al., 2025) và NoThinking (Ma et al., 2025): cắt thô hoặc bỏ hẳn pha suy nghĩ đã đủ mạnh khi ngân sách hẹp.
- **Bổ sung cho** bài gốc RPDI (Guan et al., 2026): bài gốc chỉ so với Vanilla CoT và DEER trên mô hình 1.5B, ở ngân sách 16k. Đối chứng ở đây cho thấy cần thận trọng khi gán lợi ích cho tín hiệu thích nghi.
- **Phù hợp với** LEASH (Quamar et al., 2025): dừng dựa trên entropy không tự động tốt hơn các cách dừng đơn giản.

## 5. Hạn chế

1. **Cỡ mẫu 200 câu.** Chỉ phát hiện được chênh lệch từ khoảng 6 điểm trở lên giữa các phương pháp phanh. Một lợi thế 2–3 điểm, nếu có, sẽ không thấy được.
2. **N=130 không độc lập hoàn toàn với RPDI-EE.** Con số này được chọn để khớp số token suy nghĩ trung bình của RPDI-EE, và gần với cấu hình cắt cứng tốt nhất trong ablation trên câu 0–99 (N=128). Trong thực tế triển khai, N phải được chọn trước. Ở chiều ngược lại, RPDI-EE cũng đã được chỉnh W, λ trên cùng 100 câu đó.
3. **Một ngân sách, một tập dữ liệu, một mô hình.** Lợi thế lý thuyết của RPDI-EE là tự thích nghi khi ngân sách hoặc độ khó thay đổi, nhưng thí nghiệm này chưa kiểm tra điều đó.
4. **Greedy decoding.** Lệch khỏi thiết lập khuyến nghị T=0.6 của DeepSeek (xem Gap 10).
5. **Độ trễ không so sánh được.** 4 tiến trình chạy song song trên cùng GPU (độ trễ trung bình khoảng 74–77 giây mỗi câu), nên số liệu thời gian không phản ánh chi phí thật của từng phương pháp.

## 6. Đề xuất bước tiếp theo

| Ưu tiên | Việc | Mục đích |
| :-: | :--- | :--- |
| 1 | Chạy Standard, cắt cứng N=130 và RPDI-EE ở **ngân sách 2048** trên cùng 200 câu | Kiểm tra lợi thế thích nghi: N=130 cố định có thể cắt quá sớm khi ngân sách rộng hơn |
| 2 | Thêm "Final Answer:" sau `</think>` cho cả 3 cách phanh | Giảm khoảng 27% câu cạn ngân sách ở pha trả lời (mục 4.3) |
| 3 | Mở rộng lên 605 câu (chạy tiếp với `--end 705`) | Thu hẹp KTC để kết luận chắc hơn về RPDI-EE so với cắt cứng |
| 4 | Phân tích định tính 4 câu cắt cứng làm hỏng và 17/18 câu bất đồng giữa RPDI-EE và cắt cứng | Hiểu khi nào tín hiệu entropy giúp hoặc hại |

## 7. Cách viết vào luận văn

> Ở ngân sách 768 token, RPDI-EE nâng độ chính xác của DeepSeek-R1-Distill-Qwen-1.5B trên GSM8K từ 52.5% lên 71.5% (+19.0 điểm, McNemar p ≈ 1.4×10⁻⁸, 200 câu held-out). Tuy nhiên, phép cắt cứng ở 130 token suy nghĩ, tương đương số token trung bình của RPDI-EE, đạt 72.0%, và chênh lệch giữa hai phương pháp không có ý nghĩa thống kê (Δ = +0.5 điểm, KTC 95% [−5.3, +6.3], p = 1.0). Kết quả này cho thấy, trong chế độ ngân sách hẹp, lợi ích chủ yếu đến từ việc kết thúc pha suy nghĩ sớm để dành ngân sách cho pha trả lời, chứ chưa đến từ khả năng chọn thời điểm dừng của tín hiệu entropy. Điểm khác biệt đáng chú ý là RPDI-EE không làm hỏng câu nào trong số các câu mà baseline tự hoàn thành đúng, còn cắt cứng làm hỏng 4 câu. Dù vậy, khác biệt này chưa đủ lớn để kết luận.
