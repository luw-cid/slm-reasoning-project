# Báo cáo thí nghiệm: RPDI-EE trên vLLM theo ngân sách token, từ 512 đến 16384 token (200 câu)

> Ngày chạy: 02/11/2026 · Môi trường: Google Colab, Tesla T4, float16, vLLM 0.6.3.post1 · Phân tích: `method_c_rpdi_ee/vllm_engine/analyze_budgets.py`, cộng thêm phân tích bổ sung ngày 03/11/2026
>
> Dữ liệu gốc: `results/pilot_200/raw.jsonl` (token sinh ra và entropy pha suy nghĩ), `results/pilot_200/budgets.jsonl` (kết quả theo từng ngân sách) · Bảng tự động: `results/pilot_200/analysis.md`

## Tóm tắt

1. **Ở ngân sách hẹp, RPDI-EE (W=16, λ=1.5) tăng accuracy rất rõ.** Ở 768 token, accuracy tăng từ 55.0% lên 76.0% (+21.0 điểm, KTC 95% [+14.5, +27.5], McNemar p ≈ 3×10⁻⁹). Kết quả này lặp lại mức tăng +19.0 điểm đã đo bằng HF.
2. **RPDI-EE ở 768 token kém sinh tự do ở 16384 token khoảng 6 điểm.**
   - Accuracy: 76.0% so với 82.0% (KTC 95% [−11.5, −1.0], p = 0.036).
   - Số token chỉ bằng khoảng 20%, và phương pháp lấy lại được khoảng 78% phần accuracy bị mất khi hạ ngân sách từ 16k xuống 768.
   - Muốn đạt 76%, Standard cần ngân sách 2048 token.
3. **Phần lớn khoảng cách này nằm ở pha trả lời.** Trong 20 câu bị mất, có 13 câu đã phanh nhưng lời giải bị cắt ở 768 token, và 7 câu phanh quá sớm.
4. **Cấu hình của bài gốc (W=512, λ=2.0) không phanh lần nào** trên 200 câu, với cả hai cách hiểu tập biên. Giá trị RPDI lớn nhất ở mọi vị trí chỉ đạt 1.58. Vì vậy trên GSM8K, "RPDI-EE theo bài gốc ở 16k" thực chất chính là sinh tự do.
5. **Ở 16384 token, RPDI-EE giữ nguyên accuracy (82.0% = 82.0%) với 36% số token.**
   - Phần lớn số token tiết kiệm được đến từ việc phá các vòng lặp không bao giờ đóng `</think>`.
   - KTC vẫn còn rộng, và thí nghiệm chưa có đối chứng cắt cứng.
6. **Bước kiểm tra pipeline so với HF không đạt, nhưng không phải vì pipeline sai.** Kết quả HF được chạy bằng bf16, còn vLLM chạy bằng fp16. Các kiểm tra nội bộ của vLLM đều khớp.

## 1. Mục tiêu

Đây là câu hỏi nghiên cứu của Mục 4.8.4 (chương 4):

> Với ngân sách 768 token, RPDI-EE đạt độ chính xác **hơn, tương đương hay kém hơn** thiết lập của bài gốc (16384 token, W=512, λ=2.0), khi đo trên cùng mô hình, cùng prompt và cùng tập câu?

Đây là lần chạy thử trên 200 câu, trước khi mở rộng lên 605 câu held-out.

## 2. Thiết lập

| Thành phần | Giá trị |
| :--- | :--- |
| Mô hình | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Dữ liệu | 200 câu GSM8K test, id 100–299. Đây là cùng tập với báo cáo đối chứng 200 câu, và không trùng 100 câu id 0–99 đã dùng để chọn W, λ. |
| Prompt | `User: {câu hỏi}\nAssistant: <think>`, giống các thí nghiệm HF |
| Giải mã | Greedy (temperature = 0) |
| Ngân sách | Sinh một lần ở 16384 token, rồi suy ra các mức 512, 768, 1024, 1536, 2048, 3072, 4096, 8192 và 16384 |
| Phần cứng | Tesla T4, float16. Trọng số gốc ở bf16 được ép sang fp16. |
| Phần mềm | vLLM 0.6.3.post1 (engine V0), torch 2.4.0, transformers 4.46.3, Python 3.12 |
| Code | `method_c_rpdi_ee/vllm_engine/`, notebook `rpdi_vllm_colab.ipynb` |

Bốn phương pháp được so sánh:

| Phương pháp | Mô tả | Vai trò |
| :--- | :--- | :--- |
| `standard` | Sinh tự do, không can thiệp | Baseline |
| `rpdi_legacy_W16_l1.5` | Cấu hình của đề tài: W=16, λ=1.5, phanh ở dấu câu, kiểm tra sau khi nạp entropy, ép chuỗi `</think>\n`. Đây là bản sao chính xác của bản HF đã cho các kết quả 768 token. | Phương pháp nghiên cứu |
| `rpdi_W512_l2.0_wait` | Algorithm 1 của bài gốc với W=512, λ=2.0. Phanh ở token "Wait" (cách hiểu theo mã nguồn DEER), `</think>` thay cho "Wait". | Thiết lập bài gốc, cách hiểu 1 |
| `rpdi_W512_l2.0_newline` | Như trên, nhưng phanh ở token chứa ký tự xuống dòng | Thiết lập bài gốc, cách hiểu 2 |

**Suy ra mọi ngân sách từ một lần sinh.** Mỗi cặp (câu, phương pháp) chỉ được sinh một lần, ở 16384 token. Giải mã greedy và quyết định phanh chỉ phụ thuộc vào phần đã sinh. Vì vậy kết quả ở ngân sách B chính là B token đầu của lần sinh đó, và đáp án được chấm trên phần văn bản đã cắt.

**Thống kê.**
- Chênh lệch accuracy được tính theo cặp (cùng câu hỏi), kèm KTC bootstrap 90% và 95% (10000 lần lặp) và kiểm định McNemar chính xác.
- Biên δ = 3 điểm. Quy tắc kết luận xem ở Mục 4.8.4 của chương 4.

## 3. Độ tin cậy của pipeline

### 3.1. So với kết quả HF ở 768 token

| vLLM | HF | Acc vLLM | Acc HF | Δ | Kết quả đúng/sai trùng nhau | Văn bản trùng | Số token suy nghĩ trùng |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | `standard` | 55.0 | 52.5 | +2.5 | 66.5% | 0% | 24.5% |
| `rpdi_legacy_W16_l1.5` | `rpdi_W16_l1.5` | 76.0 | 71.5 | +4.5 | 78.5% | 0% | 0.5% |

Bước kiểm tra được thiết kế với ngưỡng lệch tối đa 2 điểm, nên kết quả là **không đạt**. Đây không phải lỗi của pipeline:

- **Kết quả HF được chạy bằng bf16, còn vLLM chạy bằng fp16.** Lỗi ghi kiểu số đã được đính chính trong `bao_cao_doi_chung_200.md`, Mục 3.6.
- **Khác kiểu số làm quỹ đạo greedy tách nhau rất sớm.** Vị trí ký tự đầu tiên khác nhau có trung vị là 30 (P10 = 1, P90 = 132). Có 114/200 câu đã khác nhau ngay trong 40 ký tự đầu, nên không có câu nào trùng văn bản.
- **Không chạy vLLM bằng bf16 trên T4 được.** vLLM chỉ hỗ trợ bf16 trên GPU có compute capability từ 8.0 trở lên, còn T4 là 7.5.

### 3.2. Kiểm tra nội bộ

| Kiểm tra | Kết quả |
| :--- | :--- |
| Phát lại bộ giám sát RPDI trên entropy và token đã ghi của `legacy` | Điểm phanh khớp **128/128** |
| Phần sinh ra trước điểm phanh của `legacy` so với `standard` | Giống hệt ở **127/128** câu |
| Mức tăng của RPDI-EE so với Standard ở 768 token | +21.0 điểm (vLLM), so với +19.0 điểm (HF) |

Kết quả phát lại cho thấy logits processor trong vLLM gọi bộ giám sát đúng cách. Trước đó, bộ giám sát `legacy` đã được kiểm chứng là khớp bộ mô phỏng offline của bản HF trên 100/100 vết entropy thật. Gộp hai kiểm tra này lại: **logic phanh chạy đúng như thiết kế**, và chênh lệch so với HF đến từ kiểu số và engine.

### 3.3. Nhiễu nền của vLLM

Hai chế độ `paper` không phanh lần nào (Mục 4.2), nên về thuật toán chúng giống hệt `standard`. Dù vậy, kết quả vẫn không trùng hoàn toàn:

| Phương pháp | Output giống hệt `standard` | Vị trí bắt đầu khác (trung vị) | Chỉ PP đúng / chỉ Standard đúng @768 | @16384 |
| :--- | ---: | ---: | :---: | :---: |
| `rpdi_W512_l2.0_wait` | 176/200 | token 227 | 3 / 1 | 1 / 5 |
| `rpdi_W512_l2.0_newline` | 174/200 | token 281 | 2 / 2 | 3 / 4 |

Nguyên nhân nhiều khả năng là kết quả tính toán của một câu trong vLLM phụ thuộc vào các câu khác cùng batch. Ngoài ra, log ghi nhận nhiều lần phải tính lại do thiếu KV cache.

Hệ quả là **accuracy dao động khoảng ±2 điểm chỉ do nhiễu**. Chênh lệch dưới khoảng 2 điểm giữa các phương pháp trong pipeline này không có ý nghĩa, và biên δ = 3 điểm đã gần bằng mức nhiễu.

## 4. Kết quả

### 4.1. Accuracy theo ngân sách

Accuracy % (số token trung bình trong ngoặc):

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | 46.5 (457) | 55.0 (594) | 64.0 (692) | 74.0 (813) | 77.5 (890) | 80.5 (1031) | 81.5 (1160) | 81.5 (1652) | 82.0 (2635) |
| `rpdi_legacy_W16_l1.5` | **55.5** (449) | **76.0** (537) | **80.5** (566) | **81.5** (587) | 81.5 (601) | 81.5 (626) | 81.5 (652) | 81.5 (754) | 82.0 (959) |
| `rpdi_W512_l2.0_wait` | 46.0 (458) | 56.0 (596) | 64.5 (692) | 74.5 (807) | 76.5 (878) | 77.5 (1005) | 78.5 (1124) | 80.0 (1574) | 80.0 (2475) |
| `rpdi_W512_l2.0_newline` | 47.0 (458) | 55.0 (595) | 62.5 (693) | 74.0 (814) | 76.5 (889) | 80.0 (1017) | 80.5 (1131) | 81.0 (1561) | 81.5 (2421) |

Tỷ lệ câu cạn ngân sách %:

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | 62.5 | 45.5 | 32.0 | 17.5 | 14.5 | 13.0 | 12.0 | 12.0 | 12.0 |
| `rpdi_legacy_W16_l1.5` | 54.5 | 18.5 | 6.5 | 3.5 | 2.5 | 2.5 | 2.5 | 2.5 | 2.5 |
| `rpdi_W512_l2.0_wait` | 63.0 | 46.0 | 30.5 | 16.0 | 13.0 | 12.0 | 11.0 | 11.0 | 11.0 |
| `rpdi_W512_l2.0_newline` | 63.0 | 45.0 | 32.0 | 17.0 | 14.0 | 11.5 | 10.5 | 10.5 | 10.5 |

Tỷ lệ phanh: `legacy` phanh 63.5% số câu ở 512 token và 64.0% từ 768 token trở lên. Hai chế độ `paper` phanh 0% ở mọi ngân sách.

Nhận xét:
- **Hai chế độ `paper` trùng với `standard`** ở mọi ngân sách. Chênh lệch nhiều nhất là 2 điểm, nằm trong mức nhiễu.
- **`legacy` bão hòa rất sớm.** Accuracy đạt 80.5% ở 1024 token và 81.5% từ 1536 token, trong khi số token trung bình chỉ tăng chậm (566 → 959).
- **Ở 16384 token, 12% số lượt `standard` (24 câu) không bao giờ đóng `</think>`.** Mô hình lặp lại cho đến hết ngân sách. Với `legacy`, chỉ còn 2.5% (5 câu), và cả 5 câu đều lặp trong pha trả lời sau khi đã phanh.

### 4.2. Vì sao cấu hình bài gốc không phanh

Tôi tính RPDI với W = 512 ở **mọi vị trí** của cả 200 chuỗi suy nghĩ của `standard`:
- Giá trị lớn nhất chỉ đạt **1.58**, còn P90 của giá trị lớn nhất trên mỗi câu là 1.27.
- Không có vị trí nào vượt λ = 2.0. Vì vậy **chọn tập biên nào cũng không phanh được**, và việc bài gốc không nói rõ tập biên (wait hay newline) không ảnh hưởng đến kết quả ở đây.

Nguyên nhân là chuỗi suy nghĩ trên GSM8K ngắn so với cửa sổ:
- Độ dài chuỗi suy nghĩ của `standard`: P10 = 109, P25 = 148, trung vị = 431, P75 = 951 token.
- Chỉ 96/200 câu suy nghĩ đủ 512 token.
- Với các câu đủ dài, cửa sổ 512 token phủ gần hết chuỗi. Trung bình trong cửa sổ vì thế xấp xỉ trung bình toàn chuỗi, nên RPDI ≈ 1.

Điều này phù hợp với chính bài gốc: Hình 5 báo cáo tỷ lệ kích hoạt trên GSM8K chỉ **2.0%** (đo trên các mô hình lớn hơn).

Để tham khảo, bảng dưới đây đếm số câu có RPDI vượt ngưỡng ở ít nhất một vị trí. Đây là cận trên của số câu sẽ bị phanh. **Không dùng bảng này để chỉnh tham số**, vì id 100–299 nằm trong tập held-out.

| W | Số câu suy nghĩ ≥ W token | RPDI max > 1.3 | > 1.5 | > 2.0 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 96 | 15 | 1 | 0 |
| 256 | 111 | 51 | 20 | 1 |
| 128 | 163 | 86 | 61 | 14 |

### 4.3. So sánh chính

| Target | Reference | Acc T | Acc R | Δ (T−R) | KTC 95% | KTC 90% | McNemar p | Token T/R | Kết luận (δ = 3) |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: | ---: | :--- |
| `legacy@768` | `standard@768` | 76.0 | 55.0 | **+21.0** | [+14.5, +27.5] | – | 3.3e-09 | 0.90 | Hơn |
| `legacy@768` | `standard@16384` | 76.0 | 82.0 | **−6.0** | [−11.5, −1.0] | [−10.5, −2.0] | 0.036 | 0.20 | Kém hơn, chưa chứng minh trong δ |
| `legacy@768` | `paper_wait@16384` | 76.0 | 80.0 | −4.0 | [−9.5, +1.0] | [−8.5, +0.0] | 0.185 | 0.22 | Chưa kết luận |
| `legacy@768` | `paper_newline@16384` | 76.0 | 81.5 | −5.5 | [−10.5, −0.5] | [−10.0, −1.5] | 0.052 | 0.22 | Kém hơn, chưa chứng minh trong δ |
| `legacy@1024` | `standard@16384` | 80.5 | 82.0 | −1.5 | [−6.0, +3.0] | [−5.5, +2.5] | 0.68 | 0.21 | Chưa kết luận |
| `legacy@16384` | `standard@16384` | 82.0 | 82.0 | 0.0 | [−4.5, +4.5] | [−4.0, +3.5] | 1.0 | 0.36 | Chưa kết luận |

Ba dòng tham chiếu ở 16384 token (`standard`, `paper_wait`, `paper_newline`) **là cùng một thuật toán**, vì chế độ `paper` không phanh. Các mức 80.0%, 81.5% và 82.0% chỉ khác nhau do nhiễu, nên không được coi là ba bằng chứng độc lập. Tham chiếu tự nhiên nhất là `standard@16384`.

### 4.4. Ba góc nhìn cho RPDI-EE ở 768 token

| Góc nhìn | Kết quả |
| :--- | :--- |
| Cùng ngân sách (768 token) | +21.0 điểm so với Standard (76.0% so với 55.0%) |
| Cùng accuracy | Standard cần ngân sách **2048 token**, trung bình 890 token, để đạt 77.5% ≥ 76.0%. RPDI-EE chỉ dùng trung bình 537 token. Ở 1536 token, Standard mới đạt 74.0%. |
| So với 16384 token | Kém 6.0 điểm, dùng 20% số token (537 so với 2635). Tỷ lệ phục hồi = (76.0 − 55.0) / (82.0 − 55.0) = **78%**. |

### 4.5. Phân tích 20 câu RPDI-EE@768 sai nhưng Standard@16384 đúng

| Nhóm | Số câu | Đúng khi RPDI-EE có 16384 token |
| :--- | ---: | ---: |
| Đã phanh, nhưng lời giải bị cắt ở 768 token | 13 | 10 |
| Đã phanh, viết xong lời giải nhưng sai (phanh quá sớm) | 7 | 0 |
| Không phanh | 0 | – |

Chiều ngược lại, có 8 câu RPDI-EE@768 đúng mà Standard@16384 sai.

**Phần lớn khoảng cách 6 điểm là do pha trả lời bị cắt, không phải do tín hiệu phanh.** Trong 13 câu bị cắt, 10 câu sẽ đúng nếu pha trả lời có thêm token.

### 4.6. Hành vi phanh của RPDI-EE (`legacy`)

- Phanh **128/200 câu (64%)**.
- Vị trí phanh (số token suy nghĩ): nhỏ nhất 37, P10 = 62, P25 = 77, **trung vị 113**, P75 = 161, P90 = 219, lớn nhất 544.
- Độ dài pha trả lời sau khi phanh, đo ở ngân sách 16384 token: P10 = 233, P25 = 348, **trung vị 521**, P75 = 679, P90 = 867, P95 = 1280.

Như vậy, một câu phanh ở token 113 chỉ còn khoảng 650 token cho pha trả lời trong ngân sách 768, ít hơn mức P75. Sau khi bị ép đóng `</think>`, mô hình 1.5B vẫn tiếp tục suy luận ngay trong lời giải. Đây cũng là hiện tượng đã thấy trong báo cáo đối chứng 200 câu (Mục 4.3 của báo cáo đó).

### 4.7. Ở ngân sách lớn: lợi thế giảm dần

| Ngân sách | RPDI-EE | Standard | Chỉ RPDI-EE đúng | Chỉ Standard đúng |
| ---: | ---: | ---: | ---: | ---: |
| 768 | 76.0 | 55.0 | 48 | 6 |
| 1024 | 80.5 | 64.0 | 40 | 7 |
| 1536 | 81.5 | 74.0 | 23 | 8 |
| 2048 | 81.5 | 77.5 | 17 | 9 |
| 16384 | 82.0 | 82.0 | 10 | 10 |

Ở 16384 token, hai phương pháp hòa nhau về accuracy, nhưng RPDI-EE dùng ít hơn 64% token (959 so với 2635). Tách nhỏ ra:

| Nhóm câu (ở 16384 token) | Số câu | Chỉ RPDI-EE đúng | Chỉ Standard đúng | Token TB: Standard → RPDI-EE |
| :--- | ---: | ---: | ---: | :--- |
| Standard lặp đến hết 16384 token, không đóng `</think>` | 24 | 7 | 2 | 16384 → phần lớn kết thúc (20/24) |
| Cả hai phương pháp đều kết thúc | 175 | 3 | 7 | 749 → 546 (−27%) |
| RPDI-EE phanh rồi lặp trong pha trả lời (id 186) | 1 | 0 | 1 | – |

- **Phần lớn số token tiết kiệm được ở 16k đến từ việc phá vòng lặp.** Riêng 24 câu lặp của Standard đã chiếm 1966 trong 2635 token trung bình. Trên 24 câu này, RPDI-EE phanh cả 24 câu, 20 câu kết thúc được, và đúng 15 câu so với 10 câu của Standard.
- **Trên các câu bình thường, RPDI-EE tiết kiệm 27% token** nhưng kém hơn một chút: 85.1% so với 87.4%, tức 3 câu so với 7 câu, không có ý nghĩa thống kê. Trên nhóm này, pha suy nghĩ giảm từ 509 xuống 140 token, nhưng pha trả lời tăng từ 240 lên 405 token.

## 5. Thảo luận

### 5.1. Trả lời câu hỏi nghiên cứu

Ở 768 token, RPDI-EE **kém thiết lập 16k khoảng 6 điểm**. Kết quả này không tương đương với 16k, và cũng chưa chứng minh được là nằm trong biên δ = 3. Đổi lại, phương pháp chỉ dùng khoảng 1/5 số token và lấy lại được khoảng 78% phần accuracy bị mất khi hạ ngân sách.

Khoảng cách còn lại có hai nguồn:
1. Pha trả lời bị cắt: 13/20 câu.
2. Phanh quá sớm: 7/20 câu.

Nguồn thứ nhất có thể xử lý mà không cần thay đổi tín hiệu phanh, ví dụ chèn "Final Answer:" ngay sau `</think>`, hoặc dành riêng ngân sách cho pha trả lời.

### 5.2. Về cấu hình của bài gốc

Với W = 512 và λ = 2.0, RPDI-EE **không kích hoạt trên GSM8K với mô hình 1.5B, kể cả ở 16384 token**. Bài gốc chọn cấu hình này bằng ablation trên mô hình 32B, với ngân sách nhỏ nhất được thử là 8192 token. Cấu hình đó không chuyển được sang bài toán có chuỗi suy nghĩ ngắn.

Kết quả này cần được dùng để điều chỉnh nhận định ở Mục 4.8.3 của chương 4, rằng "nguyên nhân chính là ngân sách". Ở 16k, ngân sách không còn là giới hạn, nhưng W = 512 vẫn không kích hoạt. Như vậy, **độ dài chuỗi suy nghĩ của từng loại bài toán** cũng là nguyên nhân quan trọng.

Không suy ra được rằng phương pháp của bài gốc thất bại nói chung. Bài gốc báo cáo mức tăng trên các nhóm gồm cả AMC23, MATH500 và các bộ khó hơn, nơi chuỗi suy nghĩ dài hơn nhiều.

### 5.3. Đối chiếu mốc 16k

Trên 200 câu này, Standard ở 16384 token đạt 82.0% với trung vị độ dài chuỗi suy nghĩ là 431 token. Bài DEER báo cáo Vanilla 1.5B trên GSM8K đạt 76.1% với trung bình 1617 token. Hai con số này không so trực tiếp được vì:
- DEER chạy trên toàn bộ 1319 câu.
- DEER dùng chat template và system prompt "put your final answer within \boxed{}".

Ước lượng sơ bộ trong Mục 4.8.4 của chương 4 dựa trên mốc 76.1% của DEER, nên **cần thay bằng số đo được ở báo cáo này**.

Ngoài ra, 12% số lượt Standard lặp đến hết ngân sách. Tỷ lệ này có thể liên quan đến prompt thô không có system prompt, nhưng chưa được kiểm tra.

### 5.4. Những điều chưa được kết luận

- **RPDI-EE tốt hơn cắt cứng:** chưa có bằng chứng. Thí nghiệm này không có đối chứng cắt cứng. Ở 768 token với HF, cắt cứng N = 130 hòa với RPDI-EE.
- **Số token tiết kiệm được ở 16k là nhờ tín hiệu entropy:** chưa chắc. Phần lớn đến từ việc phá vòng lặp, và một luật cắt cứng đơn giản có thể cũng làm được điều này.
- **RPDI-EE ở 1024 token hoặc 16384 token tương đương Standard ở 16k:** KTC còn quá rộng so với δ.

## 6. Hạn chế

1. **Cỡ mẫu 200 câu.** KTC 95% của chênh lệch rộng khoảng ±5 điểm, lớn hơn δ = 3.
2. **Nhiễu của vLLM khoảng ±2 điểm** (Mục 3.3), gần bằng δ.
3. **Chưa có đối chứng cắt cứng và phanh ngẫu nhiên** trong pipeline vLLM.
4. **Bước kiểm tra so với HF không thực hiện được như thiết kế**, vì hai bên khác kiểu số. Độ tin cậy của pipeline hiện dựa trên các kiểm tra nội bộ ở Mục 3.2.
5. **Biên δ = 3 điểm** là giá trị đã truyền vào `--delta` khi phân tích. Cần ghi rõ trong luận văn rằng δ được chốt với giáo viên hướng dẫn trước khi xem kết quả.
6. **Chấm điểm trên lượt chưa kết thúc.** 10/24 lượt Standard lặp đến hết 16384 token vẫn được chấm đúng, vì bộ trích đáp án lấy được số từ phần suy nghĩ. Quy tắc chấm này giống các thí nghiệm HF trước đây.
7. **Một mô hình, một tập dữ liệu, giải mã greedy, fp16 trên T4.**

## 7. Đề xuất bước tiếp theo

| Ưu tiên | Việc | Mục đích |
| :-: | :--- | :--- |
| 1 | Thêm cắt cứng N = 130 (và phanh ngẫu nhiên) vào pipeline vLLM, chạy trên cùng 200 câu | Kiểm tra xem lợi thế ở mọi ngân sách, kể cả việc phá vòng lặp ở 16k, có đến từ tín hiệu entropy hay không |
| 2 | Chèn "Final Answer:" sau `</think>` khi phanh | Giảm 13/20 câu bị cắt ở pha trả lời |
| 3 | Mở rộng lên 605 câu held-out | Thu hẹp KTC xuống khoảng ±3 điểm |
| 4 | Thay bước kiểm tra pipeline: chạy lại HF bằng fp16 trên khoảng 50 câu ở 768 token, hoặc ghi các kiểm tra nội bộ ở Mục 3.2 làm bằng chứng chính thức | Hoàn tất bước kiểm chứng pipeline |
| 5 | Cập nhật chương 4: kết quả ở Mục 4.8.4, nhận định "nguyên nhân là ngân sách" ở Mục 4.8.3, và ước lượng sơ bộ dựa trên DEER | Đồng bộ luận văn với số đo thật |

## 8. Cách viết vào luận văn

> Trên 200 câu GSM8K held-out, RPDI-EE (W = 16, λ = 1.5) ở ngân sách 768 token đạt 76.0%, so với 55.0% của sinh tự do cùng ngân sách (+21.0 điểm, McNemar p ≈ 3×10⁻⁹) và 82.0% của sinh tự do ở 16384 token (−6.0 điểm, KTC 95% [−11.5, −1.0]). Như vậy, với khoảng 20% số token, phương pháp lấy lại được khoảng 78% phần accuracy bị mất khi hạ ngân sách từ 16384 xuống 768 token. Sinh tự do cần ngân sách 2048 token để đạt mức accuracy tương đương. Phần lớn khoảng cách còn lại (13/20 câu) đến từ việc lời giải bị cắt sau khi phanh, chứ không phải từ tín hiệu phanh. Cấu hình gốc của bài báo (W = 512, λ = 2.0) không kích hoạt lần nào trên GSM8K, kể cả ở 16384 token, vì chuỗi suy nghĩ của mô hình 1.5B (trung vị 431 token) quá ngắn so với cửa sổ 512 token.
