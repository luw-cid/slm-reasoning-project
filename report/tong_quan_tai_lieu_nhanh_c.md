# Tổng quan tài liệu Nhánh C (Inference-time) và khoảng trống nghiên cứu

> Phạm vi: các phương pháp can thiệp lúc sinh (không huấn luyện) để giảm overthinking và dừng sớm, cùng các nghiên cứu phân tích liên quan trực tiếp tới RPDI-EE trên DeepSeek-R1-Distill-Qwen-1.5B, GSM8K, greedy, ngân sách 768 token.
>
> Cách xác minh: tất cả 30 bài đã được đối chiếu với trang arXiv abs/html (hoặc Nature, ACL Anthology, model card HF) ngày 18/10/2026. Ô ghi "chưa xác minh" là thông tin chưa tìm thấy trong bản đã đọc. Số liệu lấy qua trang HTML, **trước khi đưa vào luận văn cần đối chiếu lại với PDF**, nhất là các con số được dùng để so sánh trực tiếp.

## ⚠️ Đính chính quan trọng

- **Tác giả đầu của bài RPDI gốc (arXiv:2603.14251) là Weixin Guan, không phải Li.** Cách trích đúng là "Guan et al., 2026". Hiện `context.txt` và các chương báo cáo đang ghi "Li et al., 2026", cần sửa lại.
- DEER trên chính mô hình 1.5B, GSM8K, ngân sách 16k: độ chính xác **giảm** từ 76.1% (1617 token) xuống 74.7% (984 token), theo bảng của bản v3. Một lần kiểm tra khác chỉ thấy số 1.5B trong hình, nên cần mở PDF để chắc chắn.
- Model card của DeepSeek-R1-Distill khuyến nghị temperature 0.6 (khoảng 0.5–0.7) để tránh lặp và max length 32,768. Thiết lập greedy + 768 token của đề tài **lệch có chủ đích** khỏi khuyến nghị này, nên phải nêu rõ trong phần Thiết lập thực nghiệm.

---

## 1. Phân nhóm 30 bài báo

| Nhóm                                           | Số bài | Ý tưởng chung                                                                  |
| :---------------------------------------------- | :------: | :-------------------------------------------------------------------------------- |
| A. Dừng sớm thích nghi (adaptive early exit) |    9    | Theo dõi một tín hiệu trong lúc sinh, đến ngưỡng thì chèn `</think>` |
| B. Kiểm soát ngân sách / cắt cố định    |    6    | Cắt hoặc rút gọn suy nghĩ theo số token cố định hoặc theo prompt        |
| C. Can thiệp giải mã / hidden state          |    6    | Sửa logits hoặc hidden state để định hướng mạch suy luận                |
| D. Phân tích hiện tượng overthinking       |    5    | Đo quan hệ độ dài và độ đúng, đặc thù của mô hình nhỏ            |
| E. Test-time scaling và độ bất định       |    4    | Dùng thêm compute (nhiều mẫu, PRM) hoặc độ tự tin để chọn đáp án    |

## 2. Bảng tổng hợp

Viết tắt: **TF** = training-free; **≤2B** = có thử mô hình ≤2B tham số làm mô hình suy luận; **1.5B-R1** = có thử chính DeepSeek-R1-Distill-Qwen-1.5B.

### Nhóm A: Dừng sớm thích nghi

| # | Bài báo                                                                                                                            | Tín hiệu dừng                                                                                                     | TF / chi phí thêm                                    | Mô hình nhỏ                                                      | Ngân sách, giải mã               |           GSM8K           | Kết quả chính                                                                           | Hạn chế liên quan                                                                                                           |
| :-: | :----------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- | :------------------------------------------------------------------ | :----------------------------------- | :------------------------: | :----------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| 1 | **RPDI-EE**: *Mitigating Overthinking in LRLMs via Reasoning Path Deviation Monitoring*. Guan et al., 2026, arXiv:2603.14251 | RPDI = LTF/GTF (entropy cửa sổ W=512 / entropy toàn cục), kiểm tra tại token chuyển ý, dừng khi > λ=2.0    | TF, gần như không tốn thêm                        | **1.5B-R1**                                                   | 16k (32k cho Qwen3-Thinking), greedy | Có (gộp vào Math Easy) | 1.5B: Math Easy 65.7→69.4; độ dài 9738→8934 token (≈−8%)                            | W, λ cố định; trên 1.5B chỉ so với Vanilla và DEER; không có baseline cắt cứng; không có kiểm định thống kê |
| 2 | **DEER**: *Dynamic Early Exit in Reasoning Models*. Yang et al., 2025, arXiv:2504.15895                                      | Tại từ "Wait": bắt mô hình viết thử đáp án, dừng nếu độ tự tin (trung bình nhân xác suất) ≥ 0.95 | TF, phải sinh thêm đáp án thử (tăng độ trễ)  | **1.5B-R1**, Qwen3-1.7B                                       | 16k, greedy                          |            Có            | Ngắn hơn 19–80%. 1.5B GSM8K: 76.1→74.7 (giảm)                                         | λ cố định; ít khi dừng ở bài khó; chỉ thử ngân sách dài                                                          |
| 3 | **Dynasor-CoT**: *Efficiently Scaling LLM Reasoning with Certaindex*. Fu et al., 2024, arXiv:2412.20993                      | Cứ mỗi 32–320 token chèn câu hỏi thăm dò đáp án, dừng khi các đáp án nhất quán                     | TF, tốn token cho thăm dò                           | Không (≥7B)                                                       | 16k                                  | Chỉ ở phần tương quan | Giảm 11–29% token với cùng độ chính xác                                            | Ngưỡng phải hiệu chỉnh trên dữ liệu có nhãn, theo từng cỡ mô hình                                                |
| 4 | *Answer Convergence as a Signal for Early Stopping in Reasoning*. Liu & Wang, 2025, arXiv:2506.02536                               | (a) k=10 đáp án liên tiếp giống nhau; (b) cộng thêm vào logit của `</think>`; (c) probe có giám sát   | (a)(b) TF; (c) cần huấn luyện probe                 | Không (≥7B)                                                       | chưa xác minh                      |            Có            | Mô hình hội tụ đáp án sau ≈60% số bước; QwQ GSM8K 755→419 token (−0.2 điểm) | Hội tụ không đồng nghĩa với đúng; kém ổn định ở bài khó                                                        |
| 5 | **FlashThink**: *An Early Exit Method For Efficient Reasoning*. Jiang et al., 2025, arXiv:2505.13949                         | Một LLM kiểm định 7B ngoài trả lời "đã đủ suy luận chưa?"                                               | TF, cần thêm một mô hình 7B                       | Không, tác giả chủ động tránh mô hình nhỏ                 | 32k, T=0.7                           |            Có            | Giảm ≈77% độ dài suy nghĩ, độ chính xác giữ nguyên                             | Phụ thuộc mô hình kiểm định; chỉ thử mô hình lớn                                                                   |
| 6 | **EAT**: *Entropy After `</Think>` for reasoning model early exiting*. Wang et al., 2025, arXiv:2509.26522                 | Entropy của token kế tiếp sau khi chèn thử `</think>`; dừng khi phương sai EMA < δ                        | TF, mỗi lần kiểm tra tốn 1 lượt forward          | Không (1.5B chỉ dùng làm mô hình proxy)                       | 10k, T=0.6                           |           Không           | −12–22% token, không giảm độ chính xác; rẻ hơn DEER ≈5 lần                     | Bài không giải được vẫn chạy hết ngân sách                                                                          |
| 7 | *Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification*. Zhang et al., 2025, arXiv:2504.05419      | Probe trên hidden state dự đoán đáp án tạm thời đã đúng chưa                                           | **Không TF** (probe học từ nhãn của Gemini) | Có probe trên 1.5B-R1, nhưng dừng sớm chỉ thử trên Llama-8B | 10k                                  |            Có            | MATH: −24% token, giữ 88.2% (cắt tĩnh cùng mức token chỉ ≈82.5%)                   | Dừng sớm chỉ thử 1 mô hình, 1 tập; không báo phương sai                                                             |
| 8 | **REFRAIN**: *Stop When Enough: Adaptive Early-Stopping for CoT Reasoning*. Sun et al., 2025, arXiv:2510.10103 (ACL 2026)    | Cụm từ phản tư + độ trùng ngữ nghĩa với các bước trước; ngưỡng τ chọn bằng bandit UCB            | TF, cần mô hình embedding nhỏ                      | Không (Qwen3-8B, gpt-oss-20B)                                      | 16k, sampling                        |            Có            | −20–55% token; Qwen3-8B GSM8K 94.54 vs 94.24                                             | Có thể cắt mất lần tự sửa sai muộn; 1 seed, không kiểm định                                                        |
| 9 | **DeepConf**: *Deep Think with Confidence*. Fu et al., 2025, arXiv:2508.15260                                                | Độ tự tin nhóm (cửa sổ trượt 2048 token); chế độ online dừng các vết suy luận yếu                    | TF, cần nhiều vết (≤512) và 16 vết khởi động  | Không (≥8B)                                                       | 32k–130k, T=0.6                     |           Không           | Online giảm tới 84.7% token; GPT-OSS-120B AIME25 99.9%                                   | Nhiều mẫu, không hợp với greedy 1 vết; sai khi mô hình tự tin mà sai                                                 |

### Nhóm B: Kiểm soát ngân sách / cắt cố định

| # | Bài báo                                                                                                            | Cơ chế                                                                                                   | TF                                                                              | Mô hình nhỏ                             | Ngân sách                        | GSM8K | Kết quả chính                                                  | Hạn chế                                                                            |
| :-: | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------ | :----------------------------------------- | :--------------------------------- | :----: | :---------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| 10 | **s1**: *Simple test-time scaling*. Muennighoff et al., 2025, arXiv:2501.19393                               | Budget forcing: hết ngân sách thì chèn delimiter + "Final Answer:"; muốn kéo dài thì chèn "Wait" | Phần giải mã là TF, mô hình s1-32B có SFT                                | Không                                     | ≈30k, greedy                      | Không | AIME24 50→57%                                                    | Là **baseline cắt cứng chuẩn** của lĩnh vực                             |
| 11 | **NoThinking**: *Reasoning Models Can Be Effective Without Thinking*. Ma et al., 2025, arXiv:2504.09858      | Điền sẵn khối suy nghĩ rỗng; so với Thinking ở cùng ngân sách                                   | TF                                                                              | Không (32B; phụ lục có 7B/14B)         | **≈700–3500 token**        | Không | AMC23 ở 700 token: 51.3 (NoThinking) vs 28.9 (Thinking)          | pass@1 thường thua Thinking; chỉ họ R1-distill                                   |
| 12 | **ThinkLess**. Li et al., 2025, arXiv:2505.15684                                                               | Chèn `</think>` ở vị trí sớm cố định + lệnh điều chỉnh định dạng đáp án                | TF                                                                              | Không (7–14B)                            | 8k                                 |  Có  | Ngang CoT đầy đủ, tiết kiệm nhiều thời gian               | Tác giả tự nêu: cắt cố định, không thích nghi                              |
| 13 | **Elastic Reasoning**: *Scalable Chain of Thoughts via Elastic Reasoning*. Xu et al., 2025, arXiv:2505.05315 | Ngân sách riêng cho pha suy nghĩ và pha lời giải (cắt suy nghĩ, ép sang lời giải)              | Bản E1 dùng GRPO; riêng chia ngân sách thì TF và thắng cắt thô và s1 | Gốc từ 1.5B-R1 (qua DeepScaleR)          | 0.5K–3K suy nghĩ + 1K lời giải | Không | E1-Math-1.5B: AIME24 35.0, ít hơn 32.1% token                   | Giả định suy luận tách rõ 2 pha                                                |
| 14 | **TALE**: *Token-Budget-Aware LLM Reasoning*. Han et al., 2024, arXiv:2412.18547 (Findings ACL 2025)         | Ghi ngân sách token vào prompt                                                                          | TALE-EP là TF                                                                  | Không                                     | Ngân sách theo từng câu        |  Có  | −68% token, giảm <3 điểm                                      | "Token elasticity": ngân sách quá nhỏ thì mô hình lờ đi và viết dài hơn |
| 15 | **Chain of Draft**. Xu et al., 2025, arXiv:2502.18600                                                          | Prompt bắt viết nháp ≈5 từ mỗi bước                                                                | TF                                                                              | Qwen2.5-1.5B/3B-Instruct (không phải R1) | chưa xác minh                    |  Có  | GPT-4o GSM8K: 91.1% với 43.9 token (CoT: 95.4% với 205.1 token) | **Mô hình nhỏ giảm mạnh**: Qwen-1.5B 32.5→24.2%                          |

### Nhóm C: Can thiệp giải mã / hidden state

| # | Bài báo                                                                                                                                            | Cơ chế                                                                                                                        | TF                                                    | Mô hình nhỏ    | Ngân sách, giải mã                |        GSM8K        | Kết quả chính                                                        | Hạn chế                                                                                                  |
| :-: | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------- | :---------------- | :------------------------------------ | :------------------: | :---------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| 16 | **SEAL**: *Steerable Reasoning Calibration of LLMs for Free*. Chen et al., 2025, arXiv:2504.07986                                            | Vector điều hướng (lấy offline từ 1000 mẫu MATH) cộng vào hidden state tại "\n\n" để giảm phản tư và chuyển ý | TF (không cập nhật trọng số)                     | **1.5B-R1** | greedy, max len chưa xác minh       | Có (đủ 1319 câu) | **1.5B GSM8K: 74.1→82.0, token 2015→999 (−50%)**; 7B chỉ +0.4 | Phân loại ý bằng từ khóa; không quét theo ngân sách. **Baseline gần nhất về mô hình** |
| 17 | **Overclocking LLM Reasoning**. Eisenstadt et al., 2025, arXiv:2506.07240                                                                      | Probe tuyến tính "thinking progress vector", dịch hidden state để suy nghĩ "chạy nhanh" hơn                             | Mô hình không huấn luyện, probe có huấn luyện | Không (8B, 32B)  | **GSM8K ở 256/512/1024 token** |    Có (300 câu)    | 32B GSM8K@256: 12→36 câu đúng                                       | **Gần nhất về ngân sách hẹp** nhưng không có mô hình ≤2B                                 |
| 18 | *Effectively Controlling Reasoning Models through Thinking Intervention*. Wu et al., 2025, arXiv:2503.24370                                        | Chèn hoặc viết lại token trong pha suy nghĩ                                                                                | TF                                                    | Không (≥7B)     | chưa xác minh                       |        Không        | R1-Qwen-14B: −24% token, −0.8 điểm                                  | Can thiệp càng muộn càng kém hiệu quả                                                               |
| 19 | **NoWait**: *Wait, We Don't Need to "Wait"!*. Wang et al., 2025, arXiv:2506.08343                                                            | Logit processor cấm ≈17 từ phản tư                                                                                         | TF                                                    | Không (≥4B)     | 32k                                   |        Không        | CoT ngắn hơn 27–51%                                                  | **Mô hình distill nhạy hơn nhiều**: AIME25 giảm >12 điểm                                     |
| 20 | **TIP** / *Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs*. Wang et al., 2025, arXiv:2501.18585                       | Phạt logit các token chuyển ý ("alternatively")                                                                             | TF                                                    | Không (≥32B)    | T=0.7                                 |        Không        | Câu sai dùng nhiều hơn ≈225% token; QwQ AIME24 38.3→44.1          | Đi theo hướng ngược với dừng sớm: giữ mô hình bám lâu hơn vào một ý                       |
| 21 | **EM-INF**: *The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning*. Agarwal et al., 2025, arXiv:2505.15134 (NeurIPS 2025) | Tối ưu logit tại mỗi bước để giảm entropy                                                                              | Biến thể EM-INF là TF                              | Không (≥7B)     | T=0.1                                 |        Không        | Ngang self-consistency (N=4) với ≈1/3 thời gian                      | Thất bại khi độ tự tin không khớp với độ đúng                                                  |

### Nhóm D: Phân tích hiện tượng overthinking

| # | Bài báo                                                                                               | Nội dung                                                                 | Mô hình nhỏ        |      GSM8K      | Phát hiện dùng được cho luận văn                                                                                                                                               |
| :-: | :------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------ | :-------------------- | :--------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 22 | *Do NOT Think That Much for 2+3=?* Chen et al., 2024, arXiv:2412.21187                                | Định nghĩa overthinking, độ đo outcome/process efficiency           | Không                |       Có       | MATH500 Level 1: outcome efficiency <50%. Giảm bằng SimPO (cần huấn luyện)                                                                                                        |
| 23 | *When More is Less: Understanding CoT Length in LLMs*. Wu et al., 2025, arXiv:2502.07266              | Độ chính xác theo độ dài CoT có dạng chữ U ngược              | Qwen2.5-1.5B-Instruct |      Không      | **Mô hình nhỏ cần CoT dài hơn** (độ dài tối ưu ≈14 bước ở 1.5B, ≈4 bước ở 72B). Đây là phản biện cho việc cắt mạnh trên 1.5B                         |
| 24 | *Between Underthinking and Overthinking*. Su et al., 2025, arXiv:2505.00127                           | Quan hệ độ dài và độ đúng ở cấp mẫu và cấp câu             | **1.5B-R1**     | Có (tập train) | 1.5B GSM8K: câu đúng trung bình 666.7 token, câu sai 842.5 token. Với >60% câu, mẫu ngắn nhất đã đúng. Mô hình nghĩ quá nhiều ở câu dễ và quá ít ở câu khó |
| 25 | *Missing Premise exacerbates Overthinking*. Fan et al., 2025, arXiv:2504.06514                        | Câu hỏi thiếu dữ kiện khiến mô hình suy luận dài gấp 2–4 lần | Không                |    MiP-GSM8K    | R1 dùng ≈3246 token cho MiP-GSM8K so với ≈1226 cho GSM8K thường                                                                                                                  |
| 26 | *Small Models Struggle to Learn from Strong Reasoners*. Li et al., 2025, arXiv:2502.12143 (ICML 2025) | "Learnability gap": mô hình ≤3B học CoT dài kém                     | Qwen2.5-0.5B/1.5B     |       Có       | Lý do cho việc SLM không xử lý tốt chuỗi suy luận dài                                                                                                                         |

### Nhóm E: Test-time scaling và độ bất định

| # | Bài báo                                                                           | Nội dung                                                           | Chi phí            | Mô hình nhỏ              | GSM8K | Kết quả                                |
| :-: | :---------------------------------------------------------------------------------- | :------------------------------------------------------------------ | :------------------ | :-------------------------- | :----: | :--------------------------------------- |
| 27 | *Scaling LLM Test-Time Compute Optimally…*. Snell et al., 2024, arXiv:2408.03314 | Phân bổ compute theo độ khó (PRM search, revision)             | Tới 256 mẫu + PRM | Không                      | Không | Hiệu quả gấp >4 lần best-of-N        |
| 28 | *Can 1B LLM Surpass 405B LLM?* Liu et al., 2025, arXiv:2502.06703                 | Chiến lược TTS tối ưu phụ thuộc mô hình, PRM và độ khó | N tới 512 + PRM    | **1.5B-R1**, Llama-1B | Không | 1.5B-R1 + TTS: MATH-500 91.6             |
| 29 | *Self-Consistency*. Wang et al., ICLR 2023, arXiv:2203.11171                      | Bỏ phiếu đa số trên nhiều chuỗi CoT                          | N mẫu              | Không                      |  Có  | PaLM-540B GSM8K +17.9                    |
| 30 | *Self-Certainty*. Kang et al., 2025, arXiv:2502.18581 (NeurIPS 2025)              | Độ tự tin KL(U‖p) để chọn best-of-N                          | N mẫu              | Không                      |  Có  | Borda N=64: 91.07 vs SC 90.99 (Llama-8B) |

### Tài liệu nền bổ sung (nên trích, không tính trong 30 bài)

- **DeepSeek-R1** (DeepSeek-AI, arXiv:2501.12948; Nature 645, 2025): nguồn của mô hình 1.5B. Model card báo MATH-500 83.9 và AIME24 28.9, không báo GSM8K.
- **Sui et al.**, *Stop Overthinking: A Survey on Efficient Reasoning for LLMs*, arXiv:2503.16419 (TMLR 2025): taxonomy. RPDI-EE thuộc nhóm "reasoning-output-based".
- **L1** (Aggarwal & Welleck, arXiv:2503.04697, COLM 2025): RL kiểm soát độ dài trên 1.5B, cầu nối sang Nhánh B.
- **EAGer** (Scalena et al., arXiv:2510.11170): rẽ nhánh tại token có entropy cao, cùng họ tín hiệu entropy.
- **Semantic entropy** (Farquhar et al., Nature 630, 2024): nền tảng về độ bất định dựa trên entropy.
- **Speculative Thinking** (Yang et al., arXiv:2504.12329): mô hình 32B hỗ trợ 1.5B-R1 lúc suy luận.
- Đã xác nhận có tồn tại nhưng **chưa đọc chi tiết**: arXiv:2604.06787 (*When Is Thinking Enough?*), arXiv:2601.22617 (ngân sách suy nghĩ thích nghi theo entropy), ASAG arXiv:2606.15070, survey arXiv:2508.02120. LEASH (arXiv:2511.04654) đã chuyển xuống mục 6.

---

## 3. Bản đồ định vị

| Tiêu chí                                           | RPDI gốc |         DEER         |         SEAL         |   Overclocking   |   NoThinking   | EAT | REFRAIN | **Đề tài** |
| :--------------------------------------------------- | :-------: | :-------------------: | :-------------------: | :---------------: | :------------: | :-: | :-----: | :-----------------: |
| Training-free, 1 vết suy luận                      |    ✓    | ✓ (+ đáp án thử) | ✓ (+ vector offline) |    ✗ (probe)    |       ✓       | ✓ |   ✓   |         ✓         |
| Mô hình suy luận ≤2B                             |    ✓    |          ✓          |          ✓          |        ✗        |       ✗       | ✗ |   ✗   |         ✓         |
| Có GSM8K                                            | ✓ (gộp) |          ✓          |          ✓          |        ✓        |       ✗       | ✗ |   ✓   |         ✓         |
| Greedy                                               |    ✓    |          ✓          |          ✓          |         ?         |       ?       | ✗ |   ✗   |         ✓         |
| Ngân sách ≤1k token                               |    ✗    |          ✗          |          ✗          |        ✓        |   ✓ (≈700)   | ✗ |   ✗   |      ✓ (768)      |
| Baseline cắt cứng / ngẫu nhiên                   |    ✗    |          ✗          |    1 điểm (3500)    | baseline prompt/T | budget forcing | ✗ |   ✗   |         ✓         |
| Kiểm định thống kê (KTC, McNemar)               |    ✗    |          ✗          |          ✗          |        ✗        |       ✗       | ✗ |   ✗   |         ✓         |
| Tách cơ chế (câu bị cắt / câu tự kết thúc) |    ✗    |          ✗          |          ✗          |        ✗        |       ✗       | ✗ |   ✗   |         ✓         |

**Kết luận định vị:** trong 30 bài, **không bài nào** kết hợp đủ 4 yếu tố: phương pháp dừng sớm thích nghi, training-free, mô hình ≤2B, ngân sách dưới 1k token. Overclocking và NoThinking gần nhất về ngân sách nhưng dùng mô hình 8B–32B. RPDI, DEER và SEAL gần nhất về mô hình nhưng đều chạy ở ngân sách dài.

---

## 4. Các khoảng trống (gap) của Nhánh C

Mỗi gap ghi kèm bằng chứng từ tài liệu và tình trạng hiện tại của đề tài.

### Gap 1: Dừng sớm thích nghi chưa được đánh giá ở ngân sách cực hẹp trên SLM

- **Bằng chứng:** cả 9 phương pháp nhóm A đều chạy ở 8k–130k token. Chỉ NoThinking (≈700 token, 32B) và Overclocking (256–1024 token, 8B/32B) đánh giá ngân sách hẹp, và cả hai không phải dừng sớm bằng tín hiệu bất định trên SLM.
- **Vì sao quan trọng:** ở 16k, dừng sớm là bài toán *chất lượng suy luận* (RPDI gốc chỉ +3.7, giảm ≈8% token). Ở 768 token, nó trở thành bài toán *giải cứu câu bị cạn ngân sách*, một cơ chế khác hẳn.
- **Đề tài:** ✅ đã lấp. Trên 605 câu held-out: 52.23% → 71.57%; toàn bộ mức tăng đến từ nhóm baseline cạn ngân sách (21.18% → 63.19%).

### Gap 2: Thiếu đối chứng với cắt cứng và phanh ngẫu nhiên, tức chưa chứng minh tín hiệu thích nghi có giá trị

- **Bằng chứng:** RPDI gốc, DEER, Dynasor, EAT và REFRAIN đều không so với cắt cố định. Trong khi đó ThinkLess (cắt cố định) và NoThinking (bỏ hẳn suy nghĩ) cho thấy cắt thô đã mạnh. Bài probe hidden state (#7) là một trong số ít so với "static early exit" và thấy chênh ≈5.7 điểm. Elastic Reasoning cho thấy riêng việc chia ngân sách suy nghĩ / lời giải (TF) đã thắng cắt thô.
- **Đề tài:** ⚠️ đây là **điểm yếu cần xử lý nhất**. HardTrunc(N=128) đạt 75% so với 78% của RPDI-EE trong mô phỏng, chênh 3 câu, chưa có ý nghĩa thống kê.
- **Đề xuất:**
  - Chạy HardTrunc, phanh ngẫu nhiên khớp phân bố vị trí phanh, và chia ngân sách kiểu Elastic (ví dụ 512 suy nghĩ + 256 lời giải) trên **605 câu held-out**, rồi kiểm định McNemar so với RPDI-EE.
  - Thêm NoThinking (W→0) làm điểm cực.
  - Nếu RPDI-EE không thắng có ý nghĩa, đóng góp nên được viết lại là *"ở ngân sách hẹp, lợi ích chủ yếu đến từ việc giữ lại ngân sách cho pha trả lời"*. Đây vẫn là một phát hiện có giá trị.

### Gap 3: Siêu tham số được coi là cố định, chưa ai nghiên cứu chúng dịch chuyển theo ngân sách

- **Bằng chứng:** DEER λ=0.95, RPDI W=512/λ=2.0, EAT δ và Dynasor đều cố định (Dynasor chỉ hiệu chỉnh theo cỡ mô hình). Chỉ REFRAIN chọn ngưỡng thích nghi bằng bandit.
- **Đề tài:** ✅ một phần. W=512/λ=2.0 → W=16/λ=1.5. Tuy vậy cấu hình tốt nhất đang nằm ở **góc lưới quét**.
- **Đề xuất:**
  - Quét thêm W ∈ {4, 8}, λ ∈ {1.1, 1.25} trên `gsm8k_val_200.json`.
  - Quan trọng hơn: quét **theo ngân sách** {512, 768, 1024, 2048} để vẽ đường W*(B), λ*(B). Chưa bài nào có kết quả này.

### Gap 4: Tín hiệu dừng chưa được kiểm chứng trên mô hình nhỏ, nơi độ bất định có thể kém hiệu chỉnh

- **Bằng chứng:**
  - DEER trên 1.5B-R1 GSM8K làm *giảm* độ chính xác (76.1→74.7).
  - NoWait: mô hình distill nhạy hơn nhiều.
  - Chain of Draft: Qwen-1.5B giảm 8.3 điểm.
  - Wu et al.: mô hình nhỏ cần CoT dài hơn.
  - DeepConf, EM-INF: thất bại khi mô hình "tự tin mà sai".
  - Không bài nào phân tích *các trường hợp phanh nhầm* trên SLM.
- **Đề tài:** ⚠️ có 24 câu bị phanh nhầm (baseline đúng nhưng RPDI-EE sai) và nhóm baseline tự kết thúc giảm nhẹ (80.44 → 79.18%), nhưng chưa phân tích định tính.
- **Đề xuất:** phân tích 24 câu này theo vị trí phanh, entropy tại điểm phanh và độ khó (số bước lời giải). Kiểm tra xem RPDI cao có tương ứng với "đã có đáp án đúng" (như #4, #7) hay với "đang rối" như giả thuyết của bài gốc.

### Gap 5: Thiếu tính chặt chẽ thống kê

- **Bằng chứng:** không bài nào trong nhóm A báo khoảng tin cậy hay kiểm định cặp. REFRAIN chỉ chạy 1 seed; bài probe không báo phương sai.
- **Đề tài:** ✅ đã có KTC 95% và McNemar trên tập held-out tách khỏi tập chọn tham số. Nên nêu đây là đóng góp phương pháp luận.

### Gap 6: Chưa tách cơ chế tác động theo trạng thái câu hỏi

- **Bằng chứng:** các bài chỉ báo độ chính xác và số token trung bình. Su et al. chỉ ra mô hình nghĩ quá nhiều ở câu dễ và quá ít ở câu khó, nhưng không gắn với phương pháp dừng sớm.
- **Đề tài:** ✅ đã tách theo "baseline cạn / tự kết thúc". Có thể mở rộng: tách theo độ khó (số bước lời giải GSM8K) và theo vị trí phanh tương đối.

### Gap 7: Chưa đo chi phí thực của tín hiệu dừng dưới ngân sách hẹp

- **Bằng chứng:** DEER, Dynasor và Answer Convergence phải sinh thêm đáp án thử, FlashThink cần thêm mô hình 7B, EAT tốn 1 lượt forward mỗi lần kiểm tra. Ở 768 token, chi phí thăm dò chiếm tỷ lệ lớn hơn nhiều so với ở 16k.
- **Đề xuất:** đo wall-clock và overhead token của RPDI-EE so với DEER (có thể cài DEER rút gọn) ở cùng ngân sách 768. Lợi thế "O(1), không cần thăm dò" của RPDI chỉ thành đóng góp khi có số đo.

### Gap 8: Pha trả lời sau khi phanh chưa được nghiên cứu

- **Bằng chứng:** chỉ ThinkLess (lệnh điều chỉnh định dạng) và s1 ("Final Answer:") xử lý văn bản chèn sau khi phanh. Elastic Reasoning dành riêng ngân sách cho lời giải.
- **Đề tài:** vẫn còn 21.56% câu cạn ngân sách sau RPDI-EE; đã sửa lỗi phanh trong pha trả lời.
- **Đề xuất:** phân tích 21.56% này (mô hình mở lại suy nghĩ? lời giải quá dài?) và thử chèn thêm "Final Answer:" hoặc giới hạn riêng cho pha trả lời.

### Gap 9: Tương tác giữa dừng sớm và huấn luyện RL (cầu nối sang Nhánh B)

- **Bằng chứng:** L1 và Elastic Reasoning huấn luyện mô hình tự kiểm soát độ dài trên 1.5B. Chưa bài nào hỏi *sau RLVR, phanh từ bên ngoài còn có ích không, hay hai cách thay thế cho nhau*.
- **Đề tài:** ✅ thiết kế 4 kịch bản trả lời đúng câu hỏi này. Lưu ý `max_completion_length=512` khi huấn luyện GRPO đã ngầm ép viết ngắn.

### Gap 10 (mối đe dọa tính hợp lệ): Greedy so với giải mã khuyến nghị

- **Bằng chứng:** model card R1-Distill khuyến nghị T=0.6 vì greedy dễ gây lặp. DEER, SEAL và RPDI gốc đều greedy, nhưng chưa ai kiểm tra liệu lợi ích của dừng sớm có phần đến từ việc *cắt các vòng lặp do greedy gây ra* hay không.
- **Đề xuất:** chạy thêm một cấu hình T=0.6 (vài seed, tập con 200 câu) cho cả Standard và RPDI-EE. Nếu mức tăng thu hẹp mạnh, cần nêu trong phần hạn chế.

---

## 6. Bổ sung từ tìm kiếm Consensus (18/10/2026)

Sau khi lập danh sách 30 bài, tôi tìm thêm trên Consensus (Semantic Scholar/OpenAlex) với 9 truy vấn: early exit dựa trên entropy, overthinking ở SLM, dừng suy nghĩ không cần huấn luyện, vòng lặp khi giải mã greedy, budget forcing so với dừng thích nghi, và tra cứu trực tiếp RPDI, SEAL, LEASH, EntroCut. Bài account miễn phí chỉ trả về 3 kết quả mỗi truy vấn, nên **đây chưa phải quét toàn diện**.

> Lưu ý độ tin cậy: thông tin dưới đây **chỉ lấy từ abstract** trên Consensus. Mô hình, ngân sách và cách giải mã của các bài mới **chưa xác minh** trên toàn văn. Số trích dẫn là số của Consensus tại ngày tra cứu.

### 6.1. Bài mới, liên quan trực tiếp đến RPDI-EE (nên đọc toàn văn)

| # | Bài báo                                                                                                                                                                                                                                           | Vì sao quan trọng                                                                                                                                                                                                                                                                             | Gap liên quan |
| :-: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------: |
| 31 | **EntroCut**: *Entropy-Guided Adaptive Truncation for Efficient CoT Reasoning in Small-scale Large Reasoning Models*. Yan et al., 2026, ArXiv, 1 trích dẫn. [Link](https://consensus.app/papers/details/c9bd16343b5e571aa3ed81add3beabda/) | **Đối thủ trực tiếp nhất**: dừng TF bằng entropy, *nhắm đúng mô hình suy luận cỡ nhỏ*. Entropy ở các bước đầu phân biệt được lời giải đúng/sai; giảm tới 40% token; đề xuất chỉ số EPR (tiết kiệm token trên mỗi điểm chính xác mất đi) |      1, 4      |
| 32 | *Unveiling the Entropy Dynamics of Chain-of-Thought Reasoning* (CUSUM). Xu et al., 2026, ArXiv, 4 trích dẫn. [Link](https://consensus.app/papers/details/c17b91aad38f522ca8274992d5811634/)                                                      | Entropy của CoT có 2 pha (vùng bất định → vùng tự tin); phát hiện điểm chuyển bằng CUSUM. Vượt DEER +3.28 và Dynasor +4.36 điểm. Cùng họ tín hiệu entropy với RPDI nhưng có nền tảng thống kê (change-point detection)                                           |      2, 3      |
| 33 | **LEASH**: *Logit-Entropy Adaptive Stopping Heuristic for Efficient CoT Reasoning*. Quamar et al., 2025, ArXiv, 2 trích dẫn. [Link](https://consensus.app/papers/details/6b31427ddac751e9a62a304c5b952382/)                                | TF, dừng khi độ dốc entropy và biên top-logit cùng bão hòa; có GSM8K. **Giảm 30–35% token nhưng mất 10 điểm** so với CoT trên mô hình instruction-tuned. Đây là bằng chứng rằng dừng theo entropy có thể gây hại                                           |       4       |
| 34 | *Tracing Uncertainty in Language Model "Reasoning"*. Højer et al., 2026, ArXiv, 3 trích dẫn. [Link](https://consensus.app/papers/details/034c313180895a3e9a433691b92d768f/)                                                                     | Hình dạng đường bất định (độ dốc, độ tuyến tính) dự đoán đúng/sai với AUROC 0.807 trên GSM8K; **chỉ cần vài trăm token đầu** đã đạt 0.801. Ủng hộ việc ra quyết định sớm trong ngân sách 768                                                     |      1, 6      |
| 35 | ***Wait, Wait, Wait... Why Do Reasoning Models Loop?*** Pipis et al., 2025, ArXiv, 23 trích dẫn. [Link](https://consensus.app/papers/details/2d531fa01b00507692000dd6b43c2e58/)                                                            | Mô hình suy luận **lặp nhiều khi giải mã greedy / nhiệt độ thấp**; mô hình lớn ít lặp hơn; **mô hình học trò distill lặp nhiều dù mô hình thầy hiếm khi lặp**. Bằng chứng trực tiếp cho Gap 10                                                     |       10       |
| 36 | *Circular Reasoning: Understanding Self-Reinforcing Loops in LRMs*. Duan et al., 2026, ArXiv, 5 trích dẫn. [Link](https://consensus.app/papers/details/5ec94c0fdfee5b1d9b69246294e78b77/)                                                        | Vòng lặp suy luận có điểm khởi phát rõ ràng (lặp ngữ nghĩa xảy ra trước lặp văn bản); dùng CUSUM dự đoán sớm vòng lặp                                                                                                                                                 |     10, 4     |
| 37 | *Conformal Thinking: Risk Control for Reasoning on a Compute Budget*. Wang et al., 2026, ArXiv, 12 trích dẫn. [Link](https://consensus.app/papers/details/73c991eda0395c22aa4b7efc0d936017/)                                                     | Chọn ngưỡng dừng bằng **kiểm soát rủi ro phi tham số** trên tập validation; thêm ngưỡng dưới để dừng sớm các câu không giải được. Đây là cách chọn λ có cơ sở thống kê                                                                              |       3       |
| 38 | *Certainty-Guided Reasoning: A Dynamic Thinking Budget Approach*. Nogueira et al., 2025, ArXiv, 5 trích dẫn. [Link](https://consensus.app/papers/details/453c01999c875da69d2107e846c1afeb/)                                                      | Thăm dò độ chắc chắn định kỳ, dừng khi đạt ngưỡng; **chạy 64 seed**, là một trong số ít bài có tính chặt chẽ thống kê                                                                                                                                           |       5       |
| 39 | *Plan and Budget: Effective and Efficient Test-Time Scaling on LLM Reasoning*. Lin et al., 2025, ArXiv, 32 trích dẫn. [Link](https://consensus.app/papers/details/dd766416149456d98bd12b3bb22e283f/)                                             | Chỉ ra ngân sách cố định gây *underthinking* ở câu khó; chia nhỏ câu hỏi và phân bổ ngân sách theo độ phức tạp                                                                                                                                                          |      2, 8      |

### 6.2. Bài huấn luyện kiểm soát độ dài trên R1-Distill-1.5B (dùng cho Nhánh B và phần so sánh)

| Bài báo                                                                                                                                                                                                   | Ghi chú                                                                                                                                                                                                                     |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *ShorterBetter*. Yi et al., 2025, 89 trích dẫn. [Link](https://consensus.app/papers/details/1ff25c12faf95bb3bb1e41d523cd82f0/)                                                                           | RL với phần thưởng "độ dài của lời giải đúng ngắn nhất"; R1-Distill-1.5B/7B giảm 50–80% độ dài                                                                                                            |
| *LASER / LASER-D: Learn to Reason Efficiently with Adaptive Length-based Reward Shaping*. Liu et al., 2025, 89 trích dẫn. [Link](https://consensus.app/papers/details/13e67ac8ac205581b159c282f6d043d2/) | Thưởng theo độ dài, có xét độ khó; R1-Distill 1.5B/7B/32B: +6.1 AIME24 với −63% token                                                                                                                            |
| *Leash: Adaptive Length Penalty and Reward Shaping*. Li et al., 2025, 16 trích dẫn. [Link](https://consensus.app/papers/details/d17a6142aef8592bae1f274731ea91b1/)                                       | RL với hệ số phạt độ dài điều chỉnh theo Lagrange; R1-Distill-1.5B giảm 60% độ dài. ⚠️ **Trùng tên với LEASH (#33)** nhưng là phương pháp khác (cần huấn luyện), dễ nhầm khi trích dẫn |
| *DAST: Difficulty-Adaptive Slow-Thinking*. Shen et al., 2025, 144 trích dẫn. [Link](https://consensus.app/papers/details/8bbc5dd8f4355607818d7822d6fa229e/)                                              | Phạt câu trả lời dài ở câu dễ, khuyến khích suy luận đủ ở câu khó; −30% token                                                                                                                               |

### 6.3. Số trích dẫn của các bài chủ chốt (Consensus)

| Bài                                    | Trích dẫn |
| :-------------------------------------- | ----------: |
| DEER (Yang et al., 2025)                |         253 |
| SEAL (Chen et al., 2025)                |          83 |
| Overclocking (Eisenstadt et al., 2025)  |          27 |
| EAT (Wang et al., 2025)                 |          26 |
| ThinkLess (Li et al., 2025)             |          21 |
| **RPDI gốc (Guan et al., 2026)** |           5 |

Consensus cũng ghi tác giả đầu bài RPDI là **Wei-Xin Guan**, khớp với đính chính ở đầu file. Bài gốc mới có 5 trích dẫn, tức là hướng còn mới và chưa có nhiều công trình kiểm chứng độc lập. Có thể dùng điều này làm luận cứ cho giá trị của việc tái hiện ở chế độ ngân sách khác.

Tài liệu nền về SLM có bình duyệt, dùng cho Chương 1: Wang et al., *A Comprehensive Survey of Small Language Models in the Era of LLMs* (ACM TIST, 288 trích dẫn) và Zheng et al., *A Review on Edge LLMs* (ACM Computing Surveys, 312 trích dẫn). Cả hai giúp củng cố lập luận "ngân sách 768 token mô phỏng thiết bị biên".

### 6.4. Các gap thay đổi thế nào sau khi bổ sung

- **Gap 1 thu hẹp lại.** EntroCut (#31) đã nhắm vào mô hình suy luận cỡ nhỏ với cắt dựa trên entropy. Phải đọc toàn văn để biết bài này dùng mô hình nào và ngân sách bao nhiêu. Nếu EntroCut cũng chạy ở ngân sách dài, gap "ngân sách cực hẹp" vẫn còn nguyên; nếu không, phải định vị lại và đưa EntroCut thành baseline.
- **Gap 2 có thêm baseline mạnh.** CUSUM (#32) thắng DEER và Dynasor, cùng dùng tín hiệu entropy, cài đặt rẻ. Đây là đối thủ tự nhiên cho RPDI-EE.
- **Gap 3 có thêm cách giải quyết.** Conformal Thinking (#37) cho phép chọn λ theo mức rủi ro mong muốn thay vì quét lưới. Có thể áp dụng trên `gsm8k_val_200.json`.
- **Gap 4 có thêm bằng chứng phản biện.** LEASH (#33) mất 10 điểm khi dừng theo entropy. Cần giải thích vì sao RPDI-EE ở 768 token lại tăng: rất có thể vì baseline bị cắt quá nhiều, không phải vì tín hiệu entropy tốt hơn.
- **Gap 10 được củng cố mạnh.** Pipis et al. (#35) cho thấy mô hình distill lặp nhiều khi greedy. Có khả năng một phần đáng kể trong 47% câu cạn ngân sách là do vòng lặp. Nên **đếm tỷ lệ câu có lặp n-gram** trong trace baseline (không cần GPU) để định lượng hiệu ứng này.

---

## 7. Ưu tiên thực hiện

| Ưu tiên | Việc                                                                                                                     | Gap | Chi phí GPU ước lượng                                                                                             |
| :-------: | :------------------------------------------------------------------------------------------------------------------------ | :--: | :--------------------------------------------------------------------------------------------------------------------- |
|     0     | Đọc toàn văn EntroCut (#31) và CUSUM (#32) để kiểm tra định vị                                                 | 1, 2 | Không                                                                                                                 |
|     0     | Đếm tỷ lệ vòng lặp n-gram trong trace baseline 705 câu                                                             |  10  | Không (offline)                                                                                                       |
|     1     | Baseline cắt cứng, phanh ngẫu nhiên và chia ngân sách kiểu Elastic trên 605 câu + McNemar                       |  2  | Thấp nếu mô phỏng offline từ trace baseline; cần chạy online để xác nhận (lệch mô phỏng/online 78 vs 73) |
|     2     | Phân tích định tính 24 câu phanh nhầm và 21.56% câu vẫn cạn ngân sách                                        | 4, 8 | Không cần GPU                                                                                                        |
|     3     | Quét ngân sách {512, 1024, 2048} → W*(B), λ*(B)                                                                      | 1, 3 | Trung bình                                                                                                            |
|     4     | Sửa trích dẫn "Li et al." → "Guan et al."; bổ sung SEAL, Overclocking, NoThinking vào phần công trình liên quan |  —  | Không                                                                                                                 |
|     5     | Kiểm tra với T=0.6                                                                                                      |  10  | Trung bình                                                                                                            |
|     6     | Đo overhead so với DEER                                                                                                 |  7  | Trung bình                                                                                                            |
