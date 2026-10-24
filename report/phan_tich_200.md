# Phân tích thống kê RPDI-EE trên tập held-out

- Tập đánh giá: 200 câu GSM8K test (id 100–299), loại các câu id < 100 đã dùng để chọn siêu tham số.
- Ngân sách: `max_new_tokens = 768`, giải mã Greedy.
- Tỷ lệ phanh chỉ tính các lần phanh trong pha suy nghĩ (xem ghi chú cuối).

## 1. Kết quả từng phương pháp

| Phương pháp | Đúng | Accuracy | KTC 95% | Token TB | Think TB | Cạn ngân sách | Tỷ lệ phanh |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N130` | 144/200 | 72.00% | [65.4, 77.8] | 551.0 | 126.2 | 27.50% | 85.00% |
| `random` | 137/200 | 68.50% | [61.8, 74.5] | 552.1 | 126.8 | 26.00% | 85.00% |
| `rpdi_W16_l1.5` | 143/200 | 71.50% | [64.9, 77.3] | 564.8 | 133.1 | 27.00% | 70.50% |
| `standard` | 105/200 | 52.50% | [45.6, 59.3] | 619.3 | 492.2 | 50.50% | 0.00% |

## 2. So sánh cặp với `standard`

| Phương pháp | Chỉ phương pháp đúng | Chỉ tham chiếu đúng | Δ Accuracy | KTC 95% của Δ | McNemar p |
| :--- | ---: | ---: | ---: | :---: | ---: |
| `hardtrunc_N130` | 48 | 9 | +19.50 | [+12.6, +26.4] | 1.52e-07 |
| `random` | 40 | 8 | +16.00 | [+9.6, +22.4] | 3.31e-06 |
| `rpdi_W16_l1.5` | 43 | 5 | +19.00 | [+12.7, +25.3] | 1.37e-08 |

## 3. Phân tách theo trạng thái cạn ngân sách của baseline

| Nhóm | Số câu | `hardtrunc_N130` | `random` | `rpdi_W16_l1.5` | `standard` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline tự đóng `</think>` | 99 | 81.82% | 78.79% | 82.83% | 80.81% |
| Baseline cạn 768 token | 101 | 62.38% | 58.42% | 60.40% | 24.75% |
