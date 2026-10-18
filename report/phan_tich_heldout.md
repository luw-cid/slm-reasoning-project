# Phân tích thống kê RPDI-EE trên tập held-out

- Tập đánh giá: 605 câu GSM8K test (id 100–704), loại các câu id < 100 đã dùng để chọn siêu tham số.
- Ngân sách: `max_new_tokens = 768`, giải mã Greedy.
- Tỷ lệ phanh chỉ tính các lần phanh trong pha suy nghĩ (xem ghi chú cuối).

## 1. Kết quả từng phương pháp

| Phương pháp | Đúng | Accuracy | KTC 95% | Token TB | Think TB | Cạn ngân sách | Tỷ lệ phanh |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| `rpdi_ee` | 433/605 | 71.57% | [67.8, 75.0] | 547.9 | 133.6 | 22.64% | 70.91% |
| `standard` | 316/605 | 52.23% | [48.2, 56.2] | 611.7 | 465.1 | 47.60% | 0.00% |

## 2. So sánh cặp với `standard`

| Phương pháp | Chỉ phương pháp đúng | Chỉ tham chiếu đúng | Δ Accuracy | KTC 95% của Δ | McNemar p |
| :--- | ---: | ---: | ---: | :---: | ---: |
| `rpdi_ee` | 141 | 24 | +19.34 | [+15.5, +23.2] | 2.37e-21 |

## 3. Phân tách theo trạng thái cạn ngân sách của baseline

| Nhóm | Số câu | `rpdi_ee` | `standard` |
| :--- | ---: | ---: | ---: |
| Baseline tự đóng `</think>` | 317 | 79.18% | 80.44% |
| Baseline cạn 768 token | 288 | 63.19% | 21.18% |

## Ghi chú

- Kết quả `rpdi_ee` lấy từ lần chạy 705 câu, trước khi sửa lỗi `RPDILogitsProcessor` tiếp tục giám sát sau khi mô hình tự đóng `</think>`. Cờ `early_exited` gốc báo 79.83%, nhưng chỉ 70.91% là phanh thật trong pha suy nghĩ; phần còn lại là chèn thêm `</think>` vào pha trả lời.
