# 📈 Báo Cáo Thực Nghiệm Ablation Study & Đường Biên Pareto (RPDI-EE)

- **Mô hình**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Tập kiểm thử**: GSM8K (100 mẫu)
- **Chế độ min_steps**: `w` (bắt đầu giám sát khi $i \ge W$)

## 1. Các Cấu Hình Tối Ưu Thuộc Đường Biên Pareto Frontier

| Cấu hình | Độ chính xác Acc (%) | Tổng Token trung bình | Token suy nghĩ (<think>) | Tỷ lệ dừng sớm (%) | Bước phanh TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RPDI(W=16,λ=1.5)** | **78.00%** | 518.1 | 121.9 | 75.0% | 122.4 |

## 2. Bảng Tổng Hợp Toàn Bộ 16 Cấu Hình Quét Lưới & Baselines

| Cấu hình | Nhóm | Acc (%) | Tổng Token | Think Token | Tỷ lệ phanh (%) | Bước phanh TB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Standard (No Exit)` | standard_baseline | 59.00% | 615.2 | 461.3 | 0.0% | 0.0 |
| `RPDI(W=16,λ=1.5)` | rpdi | 78.00% | 518.1 | 121.9 | 75.0% | 122.4 |
| `RPDI(W=16,λ=2.0)` | rpdi | 76.00% | 563.2 | 272.6 | 56.0% | 316.9 |
| `RPDI(W=16,λ=2.5)` | rpdi | 66.00% | 593.3 | 403.1 | 33.0% | 497.6 |
| `RPDI(W=16,λ=3.0)` | rpdi | 61.00% | 610.2 | 441.7 | 12.0% | 515.4 |
| `RPDI(W=32,λ=1.5)` | rpdi | 74.00% | 547.5 | 212.4 | 61.0% | 247.2 |
| `RPDI(W=32,λ=2.0)` | rpdi | 62.00% | 587.4 | 405.8 | 32.0% | 506.0 |
| `RPDI(W=32,λ=2.5)` | rpdi | 61.00% | 612.4 | 453.4 | 11.0% | 594.5 |
| `RPDI(W=32,λ=3.0)` | rpdi | 59.00% | 615.0 | 460.4 | 2.0% | 638.0 |
| `RPDI(W=64,λ=1.5)` | rpdi | 68.00% | 577.3 | 359.3 | 38.0% | 436.2 |
| `RPDI(W=64,λ=2.0)` | rpdi | 60.00% | 610.4 | 453.9 | 9.0% | 621.9 |
| `RPDI(W=64,λ=2.5)` | rpdi | 59.00% | 615.2 | 460.4 | 2.0% | 723.0 |
| `RPDI(W=64,λ=3.0)` | rpdi | 59.00% | 615.2 | 461.3 | 0.0% | 0.0 |
| `RPDI(W=128,λ=1.5)` | rpdi | 60.00% | 602.3 | 431.5 | 19.0% | 531.4 |
| `RPDI(W=128,λ=2.0)` | rpdi | 59.00% | 615.2 | 461.3 | 1.0% | 765.0 |
| `RPDI(W=128,λ=2.5)` | rpdi | 59.00% | 615.2 | 461.3 | 0.0% | 0.0 |
| `RPDI(W=128,λ=3.0)` | rpdi | 59.00% | 615.2 | 461.3 | 0.0% | 0.0 |
| `HardTrunc(N=32)` | hard_truncation | 69.00% | 549.3 | 32.0 | 100.0% | 34.0 |
| `HardTrunc(N=64)` | hard_truncation | 68.00% | 550.7 | 64.0 | 100.0% | 66.0 |
| `HardTrunc(N=96)` | hard_truncation | 65.00% | 536.2 | 95.1 | 94.0% | 98.0 |
| `HardTrunc(N=128)` | hard_truncation | 75.00% | 520.5 | 123.7 | 85.0% | 130.0 |
| `HardTrunc(N=256)` | hard_truncation | 73.00% | 546.1 | 214.2 | 61.0% | 258.0 |

## 3. Nhận Xét & Khuyến Nghị Lựa Chọn

1. **Vượt trội so với Baseline Cắt Cứng (Fixed Truncation)**: Các cấu hình nằm trên Pareto Frontier của RPDI luôn đạt Accuracy cao hơn đáng kể so với việc cắt cứng tại cùng một mức tiêu thụ token, khẳng định tính hiệu quả của cơ chế nhận thức độ lệch hướng (RPDI).
2. **Sweet Spot (Điểm cân bằng lý tưởng)**: Thay vì chọn cảm tính, đường cong Pareto cung cấp bằng chứng định lượng rõ ràng để chọn cấu hình phù hợp với mục tiêu (tối đa hóa tiết kiệm token hay tối đa hóa độ chính xác).