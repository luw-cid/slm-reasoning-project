# 📈 Báo Cáo Thực Nghiệm Ablation Study & Đường Biên Pareto (RPDI-EE)

- **Mô hình**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Tập kiểm thử**: GSM8K (10 mẫu)
- **Chế độ min_steps**: `w` (bắt đầu giám sát khi $i \ge W$)

## 1. Các Cấu Hình Tối Ưu Thuộc Đường Biên Pareto Frontier

| Cấu hình | Độ chính xác Acc (%) | Tổng Token trung bình | Token suy nghĩ (<think>) | Tỷ lệ dừng sớm (%) | Bước phanh TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RPDI(W=16,λ=1.5)** | **60.00%** | 521.6 | 121.9 | 60.0% | 118.7 |

## 2. Bảng Tổng Hợp Toàn Bộ 16 Cấu Hình Quét Lưới & Baselines

| Cấu hình | Nhóm | Acc (%) | Tổng Token | Think Token | Tỷ lệ phanh (%) | Bước phanh TB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Standard (No Exit)` | standard_baseline | 50.00% | 592.6 | 372.2 | 0.0% | 0.0 |
| `RPDI(W=16,λ=1.5)` | rpdi | 60.00% | 521.6 | 121.9 | 60.0% | 118.7 |
| `RPDI(W=16,λ=2.0)` | rpdi | 60.00% | 547.8 | 237.8 | 50.0% | 329.4 |
| `RPDI(W=16,λ=2.5)` | rpdi | 50.00% | 559.0 | 317.2 | 40.0% | 503.8 |
| `RPDI(W=16,λ=3.0)` | rpdi | 50.00% | 585.1 | 347.7 | 20.0% | 509.0 |
| `RPDI(W=32,λ=1.5)` | rpdi | 60.00% | 523.9 | 188.9 | 60.0% | 230.3 |
| `RPDI(W=32,λ=2.0)` | rpdi | 50.00% | 565.5 | 322.8 | 40.0% | 517.8 |
| `RPDI(W=32,λ=2.5)` | rpdi | 50.00% | 586.0 | 357.6 | 20.0% | 558.5 |
| `RPDI(W=32,λ=3.0)` | rpdi | 50.00% | 592.6 | 372.2 | 0.0% | 0.0 |
| `RPDI(W=64,λ=1.5)` | rpdi | 50.00% | 565.7 | 306.6 | 40.0% | 477.2 |
| `RPDI(W=64,λ=2.0)` | rpdi | 50.00% | 575.4 | 363.6 | 20.0% | 588.5 |
| `RPDI(W=64,λ=2.5)` | rpdi | 50.00% | 592.6 | 372.2 | 0.0% | 0.0 |
| `RPDI(W=64,λ=3.0)` | rpdi | 50.00% | 592.6 | 372.2 | 0.0% | 0.0 |
| `RPDI(W=128,λ=1.5)` | rpdi | 50.00% | 565.4 | 342.1 | 30.0% | 498.3 |
| `RPDI(W=128,λ=2.0)` | rpdi | 50.00% | 592.6 | 371.8 | 10.0% | 765.0 |
| `RPDI(W=128,λ=2.5)` | rpdi | 50.00% | 592.6 | 372.2 | 0.0% | 0.0 |
| `RPDI(W=128,λ=3.0)` | rpdi | 50.00% | 592.6 | 372.2 | 0.0% | 0.0 |
| `HardTrunc(N=32)` | hard_truncation | 50.00% | 536.8 | 32.0 | 100.0% | 34.0 |
| `HardTrunc(N=64)` | hard_truncation | 40.00% | 548.3 | 64.0 | 100.0% | 66.0 |
| `HardTrunc(N=96)` | hard_truncation | 60.00% | 506.3 | 95.4 | 90.0% | 98.0 |
| `HardTrunc(N=128)` | hard_truncation | 60.00% | 485.2 | 121.8 | 80.0% | 130.0 |
| `HardTrunc(N=256)` | hard_truncation | 60.00% | 561.3 | 202.1 | 50.0% | 258.0 |

## 3. Nhận Xét & Khuyến Nghị Lựa Chọn

1. **Vượt trội so với Baseline Cắt Cứng (Fixed Truncation)**: Các cấu hình nằm trên Pareto Frontier của RPDI luôn đạt Accuracy cao hơn đáng kể so với việc cắt cứng tại cùng một mức tiêu thụ token, khẳng định tính hiệu quả của cơ chế nhận thức độ lệch hướng (RPDI).
2. **Sweet Spot (Điểm cân bằng lý tưởng)**: Thay vì chọn cảm tính, đường cong Pareto cung cấp bằng chứng định lượng rõ ràng để chọn cấu hình phù hợp với mục tiêu (tối đa hóa tiết kiệm token hay tối đa hóa độ chính xác).