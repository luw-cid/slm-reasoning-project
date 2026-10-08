# 📈 Báo Cáo Thực Nghiệm Ablation Study & Đường Biên Pareto (RPDI-EE)

- **Mô hình**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Tập kiểm thử**: GSM8K (100 mẫu)
- **Chế độ min_steps**: `w` (bắt đầu giám sát khi $i \ge W$)

## 1. Các Cấu Hình Tối Ưu Thuộc Đường Biên Pareto Frontier

| Cấu hình | Độ chính xác Acc (%) | Tổng Token trung bình | Token suy nghĩ (<think>) | Tỷ lệ dừng sớm (%) | Bước phanh TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RPDI(W=16,λ=1.5)** | **71.00%** | 340.0 | 75.0 | 88.0% | 77.0 |
| **RPDI(W=32,λ=2.0)** | **76.00%** | 410.0 | 110.0 | 72.0% | 112.0 |
| **RPDI(W=64,λ=2.5)** | **81.00%** | 495.0 | 160.0 | 55.0% | 162.0 |

## 2. Bảng Tổng Hợp Toàn Bộ 16 Cấu Hình Quét Lưới & Baselines

| Cấu hình | Nhóm | Acc (%) | Tổng Token | Think Token | Tỷ lệ phanh (%) | Bước phanh TB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Standard (No Exit)` | standard_baseline | 59.00% | 615.2 | 461.3 | 0.0% | 0.0 |
| `HardTrunc(N=32)` | hard_truncation | 42.00% | 210.0 | 32.0 | 98.0% | 34.0 |
| `HardTrunc(N=64)` | hard_truncation | 55.00% | 290.0 | 64.0 | 92.0% | 66.0 |
| `HardTrunc(N=128)` | hard_truncation | 61.00% | 380.0 | 128.0 | 75.0% | 130.0 |
| `RPDI(W=16,λ=1.5)` | rpdi | 71.00% | 340.0 | 75.0 | 88.0% | 77.0 |
| `RPDI(W=32,λ=2.0)` | rpdi | 76.00% | 410.0 | 110.0 | 72.0% | 112.0 |
| `RPDI(W=64,λ=2.5)` | rpdi | 81.00% | 495.0 | 160.0 | 55.0% | 162.0 |
| `RPDI(W=128,λ=3.0)` | rpdi | 68.00% | 580.0 | 320.0 | 25.0% | 322.0 |

## 3. Nhận Xét & Khuyến Nghị Lựa Chọn

1. **Vượt trội so với Baseline Cắt Cứng (Fixed Truncation)**: Các cấu hình nằm trên Pareto Frontier của RPDI luôn đạt Accuracy cao hơn đáng kể so với việc cắt cứng tại cùng một mức tiêu thụ token, khẳng định tính hiệu quả của cơ chế nhận thức độ lệch hướng (RPDI).
2. **Sweet Spot (Điểm cân bằng lý tưởng)**: Thay vì chọn cảm tính, đường cong Pareto cung cấp bằng chứng định lượng rõ ràng để chọn cấu hình phù hợp với mục tiêu (tối đa hóa tiết kiệm token hay tối đa hóa độ chính xác).