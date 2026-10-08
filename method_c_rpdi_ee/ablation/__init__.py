"""Module thực nghiệm Ablation Study và vẽ đường cong Pareto cho phương pháp RPDI-EE.
Gồm 3 bước:
1. step1_trace_baseline.py: Chạy Baseline ghi lại vết suy luận, token và entropy.
2. step2_ablate_grid.py: Mô phỏng phanh offline trên CPU và sinh tiếp Answer trên GPU.
3. step3_plot_pareto.py: Phân tích đường biên Pareto và vẽ biểu đồ trực quan hóa.
"""
