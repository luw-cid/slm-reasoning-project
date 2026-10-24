"""Package phương pháp 2: RLVR (Reinforcement Learning with Verifiable Rewards).
Triển khai thuật toán GRPO (Group Relative Policy Optimization) kết hợp LoRA trên mô hình DeepSeek-R1-Distill-Qwen-1.5B.
"""

from .reward_funcs import accuracy_reward_func, format_reward_func, get_reward_functions

__all__ = [
    "accuracy_reward_func",
    "format_reward_func",
    "get_reward_functions",
]
