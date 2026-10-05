"""Module cấu hình seed ngẫu nhiên nhằm đảm bảo tính tái lập (reproducibility) qua các thực nghiệm.
Thiết lập môi trường xác định (deterministic) cho random, numpy và PyTorch (CPU và CUDA).
"""

import os
import random
from typing import Optional


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> int:
    """Thiết lập seed ngẫu nhiên cho Python random, numpy và torch để đảm bảo tính tái lập.

    Tham số:
        seed: Giá trị seed số nguyên (mặc định: 42).
        deterministic_cudnn: Nếu True, cấu hình cuDNN sử dụng các thuật toán tất định.
            Lưu ý: Có thể ảnh hưởng nhẹ đến hiệu năng tính toán trên một số dòng GPU.

    Trả về:
        Giá trị seed vừa được thiết lập.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """Hàm khởi tạo worker cho DataLoader của PyTorch.

    Đảm bảo mỗi tiến trình worker của DataLoader nhận một seed riêng biệt nhưng hoàn toàn xác định.
    """
    try:
        import torch
        worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    except ImportError:
        worker_seed = 42 + worker_id

    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass
    random.seed(worker_seed)
