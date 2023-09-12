# -*- encoding: utf-8 -*-
"""
@File       : torch_utils.py
@Time       : 2023/8/31 16:17
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import os
import platform
import torch
from utils.general import LOGGER, git_describe, file_date
import torch.backends.mps
from contextlib import contextmanager
import torch.distributed as dist


def select_device(device="", batch_size=0, newline=True):
    """device=None or 'cpu' or 0 or '0' or '0,1,2,3' """
    s = f"model 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__}"
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # 转变为string:"cuda:0" -> "0"
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal 性能着色器(MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = -1  # 强制 torch.cuda.is_available() = False
    elif device:  # 非cpu device 请求
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # 设置cuda环境变量
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(",", "")), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # 如果可用，首选GPU
        devices = device.strip(",") if device else "0"  # range(torch.cuda.device_count())  i.e. 0,1,5,8
        n = len(device)  # device统计
        if n > 1 and batch_size > 0:  # 检查batch_size 能否被device_count整除
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB\n)"  # bytes 转为MB
        arg = "cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # 如果可用，首选MPS
        s += "MPS\n"
        arg = "mps"
    else:
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """

    :param local_rank:
    :return:
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])