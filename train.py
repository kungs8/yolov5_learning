# -*- encoding: utf-8 -*-
"""
@File       : train.py
@Time       : 2023/8/30 09:41
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import argparse
import sys
import os
from pathlib import Path
import torch
import yaml
from torch import distributed

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project 根路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT加入到环境变量中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from utils.general import (LOGGER, colorstr, check_git_info, print_args, check_git_status, check_requirements,
                           check_file, get_latest_run, check_yaml, increment_path, yaml_save, methods, init_seeds,
                           check_dataset, check_suffix)
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loggers import Loggers
from utils.downloads import is_url
from utils.torch_utils import select_device, torch_distributed_zero_first
from utils.callbacks import Callbacks

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  #
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()  # 检查git信息


def parse_opt(known=False):
    """配置参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="初始化的权重路径")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml 路径")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml 路径")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="超参数路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练的总epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUS, -1 for autobatch")
    parser.add_argument("--imgsz", "-img", "--img-size", type=int, default=640, help="训练、验证的图像大小(像素)")
    parser.add_argument("--rect", action="store_true", help="矩形训练")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="恢复最近的训练")
    parser.add_argument("--nosave", action="store_true", help="仅保留最终的检查点")
    parser.add_argument("--noval", action="store_true", help="仅验证最终的epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="禁用自动锚点")
    parser.add_argument("--noplots", action="store_true", help="报错没有绘图的文件")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="为 x 代演化超参数")
    parser.add_argument("--bucket", type=str, default="", help="gsutil 存储桶")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="选择使用加权图像进行训练")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="改变图像大小 +/- 50%")
    parser.add_argument("--single-cls", action="store_true", help="将多类数据训练为单类数据")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="优化器")
    parser.add_argument("--sync-bn", action="store_true", help="使用SyncBatchNorm，仅在DDP模式下可用")
    parser.add_argument("--workers", type=int, default=8, help="数据加载器最大工作数(DDP 模式下每个 RANK)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="保存到 project/name")
    parser.add_argument("--name", default="exp", help="保存到 project/name")
    parser.add_argument("--exist-ok", action="store_true", help="现有 project/name，不增加")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="余弦LR调度器")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="标签平滑 epsilon")
    parser.add_argument("--patience", type=int, default=100, help="早期停止 patient（没有改善的时期）")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="冻结层：backbone=10，first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="每 x epoch 保存检查点（如果 <1 则禁用）")
    parser.add_argument("--seed", type=int, default=0, help="全局训练固定种子")
    parser.add_argument("--local_rank", type=int, default=-1, help="自动ddp多GPU参数，请勿修改")

    # logger 参数
    parser.add_argument("--entity", default=None, help="实体")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help="上传数据，'val'可选")
    parser.add_argument("--bbox_interval", type=int, default=-1, help="设置边界框图像记录间隔")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="要使用的数据集工件的版本")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def train(hyp, opt, device, callbacks):  # hyp是path/to/hyp.yaml or hyp字典
    # 获取参数
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run("on_pretrain_routine_start")

    # 目录
    w = save_dir / "weights"  # 权重文件夹
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # 创建文件夹
    last, best = w / "last.pt", w / "best.pt"

    # 超参数
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # 加载hyps字典
    LOGGER.info(colorstr("Hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

    # 保存run设置
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))  # vars:一个命名空间（namespace）或对象, 返回一个对象的属性和属性值的字典


    # 日志
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # 日志实例

        # 注册活动
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # 处理自定义数据集工件链接
        data_dict = loggers.remote_dataset
        if resume:
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # 配置
    plots = (not evolve) and (not opt.noplots)
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministric=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # 检查是否为None
    train_path, val_path = data_dict["train"], data_dict["val"]
    print("=====single_cls:", single_cls)
    print("=====nc:", data_dict["nc"])
    nc = 1 if single_cls else int(data_dict["nc"])  # 类别数
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # 类别名称
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO数据集

    # 模型
    check_suffix(weights, ".pt")  # 检查权重

    print(f"====data_dict:{data_dict}")




def main(opt, callbacks=Callbacks()):
    # 检查
    if RANK in {-1, 0}:
        # 打印函数参数（可选参数字典）
        print_args(vars(opt))
        # 检查git状态
        check_git_status()
        # 检查requirements
        check_requirements(ROOT / "requirements.txt")

    # 恢复(从特定的 or 最近的 last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # 训练的选项 yaml
        opt_data = opt.data  # 原始数据
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # 替换
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # 恢复
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # 避免HUB恢复超时
    else:
        # 检查
        opt.data = check_file(opt.data)
        opt.cfg = check_yaml(opt.cfg)
        opt.hyp = check_yaml(opt.hyp)
        opt.weights = str(opt.weights)
        opt.project = str(opt.project)
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # 如果是默认的项目名称，则重命名为 `runs/evolve`
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # 将resume传递给 exist_ok, 并禁止resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # 使用model.yaml 作为名字
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 保存文件的路径，存在则新建

    # DDP(Distributed Data Parallel) mode :(多GPU训练的模式，用于在分布式环境中有效地并行训练深度学习模型)
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with model MUlti-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)  # 创建一个表示当前进程所在 GPU 设备的 PyTorch 设备对象
        distributed.init_process_group(backend="ncc1" if distributed.is_nccl_available() else "gloo")  # 初始化分布式进程组。根据可用的通信后端选择 "ncc1"（NCCL）或 "gloo"（Gloo）。


    # 训练
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
    else:
        pass





if __name__ == '__main__':
    # 获取参数
    opt = parse_opt()
    # 主函数
    main(opt)