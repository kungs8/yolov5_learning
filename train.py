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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project 根路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT加入到环境变量中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from utils.general import (check_git_info, print_args)

LOCAL_RANK = int(os.getenv("LOCAL_TANK", -1))
print(LOCAL_RANK)
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()  # 检查git信息


def parse_opt(known=False):
    """配置参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=ROOT / "yolov5s.pt", help="初始化的权重路径")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml 路径")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml 路径")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyps.scratch-low.yaml", help="超参数路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练的总epochs")
    parser.add_argument("--batchz-size", type=int, default=16, help="total batch size for all GPUS, -1 for autobatch")
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


def main(opt):
    # 检查
    if RANK in {-1, 0}:
        print_args(vars(opt))


if __name__ == '__main__':
    # 获取参数
    opt = parse_opt()
    # 主函数
    main(opt)