# -*- encoding: utf-8 -*-
"""
@File       : __init__.py
@Time       : 2023/9/1 15:04
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import os
from utils.general import LOGGER, colorstr
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.loggers.clearml.clearml_utils import ClearmlLogger

# 常参数
LOGGERS = ("csv", "tb", "wandb", "clearml", "comet")  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv("RANK", -1))

# comet_ml
try:
    if RANK in {-1, 0}:
        import comet_ml
        assert hasattr(comet_ml, "__version__")  # 验证包不是本地的文件夹
        from utils.loggers.comet import CometLogger
    else:
        comet_ml = None
except (ImportError, AssertionError):
    comet_ml = None

# SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = lambda *args: None  # None = SummaryWriter(str)

# wandb (W&B帐户中，你可以查看和比较不同实验的结果、可视化模型性能、保存超参数配置等等。这有助于更好地理解和管理你的机器学习项目。)
try:
    import wandb
    assert hasattr(wandb, "__version__")  # 验证是导入的包，不是本地的
    import importlib.metadata as pkg_metadata
    if pkg_metadata.version("wandb") >= "0.12.2":
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # 已知的 no-TTY 终端问题
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

# ClearNL
try:
    import clearml
    assert hasattr(clearml, "__version__")  # 验证是导入的包，不是本地的
except (ImportError, AssertionError):
    clearml = None


class Loggers():
    """模型的log类别"""
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # 绘制结果
        self.logger = logger  # 打印结果到Console
        self.include = include
        self.keys = [
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # 训练损失
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",  # 矩阵
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # 验证损失
            "x/lr0",
            "x/lr1",
            "x/lr2"
        ]  # 参数
        self.best_keys = ["best/epoch", "best/precision", "best/recall", "best/mAP_0.5", "bets/mAP_0.5:0.95"]
        for k in LOGGERS:
            setattr(self, k, None)  # 初始化空的logger字典
        self.csv = True  # 总是将log保存到csv

        # 信息
        if not comet_ml:
            prefix = colorstr("Comet: ")
            s = f"{prefix}run 'pip install comet_ml' to automatically track and visualize model 🚀runs in Comet"
            self.logger.info(s)

        # TensorBoard
        s = self.save_dir
        if "tb" in self.include and not self.opt.evolve:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B帐户中，可以查看和比较不同实验的结果、可视化模型性能、保存超参数配置等等。这有助于更好地理解和管理你的机器学习项目。
        if wandb and "wandb" in self.include:
            self.opt.hyp = self.hyp  # 添加超参数
            self.wandb = WandbLogger(self.opt)
        else:
            self.wandb = None

        # ClearML
        if clearml and ("clearml" in self.include):
            try:
                self.clearml = ClearmlLogger(self.opt, self.hyp)
            except Exception:
                self.clearml = None
                prefix = colorstr("ClearML: ")
                LOGGER.warning(f"{prefix}WARNING ⚠️ ClearML is installed but not cnfigured, skipping ClearML logging.")
        else:
            self.clearml = None

        # Comet
        if clearml and ("comet" in self.include):
            if isinstance(self.opt.resume, str) and self.opt.resume.startswith("comet://"):
                run_id = self.opt.resume.split("/")[-1]
                self.comet_logger = CometLogger(self.opt, self.hyp, run_id=run_id)
            else:
                self.comet_logger = CometLogger(self.opt, self.hyp)
        else:
            self.comet_logger = None

    @property
    def remote_dataset(self):
        """如果提供类自定义数据集链接，则获取data_dict"""
        data_dict = None
        if self.clearml:
            data_dict = self.clearml.data_dict
        if self.wandb:
            data_dict = self.wandb.data_dict
        if self.comet_logger:
            data_dict = self.comet_logger.data_dict
        return data_dict