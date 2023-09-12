# -*- encoding: utf-8 -*-
"""
@File       : wandb_utils.py
@Time       : 2023/9/11 17:14
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # 模型的根路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
RANK = int(os.getenv("RANK", -1))
from utils.general import LOGGER, colorstr

# wandb
DEPRECATION_WARNING = f"{colorstr('wandb')}: WARNING ⚠️ wandb is deprecated and will be removed in a future release." \
                      f"See supported integrations at https://github.com/ultralytics/yolov5#integrations."
try:
    import wandb
    assert hasattr(wandb, "__version__")  # 验证是导入的包，不是本地的
    LOGGER.warning(DEPRECATION_WARNING)
except (ImportError, AssertionError):
    wandb = None


class WandbLogger():
    """
    将训练运行、数据集、模型和预测记录到权重和偏差中。
    此记录器将信息发送到wandb ai的W&B。默认情况下，此信息包括超参数、系统配置和度量、模型度量，以及基本数据度量和分析。
    通过向train.py提供额外的命令参数，还可以记录模型和预测。
    Weights&Biases文档: https://docs.wandb.com/guides/integrations/yolov5
    """
    def __init__(self, opt, run_id=None, job_type="Training"):
        """
        初始化WandbLogger实例
        如果opt.upload_dataset为True，则upload数据集
        如果job_type为"Training"，则设置训练流程
        :param opt: 此运行的命令行参数
        :param run_id: 要恢复的 W&B 运行的id
        :param job_type: 运行的job_type, 用于存储作业类型的属性。
        """
        # 训练前常规
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, wandb.run if wandb else None  # 分别用于存储W&B的主要对象和运行对象。wandb是W&B库的主要接口，wandb_run是W&B运行对象，用于记录和跟踪实验结果。
        self.val_artifact, self.train_artifact = None, None  # 用于存储验证数据和训练数据的Artifact对象。Artifact是W&B中的一种数据对象，用于存储和版本控制数据
        self.train_artifact_path, self.val_artifact_path = None, None  # 存储训练数据和验证数据Artifact的路径
        self.result_artifact = None  # 用于存储实验结果的Artifact对象
        self.val_table, self.result_table = None, None  # 存储验证数据和实验结果的表格
        self.max_imgs_to_log = 16  # 用于限制最大可记录的图像数量。
        self.data_dict = None  # 存储数据的字典
        if self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",  # 是否允许恢复（resume）先前的 W&B 运行。"allow" 表示允许恢复
                                        project="YOLOv5" if opt.project == "runs/train" else Path(opt.project).stem,  # 用于设置实验的项目名称（Project）
                                        entity=opt.entity,  # 用于设置实验的实体（Entity），通常是一个组织或用户的名称
                                        name=opt.name if opt.name != "exp" else None,  # 用于设置实验的名称（Name）。如果 opt.name 不等于 "exp"，则实验名称将设置为 opt.name，否则将使用默认的名称。
                                        job_type=job_type,  # 用于指定作业类型，通常与实验相关的作业类型。
                                        id=run_id,  # 用于指定运行的唯一标识符（ID），可以用于区分不同的实验运行。
                                        allow_val_change=True) if not wandb.run else wandb.run  # 是否允许在运行过程中更改验证集的配置
        if self.wandb_run:
            if self.job_type == "Training":
                if isinstance(opt.data, dict):
                    # 意味着另一个数据集管理器已经处理了数据信息(eg: ClearML)，并且将已经处理的dict存储在opt.data中
                    self.data_dict = opt.data
                self.setup_training(opt)

    def setup_training(self, opt):
        """
        设置训练模型的必要流程：
            - 如果使用wandb_artifact_prefix进行opt.resume统计，则尝试下载模型检查点和数据集工作
            - 更新data_dict, 如果恢复则包含上次运行的信息，如果下载则包含数据集工作的路径
            - 设置日志字典，初始化bbox_interval
        :param opt:
        :return:
        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval
        if isinstance(opt.resume, str):
            model_dir, _ = self.download_model_artifact(opt)
            if model_dir:
                self.weights = Path(model_dir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp, opt.imgsz = str(
                    self.weights), config.save_period, config.batch_size, config.bbox_interval, config.epochs,\
                    config.hyp, config.imgsz

        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = opt.epochs + 1  # 禁用bbox_interval