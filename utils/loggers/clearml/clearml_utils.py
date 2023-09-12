# -*- encoding: utf-8 -*-
"""
@File       : clearml_utils.py
@Time       : 2023/9/12 10:12
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import glob
from pathlib import Path

import yaml

try:
    import clearml
    from clearml import Dataset, Task
    assert hasattr(clearml, "__version__")  # 验证是导入的包，不是本地的
except (ImportError, AssertionError):
    clearml = None


def construct_dataset(clearml_info_string):
    """加载一个clearml数据集并用其内容填充内部的data_dict"""
    dataset_id = clearml_info_string.replace("clearml://", "")
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_root_path = Path(dataset.get_local_copy())

    # 在数据集中搜索yaml文件定义
    yaml_filenames = list(glob.glob(str(dataset_root_path / "*.yaml")) + glob.glob(str(dataset_root_path / "*.yml")))
    if len(yaml_filenames) > 1:
        raise ValueError("More than one yaml file was found in the dataset root, cannot determine which one contains the dataset definition this way.")
    elif len(yaml_filenames) == 0:
        raise ValueError("No yaml definition in dataset root path, check that there is a cirrect yaml file inside the dataset root path.")
    with open(yaml_filenames[0]) as f:
        data_definition = yaml.safe_load(f)

    assert set(data_definition.keys()).issubset({"train", "test", "val", "nc", "names"}), "The right keys were not found in the yaml file, make sure it as least has the following keys: ('train', 'test', 'val', 'nc', 'names')"

    data_dict = dict()
    data_dict["train"] = str((dataset_root_path / data_definition["train"]).resolve() if data_definition["train"] else None)
    data_dict["test"] = str((dataset_root_path / data_definition["test"]).resolve() if data_definition["test"] else None)
    data_dict["val"] = str((dataset_root_path / data_definition["val"]).resolve() if data_definition["val"] else None)
    data_dict["nc"] = data_definition["nc"]
    data_dict["names"] = data_definition["names"]
    return data_dict


class ClearmlLogger:
    """
    将训练运行、数据集、模型和预测记录到ClearML。
    此记录器将信息发送到app.clear.ml处的ClearML或自己的托管服务器。
    默认情况下，这些信息包括超参数、系统配置和指标、模型指标、代码信息和基本数据指标和分析。
    通过train.py、数据集提供额外的命令行参数。
    模型和预测也可以被记录
    """
    def __init__(self, opt, hyp):
        """
        - 初始化ClearML Task，该对象将捕获实验
        - 如果opt.upload_dataset为True， 则将数据集版本上传到ClearML Data
        :param opt: 命令行参数
        :param hyp: 超参数
        """
        self.current_epoch = 0
        # 跟踪记录的图像数量以实施限制
        self.current_epoch_logged_images = set()
        # 每个时期记录到ClearML的最大图像数量
        self.max_imgs_to_log_per_epoch = 16
        # 获取应记录边界框图像的纪元间隔
        self.bbox_interval = opt.bbox_interval
        self.clearml = clearml
        self.task = None
        self.data_dict = None
        if self.clearml:
            self.task = Task.init(
                project_name=opt.project if opt.project != "runs/train" else "YOLOv5",
                task_name=opt.name if opt.name != "exp" else "Training",
                tags=["YOLOv5"],
                output_uri=True,
                reuse_last_task_id=opt.exist_ok,
                auto_connect_frameworks={"pytorch": False}
                # 这里断开pytorch自检，因为代码中添加了手动模型保存点
            )
            # ClearML的钩子已经抓取了所有通用参数
            # 仅来自yaml配置文件的超参数必须手动添加
            self.task.connect(hyp, name="Hyperparameters")
            self.task.connect(opt, name="Args")

            # 通过设置远程代理使用的docker镜像，确保代码可以远程运行
            self.task.set_base_docker("ultralytics/yolov5:latest",
                                      docker_arguments="--ipc=host -e='CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1'",
                                      docker_setup_bash_script="pip install clearml")
            # 如果需要，获取ClearML数据集
            if opt.data.startswith("clearml://"):
                # data_dict应具有以下keys：names, nc(number of classes), test, train, val(all three relative paths to ../datasets)
                self.data_dict = construct_dataset(opt.data)
                # 将数据设置为data_dict, 因为没有这些信息wandb会崩溃，opt是最好的方法
                self.data = self.data_dict
