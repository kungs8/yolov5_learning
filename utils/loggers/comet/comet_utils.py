# -*- encoding: utf-8 -*-
"""
@File       : comet_utils.py
@Time       : 2023/8/31 10:09
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import logging
import os
from urllib.parse import urlparse

import yaml

# Comet ML 是一个用于实验跟踪、日志记录和分析的工具，通常在机器学习和数据科学项目中使用
try:
    import comet_ml
except (ModuleNotFoundError, ImportError):
    comet_ml = None

# 添加日志
logger = logging.getLogger(__name__)

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
COMET_DEFAULT_CHECKPOINT_FILENAME = os.getenv("COMET_DEFAULT_CHECKPOINT_FILENAME", "last.pt")


def set_opt_parameters(opt, experiment):
    """
    使用参数更新opt的命名空间，恢复来自Comet的现有实验的运行
    :param opt: 命令行选项的命名空间
    :param experiment: Comet API 实验对象
    :return:
    """
    asset_list = experiment.get_asset_list()  # 获取了 Comet ML 实验的资源列表，这些资源可以是与实验关联的文件或数据
    resume_string = opt.resume

    for asset in asset_list:
        if asset["fileName"] == "opt.yaml":
            asset_id = asset["assetId"]
            asset_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)  # 获取资源的二进制数据
            opt_dict = yaml.safe_load(asset_binary)  # 使用 PyYAML 库将二进制数据解析为字典，这个字典应该包含超参数信息
            for key, value in opt_dict.items():
                setattr(opt, key, value)
            opt.resume = resume_string

    # 保存超参数到YAML文件中
    # 需要通过训练脚本中的检查
    save_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(save_dir, exist_ok=True)

    hyp_yaml_path = f"{save_dir}/hyp.yaml"  # 构建保存超参数的 YAML 文件的路径
    with open(hyp_yaml_path, "w") as f:
        yaml.dump(opt.hyp, f)
    opt.hyp = hyp_yaml_path  # 更新 opt.hyp 为超参数文件的路径


def download_model_checkpoint(opt, experiment):
    """下载模型的检查点"""
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    model_name = COMET_MODEL_NAME
    model_asset_list = experiment.get_model_asset_list(model_name)  # 获取指定模型名称的资源列表

    if len(model_asset_list) == 0:
        logger.error(f"COMET ERROR: No checkpoints found for model name: {model_name}")
        return
    model_asset_list = sorted(model_asset_list, key=lambda x: x["step"], reverse=True)  # 对模型检查点列表进行按`step`排序，以获取最新的检查点
    logged_checkpoint_map = {asset["fileName"]: asset["assetId"] for asset in model_asset_list}  #  创建一个字典，将文件名映射到对应的资源标识符

    resource_url = urlparse(opt.weights)  # 解析 opt.weights 参数（可能是模型检查点的路径）的 URL
    checkpoint_filename = resource_url.query  # 从 URL 中获取查询字符串，即文件名

    if checkpoint_filename:
        asset_id = logged_checkpoint_map.get(checkpoint_filename)
    else:
        asset_id = logged_checkpoint_map.get(COMET_DEFAULT_CHECKPOINT_FILENAME)
        checkpoint_filename = COMET_DEFAULT_CHECKPOINT_FILENAME

    if asset_id is None:
        logger.error(f"COMET ERROR: Checkpoint {checkpoint_filename} not found in the given Experiment")
        return

    # 尝试从 Comet ML 平台下载模型检查点
    try:
        logger.info(f"COMET INFO: Downloading checkpoint {checkpoint_filename}")
        asset_filename = checkpoint_filename

        model_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
        model_doanload_path = f"{model_dir}/{asset_filename}"
        with open(model_doanload_path, "wb") as f:
            f.write(model_binary)
        opt.weights = model_doanload_path
    except Exception as e:
        logger.warning("COMET WARNING: Unable to download checkpoint from Comet")
        logger.exception(e)


def check_comet_resume(opt):
    """
    根据模型检查点将运行参数恢复到原始状态，并记录实验参数
    :param opt(argparse.Namespace): 传递的模型的命令行参数
    :return: 如果运行恢复成功则返回True， 否则返回None
    """
    if comet_ml is None:
        return

    if isinstance(opt.resume, str):
        if opt.resume.startswith(COMET_PREFIX):
            api = comet_ml.API()  # 创建了一个 Comet ML 的 API 实例，可以用于与 Comet ML 平台进行交互
            resource = urlparse(opt.resume)  # 将 resume 分解成 URL 的各个组成部分，比如协议、主机、路径等。
            experiment_path = f"{resource.netloc}{resource.path}"  # 构建实验路径，由 urlparse 分析得到的主机和路径组成
            experiment = api.get(experiment_path)  # 使用构建的实验路径来获取对应的 Comet ML 实验。这可以让你在代码中恢复之前创建的实验并继续记录
            set_opt_parameters(opt, experiment)  # 将一些参数从 Comet ML 的实验对象（experiment）中提取出来，并将其设置到你的代码中的 opt 对象中，以供后续使用
            download_model_checkpoint(opt, experiment)  # 从 Comet ML 平台下载与实验关联的模型检查点。模型检查点可能包含你训练的模型权重以及训练过程中的其他信息
            return True
    return None