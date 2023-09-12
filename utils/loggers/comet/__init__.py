# -*- encoding: utf-8 -*-
"""
@File       : __init__.py
@Time       : 2023/9/4 13:15
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import json
import logging
import os
import sys
from pathlib import Path
import yaml

# 常参数配置
PROJECT_DEFAULT = "yolov5"
COMET_PREFIX = "comet://"
COMMET_MODE = os.getenv("COMMET_MODE", "online")  # 在线
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", PROJECT_DEFAULT)  # 模型保存设置
COMET_UPLOAD_DATASET = os.getenv("COMET_UPLOAD_DATASET", "false").lower() == "true"  # 数据集工件设置

# batch日志设置
COMET_LOG_BATCH_METRICS = (os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true")
COMET_BATCH_LOGGING_INTERVAL = os.getenv("COMET_BATCH_LOGGING_INTERVAL", 1)
COMET_LOG_PER_CLASS_METRICS = (os.getenv("COMET_LOG_PER_CLASS_METRICS", "false").lower() == "true")
COMET_PREDICTION_LOGGING_INTERVAL = os.getenv("COMET_PREDICTION_LOGGING_INTERVAL", 1)

# 评估设置
COMET_MAX_IMAGE_UPLOADS = int(os.getenv("COMET_MAX_IMAGE_UPLOADS", 100))
COMET_LOG_CONFUSION_MATRIX = (os.getenv("COMET_LOG_CONFUSION_MATRIX", "true").lower() == "true")
COMET_LOG_PREDICTIONS = os.getenv("COMET_LOG_PREDICTIONS", "true").lower() == "true"

# 混淆矩阵设置
CONF_THRES = float(os.getenv("CONF_THRES", 0.001))
IOU_THRES = float(os.getenv("IOU_THRES", 0.6))

# 配置日志
logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # 模型的根路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加ROOT到路径中
try:
    import comet_ml

    # 项目配置
    cofig = comet_ml.get_config()
    COMMET_PROJECT_NAME = cofig.get_string(os.getenv("COMMET_PROJECT_NAME"), "comet.project_name", default=PROJECT_DEFAULT)
except ImportError:
    comet_ml = None
    COMMET_PROJECT_NAME = None


from utils.general import check_dataset


class CometLogger():
    """使用Comet记录指标、参数、源代码、模型等"""
    def __init__(self, opt, hyp, run_id=None, jog_type="Training", **experiment_kwargs) -> None:
        self.job_type= jog_type
        self.opt = opt
        self.hyp = hyp

        # Comet 标记
        self.comet_mode = COMMET_MODE

        # Batch 日志设置
        self.log_batch_metrics = COMET_LOG_BATCH_METRICS
        self.comet_log_batch_interval = COMET_BATCH_LOGGING_INTERVAL

        # 数据集工件设置
        self.upload_dataset = self.opt.upload_dataset or COMET_UPLOAD_DATASET
        self.resume = self.opt.resume

        # 传递给实验对象的默认参数
        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
            "project_name": COMMET_PROJECT_NAME
        }
        self.default_experiment_kwargs.update(experiment_kwargs)
        self.experiment = self._get_experiment(self.comet_mode, run_id)
        self.experiment.set_name(self.opt.name)

        self.data_dict = self.check_dataset(self.opt.data)
        self.class_names = self.data_dict["names"]
        self.num_classes = self.data_dict["nc"]

        self.logged_images_count = 0
        self.max_images = COMET_MAX_IMAGE_UPLOADS

        if run_id is None:
            self.experiment.log_other("Createed from ", PROJECT_DEFAULT)
            if not isinstance(self.experiment, comet_ml.OfflineExperiment):
                workspace, project_name, experiment_id = self.experiment.url.split("/")[-3:]
                self.experiment.log_other("Rin Path", f"{workspace}/{project_name}/{experiment_id}")
            self.log_parameters(vars(opt))
            self.log_parameters(self.opt.hyp)
            self.log_asset_data(self.opt.hyp, name="hyperparameters.json", metadata={"type": "hyp-config-file"})
            self.log_asset(f"{self.opt.save_dir}/opt.yaml", metadata={"type": "opt-config-file"})

        self.comet_log_confusion_matrix = COMET_LOG_CONFUSION_MATRIX

        if hasattr(self.opt, "conf_thres"):
            self.conf_thres = self.opt.conf_thres
        else:
            self.conf_thres = CONF_THRES
        if hasattr(self.opt, "iou_thres"):
            self.iou_thres = self.opt.iou_thres
        else:
            self.iou_thres = IOU_THRES
        self.log_parameters({"val_iou_threshold": self.iou_thres, "val_conf_threshold": self.conf_thres})

        self.comet_log_predictions = COMET_LOG_PREDICTIONS
        if self.opt.bbox_interval == -1:
            self.comet_log_prediction_interval = (1 if self.opt.epochs < 10 else self.opt.epochs // 10)
        else:
            self.comet_log_prediction_interval = self.opt.bbox_interval

        if self.comet_log_predictions:
            self.metadata_dict = {}
            self.logged_image_names = []

        self.comet_log_per_class_metrics = COMET_LOG_PER_CLASS_METRICS
        self.experiment.log_others({
            "comet_mode": COMMET_MODE,
            "comet_max_image_uploads": COMET_MAX_IMAGE_UPLOADS,
            "comet_log_per_class_metrics": COMET_LOG_PER_CLASS_METRICS,
            "comet_log_batch_metrics": COMET_LOG_BATCH_METRICS,
            "comet_log_confusion_matrix": COMET_LOG_CONFUSION_MATRIX,
            "comet_model_name": COMET_MODEL_NAME
        })

        # 检查是否使用 Comet 优化运行实验
        if hasattr(self.opt, "comet_optimizer_id"):
            self.experiment.log_other("optimizer_id", self.opt.comet_optimizer_id)
            self.experiment.log_other("optimizer_objective", self.opt.comet_optimizer_objective)
            self.experiment.log_other("optimizer_metric", self.opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_parameters", json.dumps(self.hyp))

    def log_asset_data(self, asset, **kwargs):
        self.experiment.log_asset_data(asset, **kwargs)

    def log_asset(self, asset_path, **kwargs):
        self.experiment.log_asset(asset_path, **kwargs)

    def _get_experiment(self, mode, experiment_id=None):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(previous_experiment=experiment_id, **self.default_experiment_kwargs)
            return comet_ml.OfflineExperiment(**self.default_experiment_kwargs)
        else:
            try:
                if experiment_id is not None:
                    return comet_ml.ExistingExperiment(previous_experiment=experiment_id, **self.default_experiment_kwargs)
                return comet_ml.Experiment(**self.default_experiment_kwargs)
            except ValueError:
                logger.warning("COMET WARNING: "
                               "Comet credentials have not been set."
                               "Comet will default to offline logging."
                               "Please set your credentials to enable online logging.")
                return self._get_experiment("offline", experiment_id)
        return

    def update_data_paths(self, data_dict):
        path = data_dict.get("path", "")

        for split in ["train", "val", "test"]:
            if data_dict.get(split):
                split_path = data_dict.get(split)
                data_dict[split] = (f"{path}/{split_path}" if isinstance(split, str) else [f"{path}/{x}" for x in split_path])
        return data_dict

    def download_dataset_artifact(self, artifact_path):
        logged_artifact = self.experiment.get_artifact(artifact_path)
        artifact_save_dir = str(Path(self.opt.save_dir) / logged_artifact.name)
        logged_artifact.download(artifact_save_dir)

        metadata = logged_artifact.metadata
        data_dict = metadata.copy()
        data_dict["path"] = artifact_save_dir

        metadata_names = metadata.get("names")
        if isinstance(metadata_names, dict):
            data_dict["names"] = {int(k): v for k, v in metadata.get("names").items()}
        elif isinstance(metadata_names, list):
            data_dict["names"] = {int(k): v for k, v in zip(range(len(metadata_names)), metadata_names)}
        else:
            raise "Invalid 'names' field in dataset yaml file. Please use a list or dictionary"
        return self.update_data_paths(data_dict)

    def check_dataset(self, data_file):
        with open(data_file) as f:
            data_config = yaml.safe_load(f)

        path = data_config.get("path")
        if path and path.startswith(COMET_PREFIX):
            path = data_config["path"].replace(COMET_PREFIX, "")
            data_dict = self.download_dataset_artifact(path)
            return data_dict
        self.log_asset(self.opt.data, metadata={"type": "data-config-file"})
        return check_dataset(data_file)
    def log_parameters(self, log_dict, **kwargs):
        self.experiment.log_parameters(log_dict, **kwargs)