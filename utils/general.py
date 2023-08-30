# -*- encoding: utf-8 -*-
"""
@File       : general.py
@Time       : 2023/8/30 13:39
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

# 导入"ultralytics"包 or 不存在则安装
try:
    import ultralytics
    assert hasattr(ultralytics, "__version__")  # 验证包是否存在
except (ImportError, AssertionError):
    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.checks import check_requirements

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project 根路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT加入到环境变量中
RANK = int(os.getenv("RANK", -1))


class WorkingDirectory(contextlib.ContextDecorator):
    """切换命令路径到指定文件夹路径，Usage：@WorkingDirectory(dir) or 'with WorkingDirectory(dir):'上下文管理器"""
    def __init__(self, new_dir):
        self.dir = new_dir  # 新的文件夹
        self.cwd = Path.cwd().resolve()  # 当前文件夹

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)
@WorkingDirectory(ROOT)
def check_git_info(path="."):
    """检查git信息, 返回 {remote, branch, commit}"""
    check_requirements("gitpython")
    import git
    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace(".git", "")  # i.e. "https://github.com/kungs8/yolov5_learning"
        commit = repo.head.commit.hexsha  # i.e. "b750df5ec0c903b503ceb939d87646b1fc042f56"
        try:
            branch = repo.active_branch.name  # i.e. "main"
        except TypeError:  # 不在任何分支上
            branch = None
        return {"remote": remote, "branch": branch, "commit": commit}
    except git.exc.InvalidGitRepositoryError:  # 路径不是git的路径
        return {"remote": None, "branch": None, "commit": None}

# def print_args(args: Optional[dict]=None, show_file=True, show_func=False):
#     """"""