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
import inspect
import logging
import logging.config
import os
import platform
import re
import sys
from pathlib import Path
from typing import Optional
from utils import emojis, TryExcept
from subprocess import check_output
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
# 常参数
LOGGING_NAME = "yolov5"  # 日志的名称

# 日志配置
def set_logging(name=LOGGING_NAME, verbose=True):
    """给定name，设置logging"""
    rank = int(os.getenv("RANK", -1))  # 全局多 GPU 训练的秩
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {"format": "%(message)s"}},
        "handlers": {
            name: {"class": "logging.StreamHandler",
                   "formatter": name,
                   "level": level}},
        "loggers": {
            name: {"level": level,
                   "handlers": [name],
                   "propagate": False}}
    })
# 运行之前，定义好LOGGER
set_logging(LOGGING_NAME)
# 定义全局日志(应用在train.py, val.py, detect.py, etc.)
LOGGER = logging.getLogger(LOGGING_NAME)
# 适应windows 的日志
if platform.system() == "Windows":
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # 表情安全记录


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


def colorstr(*input):
    """为字符串着色(https://en.wikipedia.org/wiki/ANSI_escape_code), i.e. colorstr('blue', 'hello python')"""
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # 颜色参数, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def print_args(args: Optional[dict]=None, show_file=True, show_func=False):
    """打印函数参数（可选参数字典）"""
    x = inspect.currentframe().f_back  # 前一帧,i.e. <frame at 0x7fbe21d13920, file '/Users/gyp/GGG/kungs/learning/yolov5_learning/train.py', line 79, code main>
    file, _, func, _, _ = inspect.getframeinfo(x)  # i.e. file=/Users/gyp/GGG/kungs/learning/yolov5_learning/train.py, func=main
    if args is None:  # 自动获取参数
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}

    # 获取文件路径的基本名称（即不包含文件扩展名部分）
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")  # 获取不带后缀的前一帧文件名
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def check_online():
    """互联网连接检查"""
    import socket
    def run_once():
        # 检查一次
        try:
            socket.create_connection(("1.1.1.1", 443), 5)  # 检查主机的可访问性
            return True
        except OSError:
            return False
    return run_once() or run_once()  # 检查两次以提高对间歇性连接问题的稳健性


@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo="kungs8/yolov5_learning", branch="main"):
    """项目状态检查，推荐'git pull' 如果代码超出时间"""
    url= f"https://github.com{repo}"
    msg = f", for updates see {url}"
    s = colorstr("github: ")  # string
    assert Path(".git").exists(), s + "跳过检查（不是 git 存储库）" + msg
    assert check_online(), s + "跳过检查(离线)" + msg

    splits = re.split(pattern=r"\s", string=check_output("git remote -v", shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = "kungs8"
        check_output(f"git remote add {remote} {url}", shell=True)
    check_output(f"git fetch {remote}", shell=True, timeout=5)  # git fetch
    local_branch = check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()  # checked out
    # 计算在本地分支和远程仓库的分支之间有多少个提交
    n = int(check_output(f"git rev-list {local_branch}..{remote}/{branch} --count", shell=True))
    if n > 0:
        pull = "git pull" if remote == "origin" else f"git pull {remote}/{branch}"
        s += f"⚠️ yolov5_learning is out of date by {n} commit{'s'*(n>1)}. Use '{pull}' or 'git clone {url}' to update."
    else:
        s += f"up to date with {url} ✅"
    LOGGER.info(s)