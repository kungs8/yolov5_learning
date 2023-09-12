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
import random
import subprocess
import time
from datetime import datetime
import glob
import inspect
import logging
import logging.config
import os
import platform
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Optional
from zipfile import is_zipfile, ZipFile
from tarfile import is_tarfile
import torch.hub
import torch.backends.cudnn
import yaml
from multiprocessing.pool import ThreadPool
from utils import emojis, TryExcept
from subprocess import check_output
from itertools import repeat
import numpy as np

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
from utils.downloads import curl_download

# 常参数
LOGGING_NAME = "yolov5"  # 日志的名称
DATASETS_DIR = Path(os.getenv("YOLOv5_DATASETS_DIR", ROOT.parent / "datasets"))  # 全局数据集路径
FONT = "Arial.ttf"  # https://ultralytics.com/assets/Arial.ttf


def is_writeable(dir, test=False):
    """
    如果目录具有写权限，则返回True；如果test=True，则测试打开具有写权限的文件
    :param dir:
    :param test:
    :return:
    """
    if not test:
        return os.access(dir, os.W_OK)  # windows上可能存在的问题
    file = Path(dir) / "tmp.txt"
    try:
        with open(file, "w"):  # 使用写入权限打开文件
            pass
        file.unlink()  # 删除文件
        return True
    except OSError:
        return False


def is_ascii(s=""):
    """
    字符串是否由所有ASCII(不是UTF)字符组成? (注意sr().isascii()是python3.7后引入的)
    :param s:
    :return:
    """
    s = str(s)
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def user_config_dir(dir="Ultralytics", env_var="YOLOV5_CONFIG_DIR"):
    """
    返回用户配置目录的路径.首选环境变量(如果存在)，如果需要，则创建目录
    :param dir: 文件夹名称
    :param env_var: 环境变量
    :return:
    """
    env = os.getenv(env_var)
    if env:
        path = Path(env)
    else:
        cfg = {"Windows": "AppData/Roaming", "Linux": ".config", "Darwin": "Library/Application Support"}  # 3个 OS 路径
        path = Path.home() / cfg.get(platform.system(), "")  # OS特殊配置路径
        path = (path if is_writeable(path) else Path("/tmp")) / dir  # GCP和AWS的lambda修复，只有/tmp 是可写的
    path.mkdir(exist_ok=True)  # 如果有需要则创建
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics设置的路径

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
    url = f"https://github.com{repo}"
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


def check_suffix(file, suffix, msg=""):
    """检查文件的结尾是否可接受"""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_file(file, suffix=""):
    """
    寻找/下载(如果有需要)文件并返回路径
    :param file:
    :param suffix:
    :return:
    """
    check_suffix(file, suffix)  # 可选择
    file = str(file)  # 转变为str()
    if os.path.isfile(file) or not file:  # 存在
        return file
    elif file.startswith(("http:/", "https:/")):  # 下载
        url = file  # 警告:Pathlib 使得 `://` -> `:/`
        file = Path(urllib.parse.unquote(file).split("?")[0]).name  # "%2F" -> "/", 分隔 `https://url.com/file/txt?auth`
        if os.path.isfile(file):
            LOGGER.info(f"Found {url} locally at {file}")  # 文件已经存在
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"  # 下载完成后，检查
        return file
    elif file.startswith("clearml://"):  # ClearML Dataset ID
        assert "clearml" in sys.modules, "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # 搜索
        files = []
        for d in "data", "models", "utils":  # 搜索字典
            files.extend(glob.glob(f"{ROOT}/{d}/**/{file}", recursive=True))  # 寻找文件
        assert len(files), f"File not found: {file}"  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # 返回文件


def get_latest_run(search_dir="."):
    """返回`/runs`最近的`latest.pt`文件路径. i.e. to --resume from"""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)  # recursive: 是否在子目录中递归搜索
    return max(last_list, key=os.path.getctime) if last_list else ""


def check_yaml(file, suffix=(".yaml", ".yml")):
    """寻找/下载 YAML 文件(如果需要), 并返回路径，检查后缀"""
    return check_file(file, suffix)


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """增加文件 or 文件夹路径，i.e. runs/exp -> runs/exp{sep}2, runs/{sep}3, ..., etc"""
    path = Path(path)  # 与操作系统无关
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # 方法1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # 增加的路径
            if not os.path.exists(p):
                break
        path = Path(p)

        # # 方法2
        # dirs = glob.glob(f"{path}{sep}")  # 相似的路径
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # 索引
        # n = max(i) + 1 if i else 2  # 增加序号
        # path = Path(f"{path}{sep}{n}{suffix}")  # 增加的路径

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # 创建文件夹
    return path


def git_describe(path=ROOT):  # path必须是一个文件夹
    """返回人类可读的git描述，i.e. v5.0-5-g3e25file https://git-scm.com/docs/git-describe"""
    try:
        assert (Path(path) / ".git").is_dir()
        return check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    except Exception:
        return ""


def file_date(path=__file__):
    """返回人类可读的文件修改日前. i.e. ‘2023-08-31’"""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def yaml_load(file="data.yaml"):
    """单行的 yaml 安全加载"""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)

def yaml_save(file="data.yaml", data={}):
    """单行安全保存yaml"""
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    """
    将`*.zip`文件解压缩到指定路径，排除包含排除列表中字符串到文件
    :param file: 压缩文件
    :param path: 存储到指定路径
    :param exclude: 排除的文件
    :return:
    """
    if path is None:
        path = Path(file).parent  # 默认路径
    with ZipFile(file) as zipObject:
        for f in zipObject.namelist():  # 列举zip中所有存档的文件名
            if all(x not in f for x in exclude):
                zipObject.extract(f, path=path)


def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1, retry=3):
    """
    多线程文件下载和解压缩功能，用于data.yaml的自动下载
    :param url: 数据路径
    :param dir: 存储路径
    :param unzip: 是否解压
    :param delete: 是否删除
    :param curl: 是否使用curl下载
    :param threads: 线程数
    :param retry: 尝试次数
    :return:
    """
    def download_one(url, dir):
        # 下载1个文件
        success = True
        if os.path.isfile(url):
            f = Path(url)  # 文件名称
        else:
            f = dir / Path(url).name
            LOGGER.info(f"Downloading {url} to {f}...")
            for i in range(retry + 1):
                if curl:
                    success = curl_download(url, f, silent=(threads > 1))
                else:
                    torch.hub.download_url_to_file(url=url, dst=f, progress=threads == 1)  # torch下载
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {url}...")
                else:
                    LOGGER.warning(f"❌ Failed to download {url}.")

        # 解压文件
        if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir)  # 解压缩
            elif is_tarfile(f):
                subprocess.run(["tar", "xf", f, "--directory", f.parent], check=True)  # 解压
            elif f.suffix == ".gz":
                subprocess.run(["tar", "xfz", f, "--directory", f.parent], check=True)  # 解压
            if delete:
                f.unlink()  # 删除压缩包

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # 创建文件夹
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # 多线程
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def check_font(font=FONT, progress=False):
    """
    下载字体去CONFIG_DIR, 如果有需要
    :param font: 字体
    :param progress:
    :return:
    """
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f"https://ultralytics.com/assets/{font.name}"
        LOGGER.info(f"Downloading {url} to {file} ...")
        torch.hub.download_url_to_file(url, str(file), progress=progress)

def check_dataset(data, autodownload=True):
    """下载，检查/解压 数据，如果本地不存在"""
    # 下载
    extract_dir = ""
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f"{DATASETS_DIR}/{Path(data).stem}", unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob("*.yaml"))
        extract_dir, autodownload = data.parent, False

    # 读取yaml(可选)
    if isinstance(data, (str, Path)):
        data = yaml_load(file=data)  # yaml的数据

    # 检查
    for k in ["train", "val", "names"]:
        assert k in data, emojis(f"data.yaml '{k}:' field missing ❌")
    if isinstance(data["names"], (list, tuple)):  # 旧数组格式
        data["names"] = dict(enumerate(data["names"]))  # 将其转为字典形式
    assert all(isinstance(k, int) for k in data["names"].keys()), "data.yaml names keys must be integers, i.e. 2:car"
    data["nc"] = len(data["names"])

    # 解析路径
    path = Path(extract_dir or data.get("path") or "")  # 可选的路径默认为 `.`
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path  # 下载脚本
    for k in ["train", "val", "test"]:
        if data.get(k):  # 预备路径
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # 解析yaml
    train, val, test, s = (data.get(x) for x in ("train", "val", "test", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # 验证集的路径
        if not all(x.exists() for x in val):
            LOGGER.info(f"\nDataset not found ⚠️,missing paths {[str(x) for x in val if not x.exists()]}")
            if not s or not autodownload:
                raise Exception("Dataset not found ❌.")
            t = time.time()
            if s.startswith("http") and s.endswith(".zip"):  # URL
                f = Path(s).name  # 文件名
                LOGGER.info(f"Downloading {s} to {f}...")
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  # 创建根路径
                unzip_file(f, path=DATASETS_DIR)  # 解压
                Path(f).unlink()  # 删除压缩包
                r = None  # 成功
            elif s.startswith("bash "):  # bash脚本
                LOGGER.info(f"Running {s} ...")
                r = subprocess.run(s, shell=True)
            else:
                r = exec(s, {"yaml": data})  # 返回None
            dt = f"({round(time.time() - t, 1)}s"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"failure {dt} ❌."
            LOGGER.info(f"Dataset download {s}.")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf", progress=True)  # 下载字体
    return data  # 字典


def methods(instance):
    """
    获取 类/实例的方法
    :param instance:
    :return:
    """
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def check_version(current="0.0.0", minimum="0.0.0", name="version", pinned=False, hard=False, verbose=False):
    """
    检查版本 vs.需求版本
    :param current:
    :param minimum:
    :param name:
    :param pinned:
    :param hard:
    :param verbose:
    :return:
    """
    current, minimum = str(current), str(minimum)
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed."
    if hard:
        assert result, emojis(s)  # assert满足最低要求
    if verbose and (not result):
        LOGGER.warning(s)
    return result


def init_seeds(seed=0, deterministric=False):
    """初始化随机数生成器(Random number generator, RNG)种子，https://pytorch.org/docs/stable/notes/randomness.html"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对于多GPU，异常安全
    # torch.backends.cudnn.benchmark = True  # AutoBatch问题. https://github.com/ultralytics/yolov5/issues/9287
    if deterministric and check_version(torch.__version__, "1.12.0"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
