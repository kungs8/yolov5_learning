# -*- encoding: utf-8 -*-
"""
@File       : downloads.py
@Time       : 2023/8/31 15:12
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import subprocess
import urllib.request
import urllib.parse


def is_url(url, check=True):
    """检查如果`string`是 URL 和 检查如果 URL存在"""
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # 检查如果是url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # 检查如果在线存在
    except (AssertionError, urllib.request.HTTPError):
        return False


def curl_download(url, filename, *, silent: bool=False) -> bool:
    """
    使用curl将文件从url下载到文件路径下
    :param url: 数据链接
    :param filename: 文件路径名
    :param silent: 静默模式下运行，不显示进度条，但会显示错误消息（如果有），并且会自动跟随重定向
    :return:
    """
    silent_option = "sS" if silent else ""
    proc = subprocess.run(["curl", "-#", f"-{silent_option}L", url, "--output", filename, "--retry", 9, "-C", "-"])
    return proc.returncode == 0