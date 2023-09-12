# -*- encoding: utf-8 -*-
"""
@File       : test.py
@Time       : 2023/9/11 13:51
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import os.path
from pathlib import Path

import torch.hub


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
                    LOGGER.warning(f"⚠️ Download failure, retrying {i+1}/{retry} {url}...")
                else:
                    LOGGER.warning(f"❌ Failed to download {url}.")

        # 解压文件
        if unzip and success and (f.suffix == ".gz" or is_zipfile())

if __name__ == '__main__':
    download()