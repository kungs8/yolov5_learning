# -*- encoding: utf-8 -*-
"""
@File       : __init__.py
@Time       : 2023/8/30 17:04
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import platform


def emojis(str=""):
    """返回字符串的平台相关表情符号安全版本"""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str
