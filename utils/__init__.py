# -*- encoding: utf-8 -*-
"""
@File       : __init__.py
@Time       : 2023/8/30 17:04
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import contextlib
import platform


def emojis(str=""):
    """返回字符串的平台相关表情符号安全版本"""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    """项目tryExcept 类. Usage: @TryExcept() 装饰 or 'with TryExcept():' 上下文管理器"""
    def __init__(self, msg=""):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{exc_val}"))
        return True