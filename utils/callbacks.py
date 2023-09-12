# -*- encoding: utf-8 -*-
"""
@File       : callbacks.py.py
@Time       : 2023/8/31 17:40
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import threading


class Callbacks():
    """处理模型的所有回调"""
    def __init__(self):
        # 定义变量回调
        self._callbacks = {
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_satrt": [],
            "on_train_batch_satrt": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],
            "on_model_save": [],
            "on_train_end": [],
            "on_params_uodate": [],
            "teardown": []
        }
        self.stop_training = False  # 设置True， 打断训练

    def register_action(self, hook, name="", callback=None):
        """
        将新操作注册到回调hook中
        :param hook: 将操作名 注册到回调hook
        :param name: 动作的名称，供以后参考
        :param callback: 出发的回调
        :return:
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callback {self._callbacks}."
        assert callable(callback) , f"callback '{callback}' is not callable."
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """
        通过回调hook返回所有注册的动作
        :param hook: 要检查的hook名称，默认为all
        :return:
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, thread=False, **kwargs):
        """
        循环遍历已注册的操作，并触发主线程上的所有回调
        :param hook: 要检查的hook名称，默认为all
        :param args: 从模型接收的参数
        :param thread: 在守护线程中运行回调
        :param kwargs: 从模型接收的关键参数
        :return:
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}."
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger["callback"], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger["callback"](*args, **kwargs)