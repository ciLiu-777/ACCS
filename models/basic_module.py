# coding:utf8
import torch as t
from torch import nn
import time


class BasicModule(nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = "checkpoints/do_results/" + self.model_name + "_"
            name = time.strftime(prefix + "%m%d_%H-%M-%S.pth")
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        """
        赋予优化器
        """
        return t.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
