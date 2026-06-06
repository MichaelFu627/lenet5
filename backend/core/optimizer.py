"""
优化器

提供 SGD 和 SGD with momentum。
LeNet-5 原文用的是 stochastic diagonal Levenberg-Marquardt，
工程上 SGD + momentum 效果接近且简单很多，这里就用后者。
"""

import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            for name, param, grad_name in layer.params():
                grad = getattr(layer, grad_name)
                if grad is not None:
                    param -= self.lr * grad


class Momentum:
    """带动量的 SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.mu = momentum
        self.velocity = {}   # 用 id(param) 做 key

    def step(self, layers):
        for layer in layers:
            for name, param, grad_name in layer.params():
                grad = getattr(layer, grad_name)
                if grad is None:
                    continue
                key = id(param)
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(param)
                v = self.velocity[key]
                v *= self.mu
                v -= self.lr * grad
                param += v
