"""
激活函数

提供 tanh / scaled tanh / ReLU / Softmax。
LeNet-5 原文用 scaled tanh：1.7159 * tanh(2/3 * x)
现代实现一般直接 tanh 或 ReLU 都能 work，所以这里都保留。
"""

import numpy as np


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out ** 2)

    def params(self):
        return []


class ScaledTanh:
    """LeNet-5 原文激活: y = 1.7159 * tanh(2/3 * x)"""
    A = 1.7159
    S = 2.0 / 3.0

    def forward(self, x):
        self.t = np.tanh(self.S * x)
        return self.A * self.t

    def backward(self, dout):
        # dy/dx = A * S * (1 - tanh^2(S*x))
        return dout * self.A * self.S * (1.0 - self.t ** 2)

    def params(self):
        return []


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def params(self):
        return []


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def params(self):
        return []


def softmax(x, axis=-1):
    """数值稳定的 softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
