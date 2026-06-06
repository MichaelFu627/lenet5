"""
全连接层 (Fully Connected)

LeNet-5 里的 C5 严格说是 5x5 卷积（输出 1x1），但等价于 FC。
F6 和 Output 是显式的 FC。
"""

import numpy as np
from .kernels import init_fc_weights


class FC:
    def __init__(self, in_dim, out_dim, init="xavier"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W, self.b = init_fc_weights(in_dim, out_dim, method=init)
        self.dW = None
        self.db = None

    def forward(self, x):
        # x: (N, in_dim)
        self.x = x
        return x @ self.W.T + self.b   # (N, out_dim)

    def backward(self, dout):
        # dout: (N, out_dim)
        N = dout.shape[0]
        self.dW = dout.T @ self.x / N   # (out_dim, in_dim)
        self.db = dout.mean(axis=0)
        dx = dout @ self.W              # (N, in_dim)
        return dx

    def params(self):
        return [("W", self.W, "dW"), ("b", self.b, "db")]
