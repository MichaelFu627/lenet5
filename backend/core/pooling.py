"""
池化层

LeNet-5 原文用的是带可训练参数的 2x2 平均池化：
    y = tanh(coef * sum(2x2 window) + bias)

为了实现简洁这里只做标准 2x2 平均池化（不可训练）。
学习目的足够，对精度影响不大。

同时提供 MaxPool 作为对比。
"""

import numpy as np


class AvgPool2D:
    def __init__(self, k=2, stride=2):
        self.k = k
        self.stride = stride

    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                window = x[:, :, i * s:i * s + k, j * s:j * s + k]
                out[:, :, i, j] = window.mean(axis=(2, 3))
        self.x_shape = x.shape
        return out

    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        k, s = self.k, self.stride
        dx = np.zeros(self.x_shape, dtype=dout.dtype)
        scale = 1.0 / (k * k)
        for i in range(H_out):
            for j in range(W_out):
                # 平均池化：梯度均匀分给窗口里每个位置
                dx[:, :, i * s:i * s + k, j * s:j * s + k] += (
                    dout[:, :, i:i + 1, j:j + 1] * scale
                )
        return dx

    def params(self):
        return []


class MaxPool2D:
    def __init__(self, k=2, stride=2):
        self.k = k
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        # 缓存最大值位置用于 backward
        self.mask = np.zeros_like(x)
        for i in range(H_out):
            for j in range(W_out):
                window = x[:, :, i * s:i * s + k, j * s:j * s + k]
                out[:, :, i, j] = window.max(axis=(2, 3))
                # 把最大值位置标 1
                max_vals = out[:, :, i:i + 1, j:j + 1]
                self.mask[:, :, i * s:i * s + k, j * s:j * s + k] += (
                    window == max_vals
                )
        self.x_shape = x.shape
        return out

    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        k, s = self.k, self.stride
        dx = np.zeros(self.x_shape, dtype=dout.dtype)
        for i in range(H_out):
            for j in range(W_out):
                dx[:, :, i * s:i * s + k, j * s:j * s + k] += (
                    dout[:, :, i:i + 1, j:j + 1]
                    * self.mask[:, :, i * s:i * s + k, j * s:j * s + k]
                )
        return dx

    def params(self):
        return []
