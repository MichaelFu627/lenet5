"""
损失函数

提供两个：
1. SoftmaxCrossEntropy - 现代标准，分类首选
2. MSE - LeNet-5 原文风格，需要把 label 编码成 {+1, -1}

为了精度和稳定性，默认用 SoftmaxCrossEntropy。
"""

import numpy as np
from .activations import softmax


class SoftmaxCrossEntropy:
    """
    把 softmax + cross-entropy 合并算，避免数值不稳定。
    输入: logits (N, C)，标签 y (N,) 是整数类别。
    """

    def forward(self, logits, y):
        self.probs = softmax(logits, axis=-1)
        N = logits.shape[0]
        # 取出正确类别的概率
        correct_probs = self.probs[np.arange(N), y]
        # 避免 log(0)
        correct_probs = np.clip(correct_probs, 1e-12, 1.0)
        loss = -np.log(correct_probs).mean()
        self.y = y
        return loss

    def backward(self):
        # dL/dlogits = (probs - one_hot(y)) / N
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y] -= 1.0
        return grad / N


class MSE:
    """1989/1998 LeNet 风格：MSE 配 tanh 输出 + {+1,-1} 编码."""

    def forward(self, pred, target):
        # pred / target 都是 (N, C)
        self.diff = pred - target
        return np.mean(self.diff ** 2)

    def backward(self):
        N = self.diff.shape[0]
        return 2.0 * self.diff / (N * self.diff.shape[1])


def one_hot_plus_minus(y, num_classes=10):
    """把整数 label 转成 {+1, -1} 编码（给 MSE 用）"""
    N = len(y)
    T = -np.ones((N, num_classes), dtype=np.float32)
    T[np.arange(N), y] = 1.0
    return T
