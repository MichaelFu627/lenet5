"""
LeNet-5: Yann LeCun 1998

经典架构 (Input 32x32 padded from 28x28 MNIST):

    Input  : 1  @ 32x32
    C1     : Conv 5x5, 6 maps           ->  6 @ 28x28
    S2     : AvgPool 2x2                ->  6 @ 14x14
    C3     : Conv 5x5, 16 maps (sparse) -> 16 @ 10x10
    S4     : AvgPool 2x2                -> 16 @  5x5
    C5     : Conv 5x5, 120 maps         -> 120 @ 1x1   (≡ FC)
    F6     : FC, 84 units
    Output : FC, 10 units (softmax)

激活：tanh 在所有隐藏层
损失：softmax + cross-entropy

C3 的稀疏连接表：每列对应 1 个输出 map，X 表示连接哪些输入 map。
这是 LeNet-5 原论文 Table 1 的内容。
"""

import numpy as np

from ..core import (
    Conv2D, AvgPool2D, FC, Tanh, Flatten, SoftmaxCrossEntropy, softmax
)


# C3 连接表：6 个输入 map -> 16 个输出 map
# 每行 = 一个输出 map 连接的输入 map 列表
C3_CONNECTION_TABLE = [
    [0, 1, 2],          # 0
    [1, 2, 3],          # 1
    [2, 3, 4],          # 2
    [3, 4, 5],          # 3
    [0, 4, 5],          # 4
    [0, 1, 5],          # 5
    [0, 1, 2, 3],       # 6
    [1, 2, 3, 4],       # 7
    [2, 3, 4, 5],       # 8
    [0, 3, 4, 5],       # 9
    [0, 1, 4, 5],       # 10
    [0, 1, 2, 5],       # 11
    [0, 1, 3, 4],       # 12
    [1, 2, 4, 5],       # 13
    [0, 2, 3, 5],       # 14
    [0, 1, 2, 3, 4, 5], # 15
]


class LeNet5:
    """
    LeNet-5，纯 NumPy 实现。
    
    forward 默认只返回 logits；如果 collect_activations=True，
    会同时返回每层激活，给前端可视化用。
    """

    def __init__(self, use_sparse_c3=True):
        # C1: 1 -> 6, kernel 5x5
        self.c1 = Conv2D(in_c=1, out_c=6, k=5)
        self.c1_act = Tanh()
        # S2: 2x2 avg pool
        self.s2 = AvgPool2D(k=2, stride=2)
        # C3: 6 -> 16, kernel 5x5，带稀疏连接表
        ct = C3_CONNECTION_TABLE if use_sparse_c3 else None
        self.c3 = Conv2D(in_c=6, out_c=16, k=5, connection_table=ct)
        self.c3_act = Tanh()
        # S4: 2x2 avg pool
        self.s4 = AvgPool2D(k=2, stride=2)
        # C5: 16 -> 120, kernel 5x5 (输出 1x1，等价 FC)
        self.c5 = Conv2D(in_c=16, out_c=120, k=5)
        self.c5_act = Tanh()
        # Flatten 120@1x1 -> 120
        self.flatten = Flatten()
        # F6: 120 -> 84
        self.f6 = FC(120, 84)
        self.f6_act = Tanh()
        # Output: 84 -> 10
        self.out_fc = FC(84, 10)

        self.loss_fn = SoftmaxCrossEntropy()

    def _trainable_layers(self):
        """所有有参数的层，按 forward 顺序排列，给 optimizer 用."""
        return [self.c1, self.c3, self.c5, self.f6, self.out_fc]

    def _all_layers(self):
        """所有层，按 forward 顺序排列，给 backward 用."""
        return [
            self.c1, self.c1_act,
            self.s2,
            self.c3, self.c3_act,
            self.s4,
            self.c5, self.c5_act,
            self.flatten,
            self.f6, self.f6_act,
            self.out_fc,
        ]

    def forward(self, x, collect_activations=False):
        """
        x: (N, 1, 32, 32)，已经 pad 过的 MNIST。
        如果只有 (N, 1, 28, 28)，会自动 pad。
        """
        if x.shape[-1] == 28:
            x = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)),
                       mode="constant", constant_values=-1.0)

        a_c1 = self.c1_act.forward(self.c1.forward(x))      # (N, 6, 28, 28)
        a_s2 = self.s2.forward(a_c1)                         # (N, 6, 14, 14)
        a_c3 = self.c3_act.forward(self.c3.forward(a_s2))   # (N, 16, 10, 10)
        a_s4 = self.s4.forward(a_c3)                         # (N, 16, 5, 5)
        a_c5 = self.c5_act.forward(self.c5.forward(a_s4))   # (N, 120, 1, 1)
        a_flat = self.flatten.forward(a_c5)                  # (N, 120)
        a_f6 = self.f6_act.forward(self.f6.forward(a_flat)) # (N, 84)
        logits = self.out_fc.forward(a_f6)                   # (N, 10)

        if collect_activations:
            activations = {
                "C1": a_c1,
                "S2": a_s2,
                "C3": a_c3,
                "S4": a_s4,
                "C5": a_c5.reshape(a_c5.shape[0], -1),
                "F6": a_f6,
                "logits": logits,
                "probs": softmax(logits, axis=-1),
            }
            return logits, activations
        return logits

    def backward(self, dlogits):
        """从 loss 的 backward 反向走一遍。"""
        d = self.out_fc.backward(dlogits)
        d = self.f6_act.backward(d)
        d = self.f6.backward(d)
        d = self.flatten.backward(d)
        d = self.c5_act.backward(d)
        d = self.c5.backward(d)
        d = self.s4.backward(d)
        d = self.c3_act.backward(d)
        d = self.c3.backward(d)
        d = self.s2.backward(d)
        d = self.c1_act.backward(d)
        d = self.c1.backward(d)
        return d

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)

    # 权重保存 / 加载
    def save(self, path):
        params = {}
        for name, layer in [("c1", self.c1), ("c3", self.c3),
                            ("c5", self.c5), ("f6", self.f6),
                            ("out_fc", self.out_fc)]:
            params[f"{name}.W"] = layer.W
            params[f"{name}.b"] = layer.b
        np.savez(path, **params)

    def load(self, path):
        params = np.load(path)
        for name, layer in [("c1", self.c1), ("c3", self.c3),
                            ("c5", self.c5), ("f6", self.f6),
                            ("out_fc", self.out_fc)]:
            layer.W = params[f"{name}.W"]
            layer.b = params[f"{name}.b"]
        # 如果 c3 有稀疏 mask 要重新应用
        if self.c3.mask is not None:
            self.c3.W *= self.c3.mask
