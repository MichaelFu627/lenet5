"""
Conv2D 层：手写 forward / backward

LeNet-5 的 C3 层有一个 "稀疏连接表"，每个输出通道只接特定几个输入通道。
这里支持通过 connection_table 来实现，table[i] 是一个 list，告诉第 i 个
输出通道连接哪些输入通道。

为了训练速度，forward / backward 全部基于 im2col 向量化。
"""

import numpy as np
from .kernels import init_conv_kernel
from .utils import im2col, col2im


class Conv2D:
    def __init__(self, in_c, out_c, k, stride=1, pad=0, init="xavier",
                 connection_table=None):
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = stride
        self.pad = pad
        self.W, self.b = init_conv_kernel(in_c, out_c, k, method=init)
        self.connection_table = connection_table

        # 如果有稀疏连接表，构造一个 mask 让没连接的权重恒为 0
        if connection_table is not None:
            assert len(connection_table) == out_c
            mask = np.zeros((out_c, in_c, k, k), dtype=np.float32)
            for oc, in_list in enumerate(connection_table):
                for ic in in_list:
                    mask[oc, ic] = 1.0
            self.mask = mask
            self.W *= mask  # 初始化时把不连接的位置清零
        else:
            self.mask = None

        # 梯度缓存（给 optimizer 用）
        self.dW = None
        self.db = None

    def forward(self, x):
        # x: (N, C, H, W)
        self.x_shape = x.shape
        cols, H_out, W_out = im2col(x, self.k, self.stride, self.pad)
        self.cols = cols          # (N, C*k*k, H_out*W_out)
        self.H_out = H_out
        self.W_out = W_out

        N = x.shape[0]
        W_flat = self.W.reshape(self.out_c, -1)   # (out_c, C*k*k)
        # (out_c, C*k*k) @ (N, C*k*k, H_out*W_out) -> (N, out_c, H_out*W_out)
        out = np.einsum("oc,ncl->nol", W_flat, cols) + self.b[None, :, None]
        return out.reshape(N, self.out_c, H_out, W_out)

    def backward(self, dout):
        # dout: (N, out_c, H_out, W_out)
        N = dout.shape[0]
        dout_flat = dout.reshape(N, self.out_c, -1)   # (N, out_c, H_out*W_out)
        W_flat = self.W.reshape(self.out_c, -1)       # (out_c, C*k*k)

        # 参数梯度 (在 batch 维度上求和)
        # dW = sum over N of dout * cols^T
        dW_flat = np.einsum("nol,ncl->oc",
                            dout_flat, self.cols).reshape(self.W.shape) / N
        # 修正：上面把字母搞混了，重新写
        dW_flat = np.einsum("nol,nkl->ok", dout_flat, self.cols) / N
        # dW_flat: (out_c, C*k*k)
        self.dW = dW_flat.reshape(self.W.shape)
        self.db = dout.sum(axis=(0, 2, 3)) / N

        # 稀疏连接：把未连接位置的梯度清零
        if self.mask is not None:
            self.dW *= self.mask

        # 输入梯度
        dcols = np.einsum("ok,nol->nkl", W_flat, dout_flat)   # (N, C*k*k, H_out*W_out)
        dx = col2im(dcols, self.x_shape, self.k, self.stride, self.pad)
        return dx

    def params(self):
        return [("W", self.W, "dW"), ("b", self.b, "db")]
