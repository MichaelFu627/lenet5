"""
im2col / col2im 工具函数

为了让纯 NumPy 的卷积跑得动 MNIST，必须把卷积转成矩阵乘法。
这里支持批处理（N 维），所以训练时可以一次喂多个样本。
"""

import numpy as np


def im2col(x, k, stride=1, pad=0, pad_value=0.0):
    """
    x: (N, C, H, W)
    返回: (N, C*k*k, H_out*W_out), H_out, W_out
    """
    N, C, H, W = x.shape
    if pad > 0:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant",
            constant_values=pad_value,
        )
    H_p, W_p = x.shape[2], x.shape[3]
    H_out = (H_p - k) // stride + 1
    W_out = (W_p - k) // stride + 1

    cols = np.zeros((N, C, k, k, H_out, W_out), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j, :, :] = x[
                :, :, i : i + stride * H_out : stride, j : j + stride * W_out : stride
            ]
    # (N, C, k, k, H_out, W_out) -> (N, C*k*k, H_out*W_out)
    cols = cols.reshape(N, C * k * k, H_out * W_out)
    return cols, H_out, W_out


def col2im(cols, x_shape, k, stride=1, pad=0):
    """
    cols: (N, C*k*k, H_out*W_out)
    x_shape: 原始输入形状 (N, C, H, W)
    返回 dx: (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_p, W_p = H + 2 * pad, W + 2 * pad
    H_out = (H_p - k) // stride + 1
    W_out = (W_p - k) // stride + 1

    cols = cols.reshape(N, C, k, k, H_out, W_out)
    dx = np.zeros((N, C, H_p, W_p), dtype=cols.dtype)
    for i in range(k):
        for j in range(k):
            dx[
                :, :, i : i + stride * H_out : stride, j : j + stride * W_out : stride
            ] += cols[:, :, i, j, :, :]

    if pad > 0:
        return dx[:, :, pad:-pad, pad:-pad]
    return dx
