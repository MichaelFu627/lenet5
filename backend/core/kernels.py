"""
权重 / kernel 初始化

LeNet-5 原文用的是 U[-2.4/F, 2.4/F]，F = fan_in。
这里同时提供 Xavier / He 初始化方便对比。
"""

import numpy as np


def lecun_uniform(shape, fan_in):
    """LeCun 1998 论文的初始化方式: U[-2.4/F, 2.4/F]."""
    a = 2.4 / fan_in
    return np.random.uniform(-a, a, shape).astype(np.float32)


def xavier_uniform(shape, fan_in, fan_out):
    """Xavier/Glorot 均匀初始化，适合 tanh / sigmoid."""
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-a, a, shape).astype(np.float32)


def he_normal(shape, fan_in):
    """He 初始化，适合 ReLU."""
    std = np.sqrt(2.0 / fan_in)
    return (np.random.randn(*shape) * std).astype(np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def init_conv_kernel(in_c, out_c, k, method="xavier"):
    """生成一个卷积层的权重和 bias."""
    shape = (out_c, in_c, k, k)
    fan_in = in_c * k * k
    fan_out = out_c * k * k
    if method == "lecun":
        W = lecun_uniform(shape, fan_in)
    elif method == "xavier":
        W = xavier_uniform(shape, fan_in, fan_out)
    elif method == "he":
        W = he_normal(shape, fan_in)
    else:
        raise ValueError(f"Unknown init method: {method}")
    b = zeros(out_c)
    return W, b


def init_fc_weights(in_dim, out_dim, method="xavier"):
    """生成一个全连接层的权重和 bias."""
    shape = (out_dim, in_dim)
    if method == "lecun":
        W = lecun_uniform(shape, in_dim)
    elif method == "xavier":
        W = xavier_uniform(shape, in_dim, out_dim)
    elif method == "he":
        W = he_normal(shape, in_dim)
    else:
        raise ValueError(f"Unknown init method: {method}")
    b = zeros(out_dim)
    return W, b
