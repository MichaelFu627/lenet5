"""
MNIST 数据加载

支持三种来源：
1. sklearn.fetch_openml (默认尝试)
2. GitHub 镜像 (fallback)
3. 已下载的 ubyte 文件

预处理：归一化到 [-1, 1]，可选 pad 到 32x32 (LeNet-5 标准输入)。
"""

import os
import gzip
import struct
import urllib.request
import numpy as np


# GitHub 镜像：raw 文件就是标准 MNIST ubyte
GITHUB_MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master"
MIRROR_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x":  "t10k-images-idx3-ubyte.gz",
    "test_y":  "t10k-labels-idx1-ubyte.gz",
}


def _normalize(x):
    """[0, 255] -> [-1, 1]"""
    return (x.astype(np.float32) / 255.0) * 2.0 - 1.0


def _read_idx(path):
    """读取 IDX 格式 (MNIST 原始格式)，自动处理 gzip。"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xff
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
    return data


def _download(url, dest):
    """下载文件到 dest，已存在则跳过。"""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)


def load_mnist_github(cache_dir="data_cache"):
    """从 GitHub 镜像下载 MNIST 原始文件。"""
    print("Loading MNIST from GitHub mirror...")
    paths = {}
    for key, fname in MIRROR_FILES.items():
        dest = os.path.join(cache_dir, fname)
        _download(f"{GITHUB_MIRROR}/{fname}", dest)
        paths[key] = dest

    X_train = _read_idx(paths["train_x"])   # (60000, 28, 28)
    y_train = _read_idx(paths["train_y"]).astype(np.int64)
    X_test  = _read_idx(paths["test_x"])
    y_test  = _read_idx(paths["test_y"]).astype(np.int64)

    X_train = _normalize(X_train)[:, None, :, :]
    X_test  = _normalize(X_test)[:, None, :, :]
    return X_train, y_train, X_test, y_test


def load_mnist_sklearn(cache_dir=None):
    """
    用 sklearn 拉 MNIST。如果失败 (例如离线 / 被墙)，自动 fallback 到 GitHub。
    """
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST via sklearn (first run downloads ~10 MB)...")
        X, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False,
            data_home=cache_dir,
        )
        X = X.reshape(-1, 28, 28)
        y = y.astype(np.int64)
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        X_train = _normalize(X_train)[:, None, :, :]
        X_test = _normalize(X_test)[:, None, :, :]
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"  sklearn failed ({e}), falling back to GitHub mirror.")
        return load_mnist_github(cache_dir or "data_cache")


def pad_to_32(x):
    """28x28 -> 32x32, 用 -1 (背景值) 填充."""
    return np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)),
                  mode="constant", constant_values=-1.0)


def iterate_minibatches(X, y, batch_size, shuffle=True):
    """简单的 mini-batch 迭代器。"""
    N = len(X)
    idx = np.random.permutation(N) if shuffle else np.arange(N)
    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
