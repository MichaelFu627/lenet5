# LeNet-5 from Scratch

A pure-NumPy implementation of LeCun et al. 1998's LeNet-5, with an interactive
frontend that lets you handwrite digits and watch the network's neurons light up.

No PyTorch, TensorFlow, Keras, or any other deep learning framework. Every
forward and backward pass is written by hand. ~96% test accuracy on MNIST after
4 epochs.

## Quick start

```bash
pip install -r requirements.txt

# 1. train (takes ~5 minutes on CPU)
python -m backend.train --epochs 4 --batch 64 --lr 0.1

# 2. run the web app
python -m backend.api.server

# 3. open http://localhost:5000
```

## What's inside

```
lenet5/
├── backend/
│   ├── core/                Atomic, single-purpose components.
│   │   ├── kernels.py         Weight init (LeCun / Xavier / He).
│   │   ├── conv.py            Conv2D with optional sparse connection tables.
│   │   ├── pooling.py         AvgPool2D, MaxPool2D.
│   │   ├── activations.py     Tanh, ScaledTanh, ReLU, Flatten, Softmax.
│   │   ├── fc.py              Fully-connected layer.
│   │   ├── loss.py            Softmax cross-entropy, MSE.
│   │   ├── optimizer.py       SGD, SGD+Momentum.
│   │   └── utils.py           im2col / col2im.
│   ├── model/lenet5.py      Assembles the 7 layers of LeNet-5.
│   ├── data/mnist.py        MNIST loader (sklearn → GitHub mirror fallback).
│   ├── api/server.py        Flask API + static file server.
│   └── train.py             Training script.
├── frontend/                Vanilla HTML/CSS/JS, no build step.
│   ├── index.html
│   ├── style.css
│   └── js/
│       ├── canvas.js          Drawing pad + MNIST-style preprocessing.
│       ├── network-viz.js     The "neurons lighting up" visualization.
│       ├── api.js             Talks to /predict.
│       └── app.js             Wires everything together.
├── lab/                     Jupyter notebooks for tinkering.
│   ├── 01_explore_data.ipynb
│   ├── 02_layer_sanity_check.ipynb   ← gradient checks!
│   └── 03_train_and_inspect.ipynb    ← visualize learned filters
├── models/lenet5.npz        Trained weights (created by train.py).
├── requirements.txt
└── README.md
```

## Architecture (1998 paper)

```
Input  1 @ 32×32 (28×28 MNIST padded with -1)
   │
C1   Conv 5×5  →  6 maps  @ 28×28      (156 params)
   │  tanh
S2   AvgPool 2×2 →  6 maps  @ 14×14
   │
C3   Conv 5×5  → 16 maps  @ 10×10      (sparse connection table, 1516 params)
   │  tanh
S4   AvgPool 2×2 → 16 maps  @ 5×5
   │
C5   Conv 5×5  → 120 maps @ 1×1        (≡ FC, 48,120 params)
   │  tanh
F6   FC 120 → 84                       (10,164 params)
   │  tanh
Out  FC  84 → 10                       (850 params)
   │  softmax + cross-entropy

Total: ~61k parameters
```

The C3 connection table — each of 16 output maps connects to only some of the
6 input maps — is faithfully reproduced from Table 1 of the original paper.
See `backend/model/lenet5.py:C3_CONNECTION_TABLE`.

## The frontend

Three sections:

1. **Handwriting pad** — 280×280 black canvas. Draw with mouse or touch.
   Auto-preprocessed to MNIST format (center-of-mass alignment, 20×20 fit
   inside 28×28, normalized to [-1, 1]). A live 28×28 preview shows you what
   the model actually sees.

2. **Prediction panel** — big predicted digit + probability bars for all 10
   classes.

3. **Network visualization** — every layer of the network rendered as either
   a grid of feature maps (for conv/pool layers) or a column of cells (for FC
   layers). After each prediction the layers light up in order, with brighter
   colors for stronger activations. Click "auto predict" and it re-runs on
   every stroke.

The frontend talks to a single Flask endpoint `POST /predict` that returns
both the prediction and **every layer's activations** so the visualization
can show what's happening inside the network.

## Verifying correctness

`lab/02_layer_sanity_check.ipynb` does **numerical gradient checking** on
every layer. If you touch any forward/backward code, run this first. All
layers verify to ~1e-11 relative error (vs the typical 1e-5 threshold).

## Performance notes

- 4 epochs over 60K MNIST samples: ~5 min on a modern laptop CPU
- Batch size 64–128 is a good speed/accuracy tradeoff
- Conv is the bottleneck; we use im2col + einsum to vectorize it
- Expected accuracy: ~96% (4 epochs), ~98% (15+ epochs with LR schedule)

The original 1998 paper reports 0.95% error rate (99.05% accuracy), but uses
boosting, more epochs, and learned subsampling coefficients we don't bother
with here.

## Tweaking

| Want to... | Edit |
|---|---|
| Try ReLU instead of tanh | swap `Tanh()` → `ReLU()` in `model/lenet5.py` |
| Use max pooling | swap `AvgPool2D` → `MaxPool2D` |
| Disable C3 sparse table | `LeNet5(use_sparse_c3=False)` |
| Switch to LeCun init | change `init="xavier"` → `"lecun"` in `Conv2D`/`FC` |
| Use MSE loss (1998 style) | use `MSE` from `core.loss` + `one_hot_plus_minus(y)` |

## Reference

Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner.
*Gradient-Based Learning Applied to Document Recognition.*
Proceedings of the IEEE, 86(11):2278-2324, November 1998.
