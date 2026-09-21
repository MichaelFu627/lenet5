---
title: LeNet-5 From Scratch
emoji: 🔢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Hand-drawn digits, hand-coded CNN. Live activations.
---

# LeNet-5 From Scratch ✍️

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/MichaelFu627/lenet5)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Made with NumPy](https://img.shields.io/badge/Made%20with-NumPy-013243?logo=numpy&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-0.9935-20BEFF?logo=kaggle&logoColor=white)

**Draw a digit. See how a neural network actually thinks.**

A live, interactive demo of LeNet-5 — the 1998 paper that put convolutional neural networks on the map — implemented entirely in NumPy. No PyTorch. No TensorFlow. Every gradient hand-derived. The UI is bilingual (English / 中文) with instant switching.

## 🚀 [Try the live demo →](https://huggingface.co/spaces/MichaelFu627/lenet5)

## 🖱️ What you can do

- **Draw a digit on the whiteboard.** The model predicts in real time, even mid-stroke.
- **Watch every layer light up.** All 7 layers (Conv → Pool → Conv → Pool → Conv → FC → Output) show their activations as bright/dim heatmaps. Brighter = a neuron firing harder.
- **Rotate the 3D view.** Drag your mouse over the network diagram to see convolutional feature maps stacked through depth, with 10,000+ fully-connected neuron pathways lighting up the strongest paths.

## 🧠 What makes it interesting

Most deep-learning demos hide the math behind `model.fit()`. This one has no framework to hide behind.

- **Pure NumPy, no autograd.** Conv2D, max/avg-pool, fully-connected, tanh, softmax+cross-entropy — every forward and backward pass hand-written.
- **Faithful to the 1998 paper.** The C3 layer keeps LeCun's original sparse connection table (Table 1): each of the 16 feature maps draws from a hand-specified subset of the 6 inputs, not all of them.
- **Gradients verified to 10⁻¹¹.** Hand-derived backprop matches numerical gradients to 10⁻¹¹ relative error.

## 🏗️ Architecture

```text
Input → [C1: Conv] → [S2: Pool] → [C3: Conv] → [S4: Pool] → [C5: Conv] → [F6: FC] → [Output: FC]
```

| # | Layer | Type | Output shape |
| --- | --- | --- | --- |
| 0 | Input | 28×28 MNIST (padded → 32×32) | 1×28×28 |
| 1 | C1 | Conv 5×5, tanh | 6×28×28 |
| 2 | S2 | AvgPool 2×2 | 6×14×14 |
| 3 | C3 | Conv 5×5 (sparse), tanh | 16×10×10 |
| 4 | S4 | AvgPool 2×2 | 16×5×5 |
| 5 | C5 | Conv 5×5, tanh | 120×1×1 |
| 6 | F6 | FC 120→84, tanh | 84 |
| 7 | Output | FC 84→10, softmax | 10 |

## 📊 Numbers

| Metric | Value |
| --- | --- |
| Parameters | 60,806 (~61k) |
| MNIST test accuracy | 96.3% (5 epochs) |
| Training time | ~30 min on a MacBook Air CPU |
| Conv backward speedup | 11× (im2col → BLAS `matmul`) |

> 🏅 Also submitted to **Kaggle Digit Recognizer** — scored **0.9935** with the PyTorch companion notebook.

## 💻 Run locally

```bash
git clone https://huggingface.co/spaces/MichaelFu627/lenet5
cd lenet5
pip install -r requirements.txt
python -m backend.api.server --port 5001
# Open http://localhost:5001
```

The Flask server serves both the frontend and the API from the same origin — no separate build step.

To retrain from scratch (~30 min on a Mac):

```bash
python -m backend.train --epochs 5
```

## 🔗 Source

[github.com/MichaelFu627/lenet5](https://github.com/MichaelFu627/lenet5)

## 📚 Reference

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324.

## 🙏 Acknowledgments

- **LeCun et al. (1998)** for the architecture this project re-implements from scratch.
- **NumPy**, **scikit-learn**, **Flask**, and **Three.js** — the libraries that make the demo possible.
- **Kaggle Digit Recognizer** for the MNIST data.
