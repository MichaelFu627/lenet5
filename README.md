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

**Draw a digit. See how a neural network actually thinks.**

A live, interactive demo of LeNet-5 — the 1998 paper that put convolutional neural networks on the map — implemented entirely in NumPy. No PyTorch. No TensorFlow. Every gradient hand-derived.

## What you can do

- **Draw a digit on the whiteboard.** The model predicts in real time, even mid-stroke.
- **Watch every layer light up.** All 7 layers (Conv → Pool → Conv → Pool → Conv → FC → Output) show their activations as bright/dim heatmaps. Brighter = a neuron firing harder.
- **Rotate the 3D view.** Drag your mouse over the network diagram to see convolutional feature maps stacked through depth, with 10,000+ fully-connected neuron pathways lighting up the strongest paths.

## What makes it interesting

Most deep-learning demos hide the math behind `model.fit()`. This one has no framework to hide behind.

- **Pure NumPy.** Forward and backward passes for Conv2D, max-pool, fully-connected, tanh, softmax+cross-entropy — all hand-written and verified to **10⁻¹¹ relative error** against numerical gradients.
- **96.3% MNIST test accuracy** after 5 epochs, ~30 minutes on a MacBook Air CPU.
- **11× speedup** over the naive approach by rewriting the conv backward pass to use BLAS-accelerated `matmul` via im2col / col2im, instead of `einsum`.
- **Two visualizations side by side:** detailed 2D feature-map grids + interactive 3D Three.js scene with mouse-controlled camera.

## Run locally

```bash
git clone https://huggingface.co/spaces/MichaelFu627/lenet5
cd lenet5
pip install -r requirements.txt
python -m backend.api.server --port 5001
# Open http://localhost:5001
```

To retrain from scratch (~30 min on a Mac):

```bash
python -m backend.train --epochs 5
```

## Source

[github.com/MichaelFu627/lenet5](https://github.com/MichaelFu627/lenet5)

## Reference

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324.