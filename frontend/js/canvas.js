// 手写画布 + MNIST 风格预处理
// 关键: 280x280 用户画布 -> 找到笔迹边界 -> 居中缩放到 20x20 -> pad 到 28x28
// 这是 MNIST 数据集本身的处理流程，否则模型会很不准。

const DrawingPad = (() => {
  const canvas = document.getElementById("drawing-canvas");
  const ctx = canvas.getContext("2d");
  const preCanvas = document.getElementById("preprocessed-canvas");
  const preCtx = preCanvas.getContext("2d");
  preCtx.imageSmoothingEnabled = false;

  let drawing = false;
  let lastX = 0, lastY = 0;
  let onStrokeEnd = null;    // 笔迹结束时的回调（清空后用得上）
  let onStrokeMove = null;   // 笔尖每移动时触发（实时预测用）

  // 初始化：黑底
  function clear() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    preCtx.fillStyle = "#000";
    preCtx.fillRect(0, 0, preCanvas.width, preCanvas.height);
  }
  clear();

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    const t = e.touches ? e.touches[0] : e;
    return [
      (t.clientX - r.left) * (canvas.width / r.width),
      (t.clientY - r.top) * (canvas.height / r.height),
    ];
  }

  function start(e) {
    e.preventDefault();
    drawing = true;
    [lastX, lastY] = pos(e);
  }

  function draw(e) {
    if (!drawing) return;
    e.preventDefault();
    const [x, y] = pos(e);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
    if (onStrokeMove) onStrokeMove();
  }

  function end(e) {
    if (!drawing) return;
    drawing = false;
    if (onStrokeEnd) onStrokeEnd();
  }

  canvas.addEventListener("mousedown",  start);
  canvas.addEventListener("mousemove",  draw);
  canvas.addEventListener("mouseup",    end);
  canvas.addEventListener("mouseleave", end);
  canvas.addEventListener("touchstart", start);
  canvas.addEventListener("touchmove",  draw);
  canvas.addEventListener("touchend",   end);

  // ──────────────────────────────────────────────────────────────────────
  // 预处理：280x280 -> 28x28 MNIST 风格
  // ──────────────────────────────────────────────────────────────────────
  function getMnistInput() {
    const w = canvas.width, h = canvas.height;
    const img = ctx.getImageData(0, 0, w, h);
    const data = img.data;

    // 1. 转灰度并提取笔迹 (白色 = 笔迹)
    const gray = new Float32Array(w * h);
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const v = data[i] / 255;   // R 通道（笔是白色，所以三个通道相等）
        gray[y * w + x] = v;
        if (v > 0.1) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    // 没有笔迹 -> 返回全 0
    if (maxX < 0) {
      const empty = new Float32Array(28 * 28);
      drawPreprocessed(empty);
      return empty;
    }

    // 2. 裁剪到笔迹边界 (加点 padding)
    const pad = 10;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(w - 1, maxX + pad);
    maxY = Math.min(h - 1, maxY + pad);
    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;

    // 3. 把裁剪区域缩放到 20x20，保持纵横比
    //    (MNIST 的数字是 20x20，然后 center-pad 到 28x28)
    const scale = 20 / Math.max(cropW, cropH);
    const newW = Math.round(cropW * scale);
    const newH = Math.round(cropH * scale);

    // 用临时 canvas 做缩放
    const tmp = document.createElement("canvas");
    tmp.width = newW;
    tmp.height = newH;
    const tctx = tmp.getContext("2d");
    tctx.imageSmoothingEnabled = true;
    tctx.imageSmoothingQuality = "high";
    tctx.drawImage(canvas, minX, minY, cropW, cropH, 0, 0, newW, newH);
    const smallData = tctx.getImageData(0, 0, newW, newH).data;

    // 4. 计算 center of mass 用于居中（MNIST 风格）
    let mx = 0, my = 0, total = 0;
    const small = new Float32Array(newW * newH);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const v = smallData[(y * newW + x) * 4] / 255;
        small[y * newW + x] = v;
        mx += x * v;
        my += y * v;
        total += v;
      }
    }
    if (total > 0) { mx /= total; my /= total; }
    else           { mx = newW / 2; my = newH / 2; }

    // 5. 放进 28x28，让 (mx, my) 落到 (14, 14)
    const out = new Float32Array(28 * 28);
    const ox = Math.round(14 - mx);
    const oy = Math.round(14 - my);
    for (let y = 0; y < newH; y++) {
      const ty = y + oy;
      if (ty < 0 || ty >= 28) continue;
      for (let x = 0; x < newW; x++) {
        const tx = x + ox;
        if (tx < 0 || tx >= 28) continue;
        out[ty * 28 + tx] = small[y * newW + x];
      }
    }

    drawPreprocessed(out);
    return out;
  }

  function drawPreprocessed(arr28) {
    // 在小预览框里画出 28x28
    const tmp = document.createElement("canvas");
    tmp.width = 28;
    tmp.height = 28;
    const tctx = tmp.getContext("2d");
    const img = tctx.createImageData(28, 28);
    for (let i = 0; i < 28 * 28; i++) {
      const v = Math.max(0, Math.min(255, Math.round(arr28[i] * 255)));
      img.data[i * 4 + 0] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    tctx.putImageData(img, 0, 0);
    preCtx.imageSmoothingEnabled = false;
    preCtx.clearRect(0, 0, preCanvas.width, preCanvas.height);
    preCtx.drawImage(tmp, 0, 0, preCanvas.width, preCanvas.height);
  }

  return {
    clear,
    getMnistInput,
    setOnStrokeEnd(fn)  { onStrokeEnd  = fn; },
    setOnStrokeMove(fn) { onStrokeMove = fn; },
  };
})();
