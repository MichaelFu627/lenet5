// 神经网络可视化
//
// 每层用 .layer 容器。
// 卷积层用一组 <canvas>，每个画一个特征图。
// FC 层用一列 <div>，颜色随激活强度变。
// 预测时各层按顺序"亮起来"。

const NetworkViz = (() => {
  const root = document.getElementById("network-viz");

  // 各层的布局：name, shape (用于显示), grid (展示几行几列), cellSize
  const LAYERS = [
    { name: "Input",  shape: "1@28×28",  type: "single-map", size: 56 },
    { name: "C1",     shape: "6@28×28",  type: "feature-maps", count: 6,  cols: 2, size: 40 },
    { name: "S2",     shape: "6@14×14",  type: "feature-maps", count: 6,  cols: 2, size: 28 },
    { name: "C3",     shape: "16@10×10", type: "feature-maps", count: 16, cols: 4, size: 24 },
    { name: "S4",     shape: "16@5×5",   type: "feature-maps", count: 16, cols: 4, size: 18 },
    { name: "C5",     shape: "120",      type: "fc", count: 120, cols: 4 },
    { name: "F6",     shape: "84",       type: "fc", count: 84,  cols: 3 },
    { name: "Output", shape: "10",       type: "output", count: 10 },
  ];

  const refs = {};   // 缓存每层的 DOM 节点

  function build() {
    root.innerHTML = "";

    LAYERS.forEach((layer, idx) => {
      // 连接线 (除了第一层前面)
      if (idx > 0) {
        const conn = document.createElement("div");
        conn.className = "connector";
        root.appendChild(conn);
      }

      const div = document.createElement("div");
      div.className = "layer";

      const title = document.createElement("div");
      title.className = "layer-title";
      title.innerHTML = `<strong>${layer.name}</strong>${layer.shape}`;
      div.appendChild(title);

      const body = document.createElement("div");
      body.className = "layer-body";

      if (layer.type === "single-map") {
        // Input 层：一个 28x28 的 canvas
        const c = document.createElement("canvas");
        c.width = 28; c.height = 28;
        c.style.width = layer.size + "px";
        c.style.height = layer.size + "px";
        c.className = "feature-map";
        body.appendChild(c);
        refs[layer.name] = { canvases: [c] };
      }
      else if (layer.type === "feature-maps") {
        // 卷积层：count 个特征图，按 cols 列排
        const grid = document.createElement("div");
        grid.className = "feature-grid";
        grid.style.gridTemplateColumns = `repeat(${layer.cols}, 1fr)`;
        const canvases = [];
        for (let i = 0; i < layer.count; i++) {
          const c = document.createElement("canvas");
          c.className = "feature-map";
          c.style.width = layer.size + "px";
          c.style.height = layer.size + "px";
          grid.appendChild(c);
          canvases.push(c);
        }
        body.appendChild(grid);
        refs[layer.name] = { canvases };
      }
      else if (layer.type === "fc") {
        // FC 层：count 个小方块
        const col = document.createElement("div");
        col.className = "fc-column";
        col.style.display = "grid";
        col.style.gridTemplateColumns = `repeat(${layer.cols}, 1fr)`;
        col.style.gap = "1px";
        const cells = [];
        for (let i = 0; i < layer.count; i++) {
          const cell = document.createElement("div");
          cell.className = "fc-cell";
          col.appendChild(cell);
          cells.push(cell);
        }
        body.appendChild(col);
        refs[layer.name] = { cells };
      }
      else if (layer.type === "output") {
        // 输出层：10 个圆点 + 数字标签
        const cont = document.createElement("div");
        cont.className = "output-units";
        const cells = [];
        for (let i = 0; i < 10; i++) {
          const row = document.createElement("div");
          row.className = "output-cell";
          const dot = document.createElement("div");
          dot.className = "output-dot";
          const label = document.createElement("div");
          label.className = "output-label";
          label.textContent = i;
          row.appendChild(dot);
          row.appendChild(label);
          cont.appendChild(row);
          cells.push({ row, dot, label });
        }
        body.appendChild(cont);
        refs[layer.name] = { cells };
      }

      div.appendChild(body);
      root.appendChild(div);
    });
  }

  // ──────────────────────────────────────────────────────────────────────
  // 把激活值画进各层
  // ──────────────────────────────────────────────────────────────────────

  // 把一个 H x W 的矩阵画进一个 canvas，按值映射到颜色
  function drawMap(canvas, mat) {
    const H = mat.length;
    const W = mat[0].length;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);

    // 归一化 (按这一张图的 min/max)
    let mn = Infinity, mx = -Infinity;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const v = mat[y][x];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    const span = mx - mn || 1;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = (mat[y][x] - mn) / span;
        // 用暖色映射: 暗紫 -> 黄
        // 简单灰度也行，但暖色更有"激活"的感觉
        const r = Math.round(v * 255);
        const g = Math.round(v * v * 200);
        const b = Math.round((1 - v) * 30);
        const i = (y * W + x) * 4;
        img.data[i + 0] = r;
        img.data[i + 1] = g;
        img.data[i + 2] = b;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // 把一个 H x W 的输入图画进 canvas (纯灰度，因为是原图)
  function drawInputMap(canvas, mat) {
    const H = mat.length;
    const W = mat[0].length;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);
    // 输入是 [-1, 1]，转 [0, 255]
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = Math.round(((mat[y][x] + 1) / 2) * 255);
        const i = (y * W + x) * 4;
        img.data[i + 0] = v;
        img.data[i + 1] = v;
        img.data[i + 2] = v;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // FC 层：用 viridis-like 颜色
  function activationColor(v) {
    // v ∈ [0, 1]: 暗 -> 黄
    const r = Math.round(v * 251);
    const g = Math.round(v * 191);
    const b = Math.round(v * 36 + (1 - v) * 25);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function renderConvLayer(layerName, maps) {
    // maps: 三维数组 (C, H, W)
    const ref = refs[layerName];
    if (!ref) return;
    for (let c = 0; c < maps.length && c < ref.canvases.length; c++) {
      drawMap(ref.canvases[c], maps[c]);
      // 根据平均激活强度加亮
      let sum = 0, n = 0;
      for (const row of maps[c])
        for (const v of row) { sum += Math.abs(v); n++; }
      const mean = sum / n;
      if (mean > 0.3) ref.canvases[c].classList.add("active");
      else            ref.canvases[c].classList.remove("active");
    }
  }

  function renderFCLayer(layerName, values) {
    const ref = refs[layerName];
    if (!ref) return;
    // 归一化
    let mx = 0;
    for (const v of values) if (Math.abs(v) > mx) mx = Math.abs(v);
    mx = mx || 1;
    for (let i = 0; i < values.length && i < ref.cells.length; i++) {
      const v = Math.abs(values[i]) / mx;
      ref.cells[i].style.background = activationColor(v);
      if (v > 0.7) ref.cells[i].style.boxShadow = `0 0 6px ${activationColor(v)}`;
      else         ref.cells[i].style.boxShadow = "none";
    }
  }

  function renderOutput(probs, prediction) {
    const ref = refs["Output"];
    if (!ref) return;
    for (let i = 0; i < 10; i++) {
      const v = probs[i];
      ref.cells[i].dot.style.background = activationColor(v);
      ref.cells[i].dot.style.boxShadow = v > 0.3
        ? `0 0 10px ${activationColor(v)}`
        : "none";
      if (i === prediction) ref.cells[i].row.classList.add("predicted");
      else                   ref.cells[i].row.classList.remove("predicted");
    }
  }

  function renderInput(image28) {
    // image28: 长度 784 的数组，[0, 1]
    const ref = refs["Input"];
    if (!ref) return;
    const mat = [];
    for (let y = 0; y < 28; y++) {
      const row = [];
      for (let x = 0; x < 28; x++) row.push(image28[y * 28 + x] * 2 - 1);
      mat.push(row);
    }
    drawInputMap(ref.canvases[0], mat);
  }

  function clearActivations() {
    LAYERS.forEach((layer) => {
      const r = refs[layer.name];
      if (!r) return;
      if (r.canvases) r.canvases.forEach((c) => {
        const ctx = c.getContext("2d");
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, c.width, c.height);
        c.classList.remove("active");
      });
      if (r.cells) r.cells.forEach((cell) => {
        if (cell.row) {
          cell.row.classList.remove("predicted");
          cell.dot.style.background = "#1a1d24";
          cell.dot.style.boxShadow = "none";
        } else {
          cell.style.background = "#1a1d24";
          cell.style.boxShadow = "none";
        }
      });
    });
  }

  // 按顺序"逐层点亮"。instant=true 时跳过所有动画 sleep，给实时预测用。
  async function animateForward(activations, prediction, image28, instant = false) {
    const wait = instant ? 0 : 80;

    // 先画输入
    renderInput(image28);
    if (wait) await sleep(wait);

    if (activations.C1) { renderConvLayer("C1", activations.C1); if (wait) await sleep(wait); }
    if (activations.S2) { renderConvLayer("S2", activations.S2); if (wait) await sleep(wait); }
    if (activations.C3) { renderConvLayer("C3", activations.C3); if (wait) await sleep(wait); }
    if (activations.S4) { renderConvLayer("S4", activations.S4); if (wait) await sleep(wait); }
    if (activations.C5) { renderFCLayer("C5", activations.C5);   if (wait) await sleep(wait); }
    if (activations.F6) { renderFCLayer("F6", activations.F6);   if (wait) await sleep(wait); }
    if (activations.probs) renderOutput(activations.probs, prediction);
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  return {
    build,
    clearActivations,
    animateForward,
  };
})();
