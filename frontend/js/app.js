// 主控制流：协调画布、API、可视化

(function () {
  const statusEl = document.getElementById("status");
  const predBig = document.getElementById("prediction-big");
  const probBars = document.getElementById("prob-bars");
  const clearBtn = document.getElementById("clear-btn");
  const predictBtn = document.getElementById("predict-btn");
  const autoToggle = document.getElementById("auto-predict");

  // ──────────────────────────────────────────────────────────────────────
  // 初始化网络可视化骨架（2D 平视 + 3D 立体）
  // ──────────────────────────────────────────────────────────────────────
  NetworkViz.build();
  NetworkViz3D.build();

  // 初始化概率柱条 (0-9)
  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.dataset.digit = i;
    row.innerHTML = `
      <div class="prob-label">${i}</div>
      <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: 0%"></div></div>
      <div class="prob-value">0.0%</div>
    `;
    probBars.appendChild(row);
  }

  // ──────────────────────────────────────────────────────────────────────
  // 后端健康检查
  // ──────────────────────────────────────────────────────────────────────
  (async () => {
    try {
      const h = await API.health();
      if (h.weights_loaded) {
        statusEl.textContent = "✓ 后端就绪，模型已加载";
        statusEl.className = "ready";
      } else {
        statusEl.textContent = "⚠ 后端连上了，但模型未训练 (请先运行 python -m backend.train)";
        statusEl.className = "warning";
      }
    } catch (e) {
      statusEl.textContent = "✗ 无法连接后端 (默认 :5000)";
      statusEl.className = "error";
    }
  })();

  // ──────────────────────────────────────────────────────────────────────
  // 预测
  // ──────────────────────────────────────────────────────────────────────
  let inFlight = false;
  let pending = false;
  let firstPrediction = true;   // 清空后第一次预测用动画模式，之后全实时

  async function predict({ instant = false } = {}) {
    if (inFlight) { pending = true; return; }
    inFlight = true;
    try {
      const image = DrawingPad.getMnistInput();
      // 如果完全空白，重置 UI
      const sum = image.reduce((a, b) => a + b, 0);
      if (sum < 0.5) {
        resetPredictionUI();
        firstPrediction = true;
        return;
      }
      const res = await API.predict(image);
      // 第一次给个动画，后续全 instant，避免动画累积延迟
      const useInstant = instant || !firstPrediction;
      firstPrediction = false;
      updatePredictionUI(res, image, useInstant);
    } catch (e) {
      console.error(e);
      statusEl.textContent = `预测出错: ${e.message}`;
      statusEl.className = "error";
    } finally {
      inFlight = false;
      // 有积压的请求，立刻再跑一次
      if (pending) { pending = false; predict({ instant: true }); }
    }
  }

  function updatePredictionUI(res, image28, instant = false) {
    predBig.textContent = res.prediction;
    predBig.classList.add("has-result");

    // 概率柱
    const probs = res.probs;
    const topIdx = res.prediction;
    Array.from(probBars.children).forEach((row, i) => {
      const fill = row.querySelector(".prob-bar-fill");
      const val = row.querySelector(".prob-value");
      fill.style.width = (probs[i] * 100) + "%";
      val.textContent = (probs[i] * 100).toFixed(1) + "%";
      if (i === topIdx) row.classList.add("top");
      else              row.classList.remove("top");
    });

    NetworkViz.animateForward(res.activations, res.prediction, image28, instant);
    NetworkViz3D.render(res.activations, res.prediction, image28);
  }

  function resetPredictionUI() {
    predBig.textContent = "?";
    predBig.classList.remove("has-result");
    Array.from(probBars.children).forEach((row) => {
      row.querySelector(".prob-bar-fill").style.width = "0%";
      row.querySelector(".prob-value").textContent = "0.0%";
      row.classList.remove("top");
    });
    NetworkViz.clearActivations();
    NetworkViz3D.clear();
  }

  // ──────────────────────────────────────────────────────────────────────
  // 事件
  // ──────────────────────────────────────────────────────────────────────
  clearBtn.addEventListener("click", () => {
    DrawingPad.clear();
    resetPredictionUI();
    firstPrediction = true;
  });
  predictBtn.addEventListener("click", () => predict({ instant: false }));

  // 实时预测：笔尖移动时 throttle 触发，每 ~80ms 一次
  // throttle vs debounce: debounce 是 "等你停下再做"，throttle 是 "固定节奏做"
  // 实时预测需要后者
  const THROTTLE_MS = 80;
  let lastFired = 0;
  let pendingTimer = null;

  function throttledPredict() {
    if (!autoToggle.checked) return;
    const now = Date.now();
    const elapsed = now - lastFired;
    if (elapsed >= THROTTLE_MS) {
      lastFired = now;
      predict({ instant: true });
    } else if (!pendingTimer) {
      // 距离上次发射不够久，安排一次延迟发射（确保最后一笔也能预测）
      pendingTimer = setTimeout(() => {
        pendingTimer = null;
        lastFired = Date.now();
        predict({ instant: true });
      }, THROTTLE_MS - elapsed);
    }
  }

  DrawingPad.setOnStrokeMove(throttledPredict);
  // 抬笔时再预测一次，保证最终笔画状态被预测过
  DrawingPad.setOnStrokeEnd(throttledPredict);
})();
