// API 通信
// 后端默认 http://localhost:5000 (Flask 直接 serve 前端时是同源)

const API = (() => {
  // 同源部署时为空字符串，开发模式可以填 'http://localhost:5000'
  const BASE = "";

  async function health() {
    const r = await fetch(`${BASE}/health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  async function predict(image28) {
    // image28: 长度 784 的 Float32Array 或数组，值在 [0, 1]
    const r = await fetch(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: Array.from(image28) }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  return { health, predict };
})();
