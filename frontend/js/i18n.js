// 极简 i18n：data-i18n 属性 + localStorage 记忆 + 浏览器语言探测
//
// 用法：
//   HTML 元素加 data-i18n="key"；
//   JS 里用 I18N.t(key[, vars]) 取当前语言文案（支持 {key} 插值）；
//   I18N.apply(lang) 切换语言并重刷所有 data-i18n 元素。
(function () {
  const dict = {
    en: {
      subtitle: "Draw a digit and watch every layer of the network light up",
      "section.input": "1. Handwritten Input",
      "section.flat": "2. Network — Flat View (Detail)",
      "section.prediction": "3. Prediction",
      "section.3d": "4. Network — 3D View",
      "hint.flat": "Bright = strong activation; dim = weak. Each conv layer is a set of feature maps.",
      "hint.3d": "Feature maps of each conv layer spread horizontally; FC layers draw full connections. Drag to rotate, scroll to zoom.",
      "btn.clear": "Clear",
      "btn.predict": "Predict",
      "btn.realtime": "Real-time",
      "input.sees": "Model sees:",
      "status.connecting": "Connecting to backend...",
      "status.ready": "✓ Backend ready, model loaded",
      "status.untrained": "⚠ Backend connected, but model untrained (run `python -m backend.train` first)",
      "status.unreachable": "✗ Cannot reach backend (default :5000)",
      "status.predict_error": "Prediction error: {msg}",
    },
    zh: {
      subtitle: "在白板写一个数字，看每一层的神经元如何被激活",
      "section.input": "1. 手写输入",
      "section.flat": "2. 网络平视图（细节）",
      "section.prediction": "3. 预测结果",
      "section.3d": "4. 网络立体视图",
      "hint.flat": "明亮 = 激活强；灰暗 = 激活弱。每个卷积层是一组特征图。",
      "hint.3d": "每个卷积层的多张特征图沿水平方向排开；FC 层之间画全连接线。鼠标拖动旋转，滚轮缩放。",
      "btn.clear": "清空",
      "btn.predict": "预测",
      "btn.realtime": "实时预测",
      "input.sees": "模型看到：",
      "status.connecting": "连接后端中...",
      "status.ready": "✓ 后端就绪，模型已加载",
      "status.untrained": "⚠ 后端连上了，但模型未训练 (请先运行 python -m backend.train)",
      "status.unreachable": "✗ 无法连接后端 (默认 :5000)",
      "status.predict_error": "预测出错: {msg}",
    },
  };

  const STORAGE_KEY = "lenet5.lang";
  let current = "en";

  // 加载优先级：localStorage > 浏览器语言 > en
  function detect() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && dict[saved]) return saved;
    } catch (e) { /* localStorage 不可用时忽略 */ }
    const nav = (navigator.language || "en").toLowerCase();
    if (nav.startsWith("zh")) return "zh";
    return "en";
  }

  // 取当前语言下的文案，支持 {key} 插值
  function t(key, vars) {
    let s = (dict[current] && dict[current][key]) || key;
    if (vars) {
      for (const k in vars) {
        s = s.replace(new RegExp("\\{" + k + "\\}", "g"), String(vars[k]));
      }
    }
    return s;
  }

  function apply(lang) {
    if (!dict[lang]) lang = "en";
    current = lang;
    try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.documentElement.setAttribute("data-lang", lang);

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = t(el.getAttribute("data-i18n"));
    });

    document.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-lang") === lang);
    });
  }

  window.I18N = { t, apply };

  // 页面加载时初始化
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => apply(detect()));
  } else {
    apply(detect());
  }

  // 切换按钮（事件委托，不依赖 DOM 顺序）
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".lang-btn");
    if (btn) apply(btn.getAttribute("data-lang"));
  });
})();
