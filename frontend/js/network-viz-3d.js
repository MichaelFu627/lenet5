// 3D 体素可视化（Three.js）
//
// 每层的特征图沿 z 轴（深度方向）一片一片叠起来。
// 全连接层是一长串方块向后延伸。
// 鼠标可拖动旋转视角。
//
// 颜色：冷色 (深蓝 -> 青)，避免和下面 2D 视图的暖色重复。

const NetworkViz3D = (() => {
  const container = document.getElementById("network-viz-3d");
  if (!container || typeof THREE === "undefined") {
    console.warn("[viz3d] THREE.js not loaded");
    return { build: () => {}, render: () => {}, clear: () => {} };
  }

  // 网络层定义
  // pixel: 单个体素 (cube) 的边长
  // depth_spacing: 同一层内多张 feature map 之间的 z 间距
  // x: 该层在场景中的 x 坐标（沿水平轴排开，整个网络中心在 x=0）
  const LAYERS = [
    { name: "Input",  type: "maps",  count: 1,   w: 28, h: 28, pixel: 0.22, depth_spacing: 0,    x: -36 },
    { name: "C1",     type: "maps",  count: 6,   w: 28, h: 28, pixel: 0.22, depth_spacing: 1.0,  x: -22 },
    { name: "S2",     type: "maps",  count: 6,   w: 14, h: 14, pixel: 0.38, depth_spacing: 1.0,  x: -10 },
    { name: "C3",     type: "maps",  count: 16,  w: 10, h: 10, pixel: 0.45, depth_spacing: 0.7,  x: 2   },
    { name: "S4",     type: "maps",  count: 16,  w: 5,  h: 5,  pixel: 0.75, depth_spacing: 0.7,  x: 14  },
    { name: "C5",     type: "line",  count: 120, pixel: 0.18,                                    x: 22, axis: "y" },
    { name: "F6",     type: "line",  count: 84,  pixel: 0.24,                                    x: 30, axis: "y" },
    { name: "Output", type: "cubes", count: 10,  pixel: 0.85,                                    x: 40, axis: "y" },
  ];

  let scene, camera, renderer, controls;
  let layerMeshes = {};
  let layerGroups = {};
  let layerLabels = {};
  let raf = null;
  let needsRender = true;

  // 蓝青色 HSL 映射激活强度
  function colorForActivation(v) {
    const l = 0.05 + v * 0.65;
    const s = 0.55 + v * 0.35;
    return new THREE.Color().setHSL(195 / 360, s, l);
  }
  function colorForInput(v) {
    return new THREE.Color(v, v, v);
  }

  function build() {
    container.innerHTML = "";

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e13);
    scene.fog = new THREE.Fog(0x0c0e13, 55, 120);

    const w = container.clientWidth;
    const h = container.clientHeight || 380;
    camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 400);
    // 初始视角：从右上前方俯视，能看到整个网络
    camera.position.set(15, 22, 65);
    camera.lookAt(2, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(2, 0, 0);
    controls.minDistance = 20;
    controls.maxDistance = 180;
    controls.addEventListener("change", () => { needsRender = true; });

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));

    LAYERS.forEach((layer) => {
      const group = new THREE.Group();
      group.position.x = layer.x;
      scene.add(group);
      layerGroups[layer.name] = group;

      if (layer.type === "maps")  buildMapsLayer(layer, group);
      if (layer.type === "line")  buildLineLayer(layer, group);
      if (layer.type === "cubes") buildCubesLayer(layer, group);

      const lbl = document.createElement("div");
      lbl.className = "viz3d-label-3d";
      lbl.innerHTML = `<strong>${layer.name}</strong><span>${layerShape(layer)}</span>`;
      container.appendChild(lbl);
      layerLabels[layer.name] = lbl;
    });

    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight || 380;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      needsRender = true;
    });
    ro.observe(container);

    animate();
  }

  function layerShape(l) {
    if (l.type === "maps")  return `${l.count}@${l.w}×${l.h}`;
    if (l.type === "line")  return `${l.count}`;
    if (l.type === "cubes") return `${l.count}`;
    return "";
  }

  // 卷积/池化层：count 张 H×W 平面沿 z 排开
  function buildMapsLayer(layer, group) {
    const { count, w, h, pixel, depth_spacing } = layer;
    const meshes = [];
    const geo = new THREE.BoxGeometry(pixel * 0.85, pixel * 0.85, pixel * 0.85);
    const totalDepth = (count - 1) * depth_spacing;
    const z0 = -totalDepth / 2;

    for (let f = 0; f < count; f++) {
      const layerZ = z0 + f * depth_spacing;
      const featMeshes = [];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const px = (x - (w - 1) / 2) * pixel;
          const py = -(y - (h - 1) / 2) * pixel;
          const mat = new THREE.MeshBasicMaterial({ color: 0x0c0e13 });
          const mesh = new THREE.Mesh(geo, mat);
          mesh.position.set(px, py, layerZ);
          group.add(mesh);
          featMeshes.push(mesh);
        }
      }
      meshes.push(featMeshes);
    }
    layerMeshes[layer.name] = meshes;
  }

  // FC: 一长串小方块沿 z (默认) 或 y 轴排开
  function buildLineLayer(layer, group) {
    const { count, pixel, axis = "z" } = layer;
    const meshes = [];
    const geo = new THREE.BoxGeometry(pixel * 1.2, pixel * 1.2, pixel * 1.2);
    const spacing = pixel * 1.5;
    const totalLen = (count - 1) * spacing;
    const p0 = -totalLen / 2;

    for (let i = 0; i < count; i++) {
      const mat = new THREE.MeshBasicMaterial({ color: 0x0c0e13 });
      const mesh = new THREE.Mesh(geo, mat);
      if (axis === "y") {
        mesh.position.set(0, -p0 - i * spacing, 0);
      } else {
        mesh.position.set(0, 0, p0 + i * spacing);
      }
      group.add(mesh);
      meshes.push(mesh);
    }
    layerMeshes[layer.name] = meshes;
  }

  // Output: 10 个大方块沿 z 或 y 排
  function buildCubesLayer(layer, group) {
    const { count, pixel, axis = "z" } = layer;
    const meshes = [];
    const geo = new THREE.BoxGeometry(pixel, pixel, pixel);
    const spacing = pixel * 1.6;
    const totalLen = (count - 1) * spacing;
    const p0 = -totalLen / 2;

    for (let i = 0; i < count; i++) {
      const mat = new THREE.MeshBasicMaterial({ color: 0x0c0e13 });
      const mesh = new THREE.Mesh(geo, mat);
      if (axis === "y") {
        mesh.position.set(0, -p0 - i * spacing, 0);
      } else {
        mesh.position.set(0, 0, p0 + i * spacing);
      }
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0x2c313c })
      );
      mesh.add(edges);
      group.add(mesh);
      meshes.push(mesh);
    }
    layerMeshes[layer.name] = meshes;
  }

  function render(activations, prediction, image28) {
    // Input
    if (image28 && layerMeshes["Input"]) {
      const meshes = layerMeshes["Input"][0];
      for (let i = 0; i < 28 * 28; i++) {
        meshes[i].material.color.copy(colorForInput(image28[i]));
      }
    }

    // 卷积 / 池化层
    ["C1", "S2", "C3", "S4"].forEach((name) => {
      if (!activations[name] || !layerMeshes[name]) return;
      const maps = activations[name];
      let mn = Infinity, mx = -Infinity;
      for (const m of maps) for (const row of m) for (const v of row) {
        if (v < mn) mn = v; if (v > mx) mx = v;
      }
      const span = mx - mn || 1;
      for (let f = 0; f < maps.length && f < layerMeshes[name].length; f++) {
        const featMeshes = layerMeshes[name][f];
        const mat = maps[f];
        const H = mat.length, W = mat[0].length;
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const v = (mat[y][x] - mn) / span;
            featMeshes[y * W + x].material.color.copy(colorForActivation(v));
          }
        }
      }
    });

    // FC
    ["C5", "F6"].forEach((name) => {
      if (!activations[name] || !layerMeshes[name]) return;
      const vals = activations[name];
      let mx = 0;
      for (const v of vals) { const a = Math.abs(v); if (a > mx) mx = a; }
      mx = mx || 1;
      for (let i = 0; i < vals.length && i < layerMeshes[name].length; i++) {
        const v = Math.abs(vals[i]) / mx;
        layerMeshes[name][i].material.color.copy(colorForActivation(v));
      }
    });

    // Output
    if (activations.probs && layerMeshes["Output"]) {
      for (let i = 0; i < 10; i++) {
        const v = activations.probs[i];
        if (i === prediction) {
          layerMeshes["Output"][i].material.color.set(0x34d399);
        } else {
          layerMeshes["Output"][i].material.color.copy(colorForActivation(v));
        }
      }
    }

    needsRender = true;
  }

  function clear() {
    Object.values(layerMeshes).forEach((entry) => {
      if (Array.isArray(entry[0])) {
        entry.forEach((feat) => feat.forEach((m) => m.material.color.set(0x0c0e13)));
      } else {
        entry.forEach((m) => m.material.color.set(0x0c0e13));
      }
    });
    needsRender = true;
  }

  function animate() {
    raf = requestAnimationFrame(animate);
    controls.update();
    if (needsRender) {
      renderer.render(scene, camera);
      updateLabels();
      needsRender = false;
    }
  }

  // HTML label 投影到屏幕坐标，跟随相机
  function updateLabels() {
    const rect = container.getBoundingClientRect();
    Object.keys(layerLabels).forEach((name) => {
      const layer = LAYERS.find(l => l.name === name);
      let topY;
      if (layer.type === "maps") {
        topY = ((layer.h - 1) / 2) * layer.pixel + 1.0;
      } else if (layer.axis === "y") {
        const spacing = layer.pixel * (layer.type === "line" ? 1.5 : 1.6);
        const totalLen = (layer.count - 1) * spacing;
        topY = totalLen / 2 + 1.2;
      } else {
        topY = 1.5;
      }
      const v = new THREE.Vector3(layer.x, topY, 0);
      v.project(camera);
      const x = (v.x + 1) / 2 * rect.width;
      const y = (1 - (v.y + 1) / 2) * rect.height;
      const lbl = layerLabels[name];
      // clamp 到 stage 边界内（label 宽度约 50px）
      const clampedX = Math.max(30, Math.min(rect.width - 30, x));
      const clampedY = Math.max(10, Math.min(rect.height - 30, y));
      lbl.style.left = clampedX + "px";
      lbl.style.top = clampedY + "px";
      lbl.style.display = (v.z > 1 || v.z < -1) ? "none" : "block";
    });
  }

  return { build, render, clear };
})();
