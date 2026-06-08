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
    // 卷积层：每张 feature map 是 y-z 平面上的一张"卡片"，多张卡片沿 x 轴(网络流方向)排开
    // x 是该层的中心位置，slice_spacing 是同层内多张 feature map 沿 x 的间距
    { name: "Input",  type: "maps",  count: 1,   w: 28, h: 28, pixel: 0.22, slice_spacing: 0,    x: -30 },
    { name: "C1",     type: "maps",  count: 6,   w: 28, h: 28, pixel: 0.22, slice_spacing: 1.2,  x: -18 },
    { name: "S2",     type: "maps",  count: 6,   w: 14, h: 14, pixel: 0.38, slice_spacing: 1.2,  x: -6  },
    { name: "C3",     type: "maps",  count: 16,  w: 10, h: 10, pixel: 0.45, slice_spacing: 0.6,  x:  8  },
    { name: "S4",     type: "maps",  count: 16,  w: 5,  h: 5,  pixel: 0.75, slice_spacing: 0.6,  x:  22 },
    { name: "C5",     type: "line",  count: 120, pixel: 0.10,                                    x:  34, axis: "y" },
    { name: "F6",     type: "line",  count: 84,  pixel: 0.15,                                    x:  44, axis: "y" },
    { name: "Output", type: "cubes", count: 10,  pixel: 0.70,                                    x:  54, axis: "y" },
  ];

  let scene, camera, renderer, controls;
  let layerMeshes = {};
  let layerGroups = {};
  let layerLabels = {};
  let fcConnections = {};
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
    fcConnections = {};

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e13);
    scene.fog = new THREE.Fog(0x0c0e13, 55, 120);

    const w = container.clientWidth;
    const h = container.clientHeight || 380;
    camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 400);
    // 初始视角：从右上前方俯视，能看到整个网络
    camera.position.set(30, 20, 50);
    camera.lookAt(12, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(12, 0, 0);
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

    // FC 全连接层之间画连线
    addFCConnections("C5", "F6");
    addFCConnections("F6", "Output");

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

  // 卷积/池化层：每张 feature map 在 yz 平面上（"面向右侧"），多张沿 x 轴堆叠
  function buildMapsLayer(layer, group) {
    const { count, w, h, pixel, slice_spacing } = layer;
    const meshes = [];
    const geo = new THREE.BoxGeometry(pixel * 0.85, pixel * 0.85, pixel * 0.85);
    // count 张 map 沿 x 排开，整体居中在 group.x
    const totalSpan = (count - 1) * slice_spacing;
    const x0 = -totalSpan / 2;

    for (let f = 0; f < count; f++) {
      const layerX = x0 + f * slice_spacing;
      const featMeshes = [];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          // feature map 的 x 维 → 世界 z 轴；y 维 → 世界 y 轴
          // 所有像素共享世界 x 坐标 → 整张 map 是 yz 平面上的一张"片"
          const py = -(y - (h - 1) / 2) * pixel;
          const pz = -(x - (w - 1) / 2) * pixel;
          const mat = new THREE.MeshBasicMaterial({ color: 0x0c0e13 });
          const mesh = new THREE.Mesh(geo, mat);
          mesh.position.set(layerX, py, pz);
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

  // FC 全连接：两层之间每对神经元画一条线
  // 模块顶部需要新增一个 let（跟 layerMeshes 那些放在一起）
  // let fcConnections = {};

  function addFCConnections(fromName, toName) {
    const fromMeshes = layerMeshes[fromName];
    const toMeshes = layerMeshes[toName];
    if (!fromMeshes || !toMeshes) return;

    const positions = [];
    for (const fm of fromMeshes) {
      const fp = new THREE.Vector3(); fm.getWorldPosition(fp);
      for (const tm of toMeshes) {
        const tp = new THREE.Vector3(); tm.getWorldPosition(tp);
        positions.push(fp.x, fp.y, fp.z, tp.x, tp.y, tp.z);
      }
    }
    // 顶点颜色初始全 0 (即不可见)；render() 时按激活值填充
    const colors = new Float32Array(positions.length);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.Float32BufferAttribute(colors, 3));
    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const lines = new THREE.LineSegments(geo, mat);
    scene.add(lines);
    fcConnections[`${fromName}->${toName}`] = {
      lines, fromName, toName,
      fromCount: fromMeshes.length, toCount: toMeshes.length,
    };
  }

  // 更新连线颜色：每条线两端按各自神经元的激活强度上色
  // 渐变效果让你能看到信号"流动"
  function updateFCLines(activations) {
    Object.values(fcConnections).forEach((conn) => {
      const { lines, fromName, toName, fromCount, toCount } = conn;
      // 从该层名取激活向量；Output 用 probs
      const fromActs = activations[fromName];
      const toActs = (toName === "Output") ? activations.probs : activations[toName];
      if (!fromActs || !toActs) return;

      // 归一化激活到 [0, 1]
      let fMax = 1e-9, tMax = 1e-9;
      for (const v of fromActs) { const a = Math.abs(v); if (a > fMax) fMax = a; }
      for (const v of toActs)   { const a = Math.abs(v); if (a > tMax) tMax = a; }

      const colors = lines.geometry.attributes.color.array;
      let idx = 0;
      // 蓝青色基色 #38bdf8 = (0.22, 0.74, 0.97)
      for (let i = 0; i < fromCount; i++) {
        const fs = Math.abs(fromActs[i] || 0) / fMax;
        // 强度做幂次拉伸：让弱的更弱，强的更显眼
        const fStr = Math.pow(fs, 1.5);
        for (let j = 0; j < toCount; j++) {
          const ts = Math.abs(toActs[j] || 0) / tMax;
          const tStr = Math.pow(ts, 1.5);
          // 起点端颜色按 from 激活
          colors[idx++] = fStr * 0.22;
          colors[idx++] = fStr * 0.74;
          colors[idx++] = fStr * 0.97;
          // 终点端颜色按 to 激活 → 形成沿连线的渐变
          colors[idx++] = tStr * 0.22;
          colors[idx++] = tStr * 0.74;
          colors[idx++] = tStr * 0.97;
        }
      }
      lines.geometry.attributes.color.needsUpdate = true;
    });
    needsRender = true;
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

    updateFCLines(activations);

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
    Object.values(fcConnections).forEach(({ lines }) => {
      lines.geometry.attributes.color.array.fill(0);
      lines.geometry.attributes.color.needsUpdate = true;
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
