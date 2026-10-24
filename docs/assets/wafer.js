// The gallery as a silicon wafer: one die per example figure, textured with its render.
// Drag to turn, scroll to zoom, point at a die to lift it, click to open the example.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const R = 5;                 // wafer radius (scene units)
const FLAT = -0.955 * R;     // primary flat, on the side nearest the camera
const THICK = 0.14;
const LIFT = 0.42;

// Square cells inside the wafer, row by row from the far edge, until there are at least n.
function layout(n) {
  const margin = 0.28;
  for (let c = 1.7; c > 0.3; c -= 0.01) {
    const cells = [];
    const rows = Math.floor((2 * (R - margin)) / c);
    const z0 = -(rows * c) / 2;
    for (let r = 0; r < rows; r++) {
      const za = z0 + r * c, zb = za + c;
      if (zb > -FLAT - margin) continue;           // the flat is at +z in scene space
      const zz = Math.max(Math.abs(za), Math.abs(zb));
      const half = Math.sqrt(Math.max(0, (R - margin) ** 2 - zz * zz));
      const k = Math.floor((2 * half) / c);
      for (let i = 0; i < k; i++) cells.push({ x: (i - (k - 1) / 2) * c, z: za + c / 2 });
    }
    if (cells.length >= n) return { c, cells };
  }
  throw new Error('wafer layout failed');
}

function waferShape() {
  const s = new THREE.Shape();
  const a = Math.acos(FLAT / R);                    // angle where the flat meets the rim
  s.absarc(0, 0, R, -Math.PI / 2 + (Math.PI - a), -Math.PI / 2 - (Math.PI - a) + 2 * Math.PI, false);
  s.closePath();
  return s;
}

export function createWafer(host, { examples, style, onHover, onPick }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x000000, 18, 34);   // setTheme gives the fog and the clear colour the stage colour
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 80);
  const home = new THREE.Vector3(0, 10.2, 9.4);
  camera.position.copy(home);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 5.5;
  controls.maxDistance = 22;
  controls.maxPolarAngle = Math.PI * 0.46;
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  controls.autoRotate = !still;
  controls.autoRotateSpeed = 0.28;
  controls.target.set(0, 0, 0.3);

  scene.add(new THREE.HemisphereLight(0xd6deff, 0x0b0d12, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(-5, 9, 4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x7ea6ff, 0.7);
  rim.position.set(6, 3, -6);
  scene.add(rim);

  // wafer body: polished silicon, dark blue-grey, with the primary flat toward the viewer
  const body = new THREE.Mesh(
    new THREE.ExtrudeGeometry(waferShape(), { depth: THICK, bevelEnabled: true, bevelThickness: 0.03, bevelSize: 0.05,
      bevelSegments: 3, curveSegments: 160 }),
    new THREE.MeshStandardMaterial({ color: 0x252d3a, metalness: 0.75, roughness: 0.32 }));
  body.rotation.x = -Math.PI / 2;
  body.position.y = -THICK;
  scene.add(body);

  // a faint scribe grid between dies
  const { c, cells } = layout(examples.length);
  const die = c * 0.9;
  const grid = new THREE.Group();
  const lineMat = new THREE.LineBasicMaterial({ color: 0x3a4658, transparent: true, opacity: 0.55 });
  const zs = [...new Set(cells.map(p => p.z))];
  for (const z of zs) {
    const row = cells.filter(p => p.z === z);
    const x0 = Math.min(...row.map(p => p.x)) - c / 2, x1 = Math.max(...row.map(p => p.x)) + c / 2;
    for (const zz of [z - c / 2, z + c / 2]) {
      grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0, 0.002, zz), new THREE.Vector3(x1, 0.002, zz)]), lineMat));
    }
    for (const p of row) {
      for (const xx of [p.x - c / 2, p.x + c / 2]) {
        grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(xx, 0.002, z - c / 2), new THREE.Vector3(xx, 0.002, z + c / 2)]), lineMat));
      }
    }
  }
  scene.add(grid);

  // dies: a thin block with a white top (paper) and the figure fitted on it
  const loader = new THREE.TextureLoader();
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const cache = { field: new Map(), folio: new Map() };
  const blockGeo = new THREE.BoxGeometry(die, 0.05, die);
  const blockMat = new THREE.MeshStandardMaterial({ color: 0x8994a6, metalness: 0.4, roughness: 0.5 });
  const plane = new THREE.PlaneGeometry(1, 1);
  const dies = [];
  examples.forEach((ex, i) => {
    const g = new THREE.Group();
    g.position.set(cells[i].x, 0, cells[i].z);
    const block = new THREE.Mesh(blockGeo, blockMat.clone());
    block.position.y = 0.025;
    const paper = new THREE.Mesh(plane, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true }));
    paper.rotation.x = -Math.PI / 2;
    paper.position.y = 0.0515;
    paper.scale.set(die * 0.96, die * 0.96, 1);
    const fig = new THREE.Mesh(plane, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, toneMapped: false }));
    fig.rotation.x = -Math.PI / 2;
    fig.position.y = 0.053;
    g.add(block, paper, fig);
    g.userData = { ex, y: 0, target: 0, s: 1, fade: 1, fadeTarget: 1, visible: true };
    paper.userData.die = g;
    block.userData.die = g;
    fig.userData.die = g;
    scene.add(g);
    dies.push({ g, fig, paper, block });
  });
  // leftover edge cells hold blank dies, as a real wafer does
  for (let i = examples.length; i < cells.length; i++) {
    const m = new THREE.Mesh(blockGeo, new THREE.MeshStandardMaterial({ color: 0x3b4556, metalness: 0.6, roughness: 0.4 }));
    m.position.set(cells[i].x, 0.025, cells[i].z);
    scene.add(m);
  }

  function fit(d, img) {
    const inner = die * 0.9, a = img.w / img.h;
    const w = a >= 1 ? inner : inner * a, h = a >= 1 ? inner / a : inner;
    d.fig.scale.set(w, h, 1);
  }

  function setStyle(st) {
    dies.forEach(d => {
      const ex = d.g.userData.ex, img = ex.images[st];
      fit(d, img);
      const hit = cache[st].get(ex.id);
      if (hit) { d.fig.material.map = hit; d.fig.material.needsUpdate = true; return; }
      loader.load(img.thumb, tex => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = maxAniso;
        cache[st].set(ex.id, tex);
        if (current === st) { d.fig.material.map = tex; d.fig.material.needsUpdate = true; }
      });
    });
    current = st;
  }
  let current = style;
  setStyle(style);

  // pointer: hover lifts a die, a click without a drag opens it
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let hovered = null, highlighted = null, down = null;
  const el = renderer.domElement;
  function pick(e) {
    const r = el.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const targets = dies.filter(d => d.g.userData.visible).flatMap(d => [d.paper, d.block]);
    const hit = ray.intersectObjects(targets, false)[0];
    return hit ? hit.object.userData.die : null;
  }
  function setHover(g, e) {
    if (g !== hovered) {
      hovered = g;
      el.classList.toggle('pointing', !!g);
      onHover && onHover(g ? g.userData.ex : null, e);
    } else if (g && onHover) onHover(g.userData.ex, e);
  }
  el.addEventListener('pointermove', e => { if (e.pointerType === 'mouse') setHover(pick(e), e); });
  el.addEventListener('pointerleave', () => setHover(null));
  el.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; controls.autoRotate = false; });
  el.addEventListener('pointerup', e => {
    if (!down) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    down = null;
    controls.autoRotate = spin;
    if (moved > 6) return;
    const g = pick(e);
    if (!g) return;
    if (e.pointerType !== 'mouse' && hovered !== g) { setHover(g, e); return; }   // touch: first tap lifts, second opens
    onPick && onPick(g.userData.ex);
  });

  let spin = !still;
  function setSpin(on) { spin = on; controls.autoRotate = on; }
  function setFilter(ids) {
    dies.forEach(d => {
      const u = d.g.userData;
      u.visible = ids.has(u.ex.id);
      u.fadeTarget = u.visible ? 1 : 0.12;
    });
    if (hovered && !hovered.userData.visible) setHover(null);
  }
  function highlight(id) { highlighted = id ? dies.find(d => d.g.userData.ex.id === id)?.g || null : null; }
  function resetView() {
    camera.position.copy(home);
    controls.target.set(0, 0, 0.3);
    controls.update();
  }

  // render loop runs only while the stage is on screen
  let raf = 0, running = false, last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const k = 1 - Math.exp(-dt * 12);
    for (const d of dies) {
      const u = d.g.userData;
      u.target = (d.g === hovered || d.g === highlighted) && u.visible ? LIFT : 0;
      u.y += (u.target - u.y) * k;
      u.fade += (u.fadeTarget - u.fade) * k;
      d.g.position.y = u.y;
      const s = 1 + 0.08 * (u.y / LIFT);
      d.g.scale.set(s, 1, s);
      d.fig.material.opacity = u.fade;
      d.paper.material.opacity = u.fade;
      d.block.material.opacity = u.fade;
      d.block.material.transparent = u.fade < 0.99;
    }
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  function start() { if (!running) { running = true; last = performance.now(); raf = requestAnimationFrame(frame); } }
  function stop() { running = false; cancelAnimationFrame(raf); }
  // the background follows the page theme: site.css defines --stage for light and dark
  function setTheme() {
    const bg = new THREE.Color(getComputedStyle(host).getPropertyValue('--stage').trim() || '#000');
    renderer.setClearColor(bg, 1);
    scene.fog.color.copy(bg);
    if (!running) renderer.render(scene, camera);
  }
  setTheme();

  const ro = new ResizeObserver(() => {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // keep the whole wafer in view on narrow stages: widen the vertical field until the width fits
    const half = Math.atan((R + 0.9) / camera.position.distanceTo(controls.target));
    camera.fov = Math.max(34, (2 * Math.atan(Math.tan(half) / (w / h)) * 180) / Math.PI);
    camera.updateProjectionMatrix();
  });
  ro.observe(host);
  let onScreen = false;
  const sync = () => (onScreen && !document.hidden ? start() : stop());
  new IntersectionObserver(([en]) => { onScreen = en.isIntersecting; sync(); }).observe(host);
  document.addEventListener('visibilitychange', sync);

  return { setStyle, setFilter, setSpin, setTheme, highlight, resetView, start, stop };
}
