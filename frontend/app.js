const API_BASE = "http://localhost:8000";
const ENDPOINT = "/predict/yolov11";
const HISTORY_KEY = "buncis_scan_history";
const MAX_HISTORY = 12;

const CLASS_LABELS = {
  embun_tepung: "Embun Tepung",
  hawar: "Hawar Daun",
  mozaik: "Mosaik",
  sehat: "Sehat",
  "0": "Kelas Tidak Valid",
};
const RISK_BY_CLASS = {
  embun_tepung: "sedang",
  hawar: "tinggi",
  mozaik: "tinggi",
  sehat: "rendah",
  "0": "rendah",
};

// Normalisasi label kelas dari API (kadang berspasi/berbeda kapital,
// mis. "embun tepung") supaya cocok dengan key di CLASS_LABELS/RISK_BY_CLASS.
function normalizeKey(label) {
  return String(label ?? "").toLowerCase().trim().replace(/\s+/g, "_");
}

// ---------------- Element refs ----------------
const $ = (id) => document.getElementById(id);

const viewfinder = $("viewfinder");
const fileInput = $("fileInput");
const viewfinderEmpty = $("viewfinderEmpty");
const imageStage = $("imageStage");
const previewImg = $("previewImg");
const bboxLayer = $("bboxLayer");
const scanLine = $("scanLine");
const cameraFeed = $("cameraFeed");

const pickBtn = $("pickBtn");
const cameraBtn = $("cameraBtn");
const analyzeBtn = $("analyzeBtn");

const readoutConf = $("readoutConf");
const readoutCount = $("readoutCount");
const readoutTime = $("readoutTime");

const resultEmpty = $("resultEmpty");
const resultContent = $("resultContent");
const gaugeFill = $("gaugeFill");
const gaugeText = $("gaugeText");
const riskBadge = $("riskBadge");
const resultLabel = $("resultLabel");
const probBars = $("probBars");
const causeText = $("causeText");
const checklist = $("checklist");
const checklistProgress = $("checklistProgress");
const detectionList = $("detectionList");
const downloadBtn = $("downloadBtn");

const historyStrip = $("historyStrip");
const historyEmpty = $("historyEmpty");
const clearHistoryBtn = $("clearHistoryBtn");

const cameraModal = $("cameraModal");
const modalVideo = $("modalVideo");
const captureCanvas = $("captureCanvas");
const closeCameraBtn = $("closeCameraBtn");
const captureBtn = $("captureBtn");

const toastContainer = $("toastContainer");

// ---------------- State ----------------
let currentFile = null;
let currentImgW = 0;
let currentImgH = 0;
let lastResultData = null;
let mediaStream = null;

// ==================================================================
// Toasts
// ==================================================================
function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = `toast${type === "error" ? " toast--error" : type === "success" ? " toast--success" : ""}`;
  el.textContent = message;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

// ==================================================================
// File intake: click / drag-drop / paste / camera
// ==================================================================
pickBtn.addEventListener("click", () => fileInput.click());
viewfinder.addEventListener("click", (e) => {
  if (!cameraFeed.hidden) return; // don't reopen picker while camera preview active
  fileInput.click();
});
viewfinder.addEventListener("keydown", (e) => { if (e.key === "Enter") fileInput.click(); });

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

["dragover", "dragenter"].forEach(evt =>
  viewfinder.addEventListener(evt, (e) => { e.preventDefault(); viewfinder.style.opacity = "0.85"; })
);
["dragleave", "drop"].forEach(evt =>
  viewfinder.addEventListener(evt, (e) => { viewfinder.style.opacity = "1"; })
);
viewfinder.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
});

document.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items || [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) { handleFile(file); toast("Gambar ditempel dari clipboard", "success"); }
    }
  }
});

function handleFile(file) {
  currentFile = file;
  stopCamera();
  cameraFeed.hidden = true;

  const url = URL.createObjectURL(file);
  previewImg.onload = () => {
    currentImgW = previewImg.naturalWidth;
    currentImgH = previewImg.naturalHeight;
  };
  previewImg.src = url;
  imageStage.hidden = false;
  viewfinderEmpty.hidden = true;
  bboxLayer.innerHTML = "";

  analyzeBtn.disabled = false;
  resetResultPanel();
}

// ---------------- Camera capture ----------------
cameraBtn.addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    modalVideo.srcObject = mediaStream;
    cameraModal.hidden = false;
  } catch (err) {
    toast("Tidak bisa mengakses kamera. Periksa izin browser.", "error");
  }
});

closeCameraBtn.addEventListener("click", () => {
  cameraModal.hidden = true;
  stopCamera();
});

captureBtn.addEventListener("click", () => {
  const w = modalVideo.videoWidth, h = modalVideo.videoHeight;
  captureCanvas.width = w;
  captureCanvas.height = h;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(modalVideo, 0, 0, w, h);
  captureCanvas.toBlob((blob) => {
    const file = new File([blob], `pindai-${Date.now()}.jpg`, { type: "image/jpeg" });
    handleFile(file);
    cameraModal.hidden = true;
    stopCamera();
    toast("Foto berhasil diambil", "success");
  }, "image/jpeg", 0.92);
});

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
}

// ==================================================================
// Analyze
// ==================================================================
analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Menganalisis...";
  scanLine.hidden = false;

  const startTime = performance.now();

  try {
    const formData = new FormData();
    formData.append("file", currentFile);
    const res = await fetch(`${API_BASE}${ENDPOINT}`, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Terjadi kesalahan pada server.");

    const elapsedMs = performance.now() - startTime;
    lastResultData = data;
    renderResult(data, elapsedMs);
    addHistoryEntry(data);
    toast("Analisis selesai", "success");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    scanLine.hidden = true;
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analisis Gambar";
  }
});

// ==================================================================
// Result rendering
// ==================================================================
function resetResultPanel() {
  resultEmpty.hidden = false;
  resultContent.hidden = true;
  readoutConf.textContent = "—";
  readoutCount.textContent = "—";
  readoutTime.textContent = "—";
}

function renderResult(data, elapsedMs, opts = {}) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;

  const info = data.info || {};
  const risk = info.tingkat_risiko || "rendah";
  const conf = data.confidence ?? 0;
  const pct = Math.round(conf * 100);

  // Readout strip
  readoutConf.textContent = `${pct}%`;
  readoutCount.textContent = String((data.deteksi || []).length);
  readoutTime.textContent = elapsedMs ? `${(elapsedMs / 1000).toFixed(2)}s` : "riwayat";

  // Gauge
  const circumference = 327;
  gaugeFill.style.strokeDashoffset = String(circumference - circumference * conf);
  gaugeFill.setAttribute("class", `gauge-fill risk-${risk}`);
  gaugeText.textContent = `${pct}%`;

  riskBadge.setAttribute("class", `risk-badge risk-${risk}`);
  riskBadge.textContent = risk === "tinggi" ? "Risiko Tinggi" : risk === "sedang" ? "Risiko Sedang" : "Risiko Rendah";
  resultLabel.textContent = info.nama || data.label_utama || "Tidak diketahui";

  // Cause / condition text
  causeText.textContent = info.penyebab && info.penyebab !== "-"
    ? `Penyebab: ${info.penyebab}. ${info.kondisi || ""}`
    : (info.kondisi || "");

  renderProbBars(data.probabilitas_kelas || {}, data.label_utama);
  renderChecklist(info.penanganan || []);
  renderDetections(data.deteksi || []);

  if (!opts.skipOverlay) {
    drawBboxOverlay(data.deteksi || []);
  }

  downloadBtn.onclick = () => downloadAnnotated(data.gambar_anotasi_base64);
}

function renderProbBars(probs, leadLabel) {
  probBars.innerHTML = "";
  const leadKey = normalizeKey(leadLabel);
  const entries = Object.entries(probs).length ? Object.entries(probs)
    : Object.keys(CLASS_LABELS).map(k => [k, 0]);

  entries
    .sort((a, b) => b[1] - a[1])
    .forEach(([key, value]) => {
      const pct = Math.round(value * 100);
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-name">${CLASS_LABELS[key] || key}</span>
        <span class="prob-track"><span class="prob-fill${key === leadKey ? " is-lead" : ""}" style="width:${pct}%"></span></span>
        <span class="prob-value">${pct}%</span>
      `;
      probBars.appendChild(row);
    });
}

function renderChecklist(steps) {
  checklist.innerHTML = "";
  checklistProgress.style.width = "0%";

  if (!steps.length) {
    checklist.innerHTML = `<li><span>Tidak ada langkah penanganan khusus.</span></li>`;
    return;
  }

  steps.forEach((step, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<input type="checkbox" id="step-${i}"><span>${step}</span>`;
    const checkbox = li.querySelector("input");
    checkbox.addEventListener("change", () => {
      li.classList.toggle("is-done", checkbox.checked);
      updateChecklistProgress();
    });
    checklist.appendChild(li);
  });
}

function updateChecklistProgress() {
  const boxes = checklist.querySelectorAll("input[type=checkbox]");
  if (!boxes.length) return;
  const done = [...boxes].filter(b => b.checked).length;
  checklistProgress.style.width = `${(done / boxes.length) * 100}%`;
}

function renderDetections(detections) {
  detectionList.innerHTML = "";
  if (!detections.length) {
    detectionList.innerHTML = `<li class="detection-empty">Tidak ada objek/lesi terdeteksi &mdash; daun tampak sehat.</li>`;
    return;
  }
  detections.forEach((d, i) => {
    const li = document.createElement("li");
    li.className = "detection-item";
    li.dataset.index = String(i);
    const key = normalizeKey(d.label);
    li.innerHTML = `
      <span>${CLASS_LABELS[key] || d.label}</span>
      <span class="conf">${Math.round(d.confidence * 100)}%</span>
    `;
    li.addEventListener("mouseenter", () => highlightBbox(i));
    li.addEventListener("mouseleave", () => highlightBbox(null));
    detectionList.appendChild(li);
  });
}

function drawBboxOverlay(detections) {
  bboxLayer.innerHTML = "";
  if (!currentImgW || !currentImgH || !detections.length) return;

  detections.forEach((d, i) => {
    const { x1, y1, x2, y2 } = d.bbox;
    const key = normalizeKey(d.label);
    const risk = RISK_BY_CLASS[key] || "rendah";
    const box = document.createElement("div");
    box.className = `bbox-box risk-${risk}`;
    box.dataset.index = String(i);
    box.style.left = `${(x1 / currentImgW) * 100}%`;
    box.style.top = `${(y1 / currentImgH) * 100}%`;
    box.style.width = `${((x2 - x1) / currentImgW) * 100}%`;
    box.style.height = `${((y2 - y1) / currentImgH) * 100}%`;

    const tag = document.createElement("span");
    tag.className = "bbox-tag";
    tag.textContent = `${CLASS_LABELS[key] || d.label} ${Math.round(d.confidence * 100)}%`;
    box.appendChild(tag);

    bboxLayer.appendChild(box);
  });
}

function highlightBbox(index) {
  const boxes = bboxLayer.querySelectorAll(".bbox-box");
  const items = detectionList.querySelectorAll(".detection-item");
  boxes.forEach(b => b.classList.toggle("is-highlighted", index !== null && Number(b.dataset.index) === index));
  items.forEach(it => it.classList.toggle("is-active", index !== null && Number(it.dataset.index) === index));
}

function downloadAnnotated(base64) {
  if (!base64) { toast("Gambar beranotasi tidak tersedia.", "error"); return; }
  const a = document.createElement("a");
  a.href = `data:image/jpeg;base64,${base64}`;
  a.download = `hasil-deteksi-${Date.now()}.jpg`;
  a.click();
}

// ==================================================================
// Tabs
// ==================================================================
const tabBtns = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");
const tabUnderline = document.querySelector(".tab-underline");

function positionUnderline(btn) {
  tabUnderline.style.width = `${btn.offsetWidth}px`;
  tabUnderline.style.transform = `translateX(${btn.offsetLeft}px)`;
}

tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.toggle("active", b === btn));
    tabPanels.forEach(p => p.classList.toggle("active", p.dataset.panel === btn.dataset.tab));
    positionUnderline(btn);
  });
});
window.addEventListener("resize", () => {
  const active = document.querySelector(".tab-btn.active");
  if (active) positionUnderline(active);
});
if (tabBtns[0]) requestAnimationFrame(() => positionUnderline(tabBtns[0]));

// ==================================================================
// History (persisted in localStorage on this device)
// ==================================================================
function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory(list) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, MAX_HISTORY)));
}

function addHistoryEntry(data) {
  const thumb = shrinkThumbnail(previewImg);
  thumb.then((thumbDataUrl) => {
    const list = loadHistory();
    list.unshift({
      thumb: thumbDataUrl,
      label: (data.info && data.info.nama) || data.label_utama,
      confidence: data.confidence,
      risk: (data.info && data.info.tingkat_risiko) || "rendah",
      time: new Date().toLocaleString("id-ID", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }),
      full: data,
    });
    saveHistory(list);
    renderHistory();
  });
}

function shrinkThumbnail(imgEl) {
  return new Promise((resolve) => {
    try {
      const canvas = document.createElement("canvas");
      const size = 120;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      const ratio = Math.max(size / imgEl.naturalWidth, size / imgEl.naturalHeight);
      const w = imgEl.naturalWidth * ratio, h = imgEl.naturalHeight * ratio;
      ctx.drawImage(imgEl, (size - w) / 2, (size - h) / 2, w, h);
      resolve(canvas.toDataURL("image/jpeg", 0.7));
    } catch {
      resolve("");
    }
  });
}

function renderHistory() {
  const list = loadHistory();
  historyStrip.innerHTML = "";
  if (!list.length) {
    historyStrip.appendChild(historyEmpty);
    return;
  }
  list.forEach((entry) => {
    const chip = document.createElement("div");
    chip.className = "history-chip";
    chip.innerHTML = `
      <img src="${entry.thumb}" alt="${entry.label}">
      <div class="history-chip-label">${entry.label} · ${Math.round((entry.confidence || 0) * 100)}%</div>
    `;
    chip.title = entry.time;
    chip.addEventListener("click", () => {
      lastResultData = entry.full;
      if (entry.thumb) {
        previewImg.src = entry.thumb;
        imageStage.hidden = false;
        viewfinderEmpty.hidden = true;
        bboxLayer.innerHTML = "";
      }
      renderResult(entry.full, 0, { skipOverlay: true });
      toast(`Menampilkan hasil: ${entry.time}`, "info");
    });
    historyStrip.appendChild(chip);
  });
}

clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  toast("Riwayat dihapus", "success");
});

// ---------------- Nav UX polish ----------------
document.querySelectorAll('.navbar-links a, .hero-cta a').forEach(link => {
  link.addEventListener("click", (e) => {
    const id = link.getAttribute("href");
    if (id && id.startsWith("#")) {
      const target = document.querySelector(id);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  });
});

const navbarEl = document.getElementById("navbar");
if (navbarEl) {
  window.addEventListener("scroll", () => {
    navbarEl.classList.toggle("is-scrolled", window.scrollY > 8);
  }, { passive: true });
}

// ---------------- Init ----------------
resetResultPanel();
renderHistory();
