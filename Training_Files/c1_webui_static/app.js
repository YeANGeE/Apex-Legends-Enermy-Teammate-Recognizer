const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const inputPreview = document.querySelector("#inputPreview");
const resultImage = document.querySelector("#resultImage");
const resultEmpty = document.querySelector("#resultEmpty");
const detectButton = document.querySelector("#detectButton");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#thresholdValue");
const fileMeta = document.querySelector("#fileMeta");
const errorMessage = document.querySelector("#errorMessage");
const modelStatus = document.querySelector("#modelStatus");
const detectionList = document.querySelector("#detectionList");
const downloadLink = document.querySelector("#downloadLink");

let imageDataUrl = null;
let selectedFilename = "uploaded_image";

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove("hidden");
}

function clearError() {
  errorMessage.classList.add("hidden");
  errorMessage.textContent = "";
}

function setFile(file) {
  clearError();
  if (!file || !file.type.startsWith("image/")) {
    showError("请选择 JPG、PNG、WEBP 或 BMP 图片。");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    selectedFilename = file.name;
    imageDataUrl = reader.result;
    inputPreview.src = imageDataUrl;
    inputPreview.classList.remove("hidden");
    dropZone.classList.add("hidden");
    fileMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    detectButton.disabled = false;
  };
  reader.onerror = () => showError("浏览器无法读取该图片。");
  reader.readAsDataURL(file);
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || "模型状态不可用");
    if (status.gallery_enabled === false) {
      document.querySelector('a[href="/gallery.html"]').hidden = true;
    }
    modelStatus.classList.add("ready");
    const version = `${status.model_name} · epoch ${status.checkpoint_epoch ?? "未知"}`;
    modelStatus.querySelector("span:last-child").textContent = `模型已就绪 · ${version}`;
    document.querySelector("#modelEpoch").textContent = version;
    threshold.value = status.default_threshold;
    thresholdValue.textContent = Number(status.default_threshold).toFixed(2);
    document.querySelector("#gpuName").textContent = status.gpu;
  } catch (error) {
    modelStatus.querySelector("span:last-child").textContent = "模型连接失败";
    showError(error.message);
  }
}

function renderDetections(detections) {
  if (!detections.length) {
    detectionList.innerHTML = '<p class="empty-copy">当前阈值下未检测到敌人或队友。</p>';
    return;
  }
  detectionList.innerHTML = detections.map((item, index) => {
    const [x1, y1, x2, y2] = item.box;
    return `<div class="detection-item">
      <b>${String(index + 1).padStart(2, "0")} · ${item.label}</b>
      <span>置信度 ${(item.score * 100).toFixed(1)}%</span>
      <code>x1 ${x1} · y1 ${y1} · x2 ${x2} · y2 ${y2}</code>
    </div>`;
  }).join("");
}

async function detect() {
  if (!imageDataUrl) return;
  clearError();
  detectButton.disabled = true;
  detectButton.textContent = "正在检测…";
  try {
    const response = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageDataUrl,
        filename: selectedFilename,
        threshold: Number(threshold.value),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "检测失败");
    resultImage.src = result.image_data_url;
    resultImage.classList.remove("hidden");
    resultEmpty.classList.add("hidden");
    downloadLink.href = result.image_data_url;
    downloadLink.classList.remove("hidden");
    document.querySelector("#detectionCount").textContent = result.detection_count;
    document.querySelector("#inferenceTime").textContent = `${result.inference_ms} ms`;
    document.querySelector("#predictionTime").textContent = `${result.prediction_ms} ms`;
    document.querySelector("#modelEpoch").textContent = `${result.model_name} · epoch ${result.checkpoint_epoch ?? "未知"}`;
    document.querySelector("#gpuName").textContent = result.gpu;
    document.querySelector("#savedPath").textContent = result.saved_path;
    renderDetections(result.detections);
  } catch (error) {
    showError(error.message);
  } finally {
    detectButton.disabled = false;
    detectButton.textContent = "检测敌人 / 队友";
  }
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
threshold.addEventListener("input", () => { thresholdValue.textContent = Number(threshold.value).toFixed(2); });
detectButton.addEventListener("click", detect);

["dragenter", "dragover"].forEach(eventName => {
  dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", event => setFile(event.dataTransfer.files[0]));
inputPreview.addEventListener("click", () => fileInput.click());

loadStatus();
