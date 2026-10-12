const videoInput = document.querySelector("#videoInput");
const videoStart = document.querySelector("#videoStart");
const videoCancel = document.querySelector("#videoCancel");
const videoProgress = document.querySelector("#videoProgress");
const videoMeter = document.querySelector("#videoMeter");
const videoPreview = document.querySelector("#videoPreview");
const videoResult = document.querySelector("#videoResult");
const videoDownload = document.querySelector("#videoDownload");
let videoJob = null;
let videoUpload = null;
let videoTimer = null;
let videoBusy = false;

function videoIdle() {
  videoBusy = false;
  videoUpload = null;
  videoInput.disabled = false;
  videoStart.disabled = !videoInput.files.length;
  videoCancel.disabled = true;
}

videoInput.addEventListener("change", () => {
  videoStart.disabled = !videoInput.files.length || videoBusy;
  videoProgress.textContent = videoInput.files[0]?.name || "尚未选择视频";
});

videoStart.addEventListener("click", () => {
  const file = videoInput.files[0];
  if (!file || videoBusy) return;
  if (file.size > 2 * 1024 ** 3 || !file.size) {
    showError("请选择非空视频，最大 2 GB。");
    return;
  }
  clearError();
  videoBusy = true;
  videoJob = null;
  clearTimeout(videoTimer);
  videoInput.disabled = true;
  videoStart.disabled = true;
  videoCancel.disabled = false;
  videoResult.pause();
  videoResult.removeAttribute("src");
  videoResult.load();
  videoResult.classList.add("hidden");
  videoPreview.classList.add("hidden");
  videoDownload.classList.add("hidden");
  document.querySelector("#videoSavedPath").textContent = "";
  videoMeter.value = 0;
  const upload = videoUpload = new XMLHttpRequest();
  upload.open("POST", "/api/video");
  upload.setRequestHeader("Content-Type", "application/octet-stream");
  upload.setRequestHeader("X-Threshold", threshold.value);
  upload.upload.onprogress = event => {
    videoProgress.textContent = `上传中 · ${(event.loaded / file.size * 100).toFixed(1)}%`;
    videoMeter.value = event.loaded / file.size * 100;
  };
  upload.onload = () => {
    videoUpload = null;
    try {
      const job = JSON.parse(upload.responseText);
      if (upload.status !== 202) throw new Error(job.error || "视频上传失败");
      videoJob = job.id;
      pollVideo();
    } catch (error) {
      showError(error.message);
      videoProgress.textContent = "上传失败";
      videoIdle();
    }
  };
  upload.onerror = () => {
    showError("上传连接中断，请检查 WebUI 服务。");
    videoIdle();
  };
  upload.onabort = () => {
    videoProgress.textContent = "已取消上传";
    videoIdle();
  };
  upload.send(file);
});

async function pollVideo() {
  try {
    const response = await fetch(`/api/video/${videoJob}`, { cache: "no-store" });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "读取进度失败");
    const names = { processing: "逐帧检测中", encoding: "封装并校验视频中", completed: "生成完成", failed: "生成失败", cancelled: "已取消" };
    videoProgress.textContent = `${names[job.state]} · ${job.frames}${job.total_frames ? ` / ${job.total_frames}` : ""} 帧 · 视频 ${job.video_time_s.toFixed(2)} 秒 · 已用 ${job.elapsed_s} 秒`;
    if (job.total_frames) videoMeter.value = job.frames / job.total_frames * 100;
    else videoMeter.removeAttribute("value");
    if (job.preview_url && !videoPreview.dataset.loading) {
      videoPreview.dataset.loading = "1";
      videoPreview.onload = videoPreview.onerror = () => { delete videoPreview.dataset.loading; };
      videoPreview.src = `${job.preview_url}?frame=${job.frames}`;
      videoPreview.classList.remove("hidden");
    }
    if (job.state === "completed") {
      videoMeter.value = 100;
      videoResult.src = job.result_url;
      videoResult.classList.remove("hidden");
      videoPreview.classList.add("hidden");
      videoDownload.href = job.result_url;
      videoDownload.classList.remove("hidden");
      document.querySelector("#videoSavedPath").textContent = `自动保存：${job.saved_path}`;
      videoIdle();
    } else if (["failed", "cancelled"].includes(job.state)) {
      if (job.error) showError(job.error);
      videoIdle();
    } else videoTimer = setTimeout(pollVideo, 800);
  } catch (error) {
    showError(`${error.message}；正在重试读取进度。`);
    videoTimer = setTimeout(pollVideo, 2000);
  }
}

videoCancel.addEventListener("click", async () => {
  if (videoUpload) { videoUpload.abort(); return; }
  if (!videoJob) return;
  videoCancel.disabled = true;
  try {
    const response = await fetch(`/api/video/${videoJob}/cancel`, { method: "POST" });
    if (!response.ok) throw new Error("取消失败");
    videoProgress.textContent = "正在取消…";
  } catch (error) {
    showError(error.message);
    videoCancel.disabled = false;
  }
});
