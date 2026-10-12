const $ = selector => document.querySelector(selector);
let items = [], selected = new Set(), rosterHash = "", zoomId = 1;
let timer = null, saveQueue = Promise.resolve(), busy = false, revision = 0, savedRevision = 0;
const cards = new Map();

function showError(error) { $("#galleryError").textContent = error.message || String(error); $("#galleryError").classList.remove("hidden"); }
async function api(url, data) {
  const response = await fetch(url, data ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)} : {cache: "no-store"});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败，请重试。");
  return result;
}
function payload() { return {selected_ids: [...selected].sort((a,b) => a-b), roster_hash: rosterHash}; }
function update() {
  $("#selectionCount").textContent = `已选 ${selected.size} / ${items.length} 张`;
  const only = $("#selectedOnly").checked;
  cards.forEach((card, id) => {
    card.classList.toggle("selected", selected.has(id));
    card.classList.toggle("hidden", only && !selected.has(id));
    card.querySelector("input").checked = selected.has(id);
  });
  $("#emptySelection").classList.toggle("hidden", !only || selected.size !== 0);
  $("#finishButton").disabled = busy || !$("#reviewedAll").checked || !selected.size;
  $("#zoomSelected").checked = selected.has(zoomId);
}
function save() {
  clearTimeout(timer);
  const data = payload(), version = revision;
  $("#saveStatus").textContent = "正在保存…";
  const request = saveQueue.catch(() => {}).then(() => api("/api/gallery/selection", data));
  saveQueue = request;
  request.then(() => {
    savedRevision = Math.max(savedRevision, version);
    $("#saveStatus").textContent = savedRevision === revision ? "已保存，可关闭页面" : "有新选择待保存…";
  }).catch(error => { $("#saveStatus").textContent = "保存失败，请点保存选择重试"; showError(error); });
  return request;
}
function toggle(id, checked) {
  if (busy) return;
  checked ? selected.add(id) : selected.delete(id);
  revision++;
  $("#reviewedAll").checked = false;
  $("#finishResult").classList.add("hidden");
  $("#saveStatus").textContent = "选择已更改，等待保存…";
  update(); clearTimeout(timer); timer = setTimeout(save, 400);
}
function openZoom(id) {
  zoomId = id;
  const item = items[id-1];
  $("#zoomTitle").textContent = `编号 ${String(id).padStart(3, "0")} · ${item.encounter} · ${item.split === "train" ? "训练集" : "验证集"}`;
  $("#zoomImage").src = item.image;
  $("#zoomImage").alt = `编号 ${id} 的原图`;
  $("#zoomSelected").checked = selected.has(id);
  $("#previousImage").disabled = id === 1;
  $("#nextImage").disabled = id === items.length;
  if (!$("#zoomDialog").open) $("#zoomDialog").showModal();
}
function showFinal(result) {
  const box = $("#finishResult"); box.replaceChildren();
  const text = document.createElement("p"); text.textContent = `已生成 ${result.selected_count} 张精选任务。其余 ${result.unselected_count} 张记录为人工初筛无队友，旧敌人框保留。`;
  const link = document.createElement("a"); link.href = result.project_url; link.target = "_blank"; link.rel = "noopener"; link.textContent = "打开精选标注项目 →";
  box.append(text, link); box.classList.remove("hidden");
}
async function load() {
  try {
    const data = await api("/api/gallery");
    items = data.items; rosterHash = data.roster_hash; selected = new Set(data.state.selected_ids);
    const fragment = document.createDocumentFragment();
    for (const item of items) {
      const card = document.createElement("article"); card.className = "gallery-card"; card.id = `image-${item.id}`;
      const header = document.createElement("header");
      const number = document.createElement("strong"); number.className = "image-number"; number.textContent = String(item.id).padStart(3, "0");
      const tag = document.createElement("span"); tag.className = "split-tag"; tag.textContent = `${item.encounter.replace("encounter_", "交战 ")} · ${item.split === "train" ? "训练" : "验证"}`;
      header.append(number, tag);
      const image = document.createElement("img"); image.src = item.thumbnail; image.loading = "lazy"; image.decoding = "async"; image.alt = `编号 ${item.id} 游戏截图`; image.title = item.name;
      image.addEventListener("error", () => { image.alt = `编号 ${item.id} 加载失败，请刷新后检查，勿判为无队友`; });
      const footer = document.createElement("footer");
      const label = document.createElement("label"), checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.setAttribute("aria-label", `选中编号 ${item.id}`);
      checkbox.addEventListener("change", () => toggle(item.id, checkbox.checked));
      label.append(checkbox, document.createTextNode(" 有队友 / 待确认"));
      const zoom = document.createElement("button"); zoom.type = "button"; zoom.className = "quiet-button"; zoom.textContent = "放大"; zoom.setAttribute("aria-label", `放大编号 ${item.id}`); zoom.addEventListener("click", () => openZoom(item.id));
      footer.append(label, zoom); card.append(header, image, footer); cards.set(item.id, card); fragment.append(card);
    }
    $("#gallery").append(fragment); update();
    if (data.state.completed && data.state.final_result) { $("#reviewedAll").checked = true; showFinal(data.state.final_result); update(); }
  } catch (error) { showError(error); }
}
$("#selectedOnly").addEventListener("change", update);
$("#saveButton").addEventListener("click", () => { save().catch(() => {}); });
$("#reviewedAll").addEventListener("change", update);
$("#jumpButton").addEventListener("click", () => {
  const id = Number($("#jumpId").value);
  if (!Number.isInteger(id) || !cards.has(id)) { showError(new Error(`请输入 1–${items.length} 的编号。`)); return; }
  $("#selectedOnly").checked = false; update();
  cards.forEach(card => card.classList.remove("located"));
  cards.get(id).classList.add("located"); cards.get(id).scrollIntoView({block: "center"});
});
$("#closeZoom").addEventListener("click", () => $("#zoomDialog").close());
$("#previousImage").addEventListener("click", () => openZoom(Math.max(1, zoomId-1)));
$("#nextImage").addEventListener("click", () => openZoom(Math.min(items.length, zoomId+1)));
$("#zoomSelected").addEventListener("change", () => toggle(zoomId, $("#zoomSelected").checked));
$("#zoomDialog").addEventListener("keydown", event => {
  if (event.key === "ArrowLeft" && zoomId > 1) { event.preventDefault(); openZoom(zoomId-1); }
  if (event.key === "ArrowRight" && zoomId < items.length) { event.preventDefault(); openZoom(zoomId+1); }
});
$("#finishButton").addEventListener("click", async () => {
  busy = true; update();
  document.querySelectorAll(".gallery-card input, #zoomSelected, #saveButton").forEach(e => e.disabled = true);
  $("#finishButton").textContent = "正在创建标注项目…";
  try { await save(); const result = await api("/api/gallery/finalize", {...payload(), reviewed_all: $("#reviewedAll").checked}); showFinal(result); }
  catch (error) { showError(error); }
  finally { busy = false; document.querySelectorAll(".gallery-card input, #zoomSelected, #saveButton").forEach(e => e.disabled = false); $("#finishButton").textContent = "生成选中图片标注项目"; update(); }
});
window.addEventListener("pagehide", () => {
  if (rosterHash && savedRevision < revision && !busy) navigator.sendBeacon("/api/gallery/selection", new Blob([JSON.stringify(payload())], {type: "application/json"}));
});
load();
