"use strict";

const MAX_BYTES = 10 * 1024 * 1024;
const LABELS = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumor", pituitary: "Pituitary" };
const ERRORS = {
  not_an_image: ["Not an image", "The file couldn't be read as an image. Use a PNG, JPEG or BMP slice."],
  bad_size: ["Unusual image size", "The image is too small (under 64 px) or too large to be an MRI slice."],
  file_too_large: ["File too large", "The limit is 10 MB."],
  model_not_loaded: ["Model not ready", "The server is still loading the model. Try again in a moment."],
  network: ["Can't reach the server", "Check that the API is running and try again."],
};

const $ = (id) => document.getElementById(id);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
let info = null;

function show(state) {
  for (const s of ["empty", "loading", "error", "result"]) $(`state-${s}`).hidden = s !== state;
}

function showError(code) {
  const [title, text] = ERRORS[code] || ["Something went wrong", "Please try again."];
  $("error-title").textContent = title;
  $("error-text").textContent = text;
  show("error");
}

async function loadInfo() {
  try {
    const res = await fetch("/v1/info");
    if (!res.ok) throw new Error();
    info = await res.json();
    $("model-version").textContent = `v${info.model_version}`;
    const m = info.metrics || {};
    if (m.ood_auroc) $("stat-ood").textContent = `OOD AUROC ${m.ood_auroc.toFixed(3)} on photos, X-rays, CT, spine MRI`;
    if (m.test_macro_f1) $("stat-f1").textContent = `Test macro F1 ${m.test_macro_f1.toFixed(3)}`;
    if (m.test_accuracy_when_answered && m.test_uncertain_rate !== undefined) {
      $("stat-line").textContent =
        `On held-out test MRIs: ${pct(m.test_uncertain_rate)} are sent to review as uncertain; ` +
        `the answered rest is ${pct(m.test_accuracy_when_answered)} accurate` +
        (m.test_tumor_sensitivity ? `, and ${pct(m.test_tumor_sensitivity)} of tumors are detected.` : ".");
    }
  } catch {
    $("model-version").textContent = "offline";
  }
}

const VERDICTS = {
  ok: ["Prediction", "ok", (r) => `Calibrated confidence ${pct(r.confidence)}.`],
  uncertain: ["Needs review", "uncertain", (r) =>
    `Leaning ${LABELS[r.prediction]}, but calibrated confidence is ${pct(r.confidence)}, ` +
    `below the ${pct(info?.uncertain_threshold ?? 0.86)} needed to answer. A human should look at this one.`],
  rejected_ood: ["Not answered", "rejected", () =>
    "This image doesn't look like the brain MRIs the model was trained on, so it won't guess. " +
    "Other scanners, crops or body parts can trigger this."],
  rejected_input: ["Not answered", "rejected", () =>
    "This doesn't look like a grayscale MRI slice (for example, a color photo)."],
};

function render(r) {
  const [chip, tone, text] = VERDICTS[r.status];
  const answered = r.status === "ok" || r.status === "uncertain";
  $("verdict").className = `verdict ${tone}`;
  $("status-chip").textContent = chip;
  $("verdict-label").textContent = answered ? LABELS[r.prediction] : "Out of scope";
  $("verdict-text").textContent = text(r);

  $("input-img").src = `data:image/png;base64,${r.input_png_base64}`;
  const heat = $("heat-img");
  if (r.heatmap_png_base64) {
    heat.src = `data:image/png;base64,${r.heatmap_png_base64}`;
    heat.hidden = false;
  } else {
    heat.removeAttribute("src");
    heat.hidden = true;
  }
  $("viewer-controls").hidden = !r.heatmap_png_base64;
  $("viewer-caption").textContent = r.heatmap_png_base64
    ? "What the model saw (224 × 224). Warm colors mark the regions that drove the prediction."
    : "What the model saw (224 × 224).";
  applyHeat();

  const list = $("probs");
  list.replaceChildren();
  $("probs-block").hidden = !answered;
  if (answered) {
    const top = r.prediction;
    for (const [name, p] of Object.entries(r.probabilities)) {
      const li = document.createElement("li");
      li.className = `prob${name === top ? " is-top" : ""}`;
      li.innerHTML =
        '<span class="prob-name"></span><span class="prob-bar"><span class="prob-fill"></span></span><span class="prob-value"></span>';
      li.querySelector(".prob-name").textContent = LABELS[name] || name;
      li.querySelector(".prob-value").textContent = pct(p);
      list.append(li);
      const fill = li.querySelector(".prob-fill");
      void fill.offsetWidth; // start the transition from 0
      fill.style.width = `${Math.max(p * 100, 0.5)}%`;
    }
  }

  $("fact-confidence").textContent = answered ? pct(r.confidence) : "–";
  const threshold = r.ood_threshold;
  if (r.status !== "rejected_input" && r.ood_score !== null && threshold) {
    const ratio = r.ood_score / threshold;
    $("fact-ood").textContent = ratio <= 1 ? "passed" : "failed";
    const pos = Math.min(ratio / 1.5, 1) * 100;
    $("ood-fill").style.left = `${pos}%`;
    $("ood-mark").style.left = `${100 / 1.5}%`;
    $("ood-meter").hidden = false;
  } else {
    $("fact-ood").textContent = r.status === "rejected_input" ? "skipped" : "–";
    $("ood-meter").hidden = true;
  }
  $("fact-latency").textContent = `${Math.round(r.latency_ms)} ms`;
  $("fact-version").textContent = `v${r.model_version}`;
  show("result");
}

function applyHeat() {
  const heat = $("heat-img");
  heat.style.opacity = $("heat-toggle").checked ? $("heat-opacity").value / 100 : 0;
}

async function analyze(file) {
  if (!file) return;
  if (file.size > MAX_BYTES) return showError("file_too_large");
  show("loading");
  const body = new FormData();
  body.append("file", file, "upload");
  try {
    const res = await fetch("/v1/predict", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return showError(data.error);
    render(data);
  } catch {
    showError("network");
  }
}

function setActiveSample(button) {
  for (const b of document.querySelectorAll(".sample")) b.classList.toggle("active", b === button);
}

function init() {
  loadInfo();
  const input = $("file");
  const drop = $("drop");
  input.addEventListener("change", () => { setActiveSample(null); analyze(input.files[0]); input.value = ""; });

  for (const ev of ["dragenter", "dragover"]) {
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragging"); });
  }
  drop.addEventListener("drop", (e) => { setActiveSample(null); analyze(e.dataTransfer.files[0]); });

  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) { setActiveSample(null); analyze(item.getAsFile()); }
  });

  for (const button of document.querySelectorAll(".sample")) {
    button.addEventListener("click", async () => {
      setActiveSample(button);
      const blob = await (await fetch(button.dataset.src)).blob();
      analyze(new File([blob], "sample.jpg", { type: blob.type }));
    });
  }

  $("heat-toggle").addEventListener("change", applyHeat);
  $("heat-opacity").addEventListener("input", applyHeat);
}

document.addEventListener("DOMContentLoaded", init);
