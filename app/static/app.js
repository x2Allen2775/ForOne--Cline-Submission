// Anemia-Screen UI — vanilla JS, no external dependencies (works fully offline).

const SYMPTOMS = [
  "fatigue", "dizziness", "breathlessness", "palpitations",
  "fainting", "chest pain", "severe pallor", "headache",
  "swelling", "blurred vision",
];
const selected = new Set();
let fileObj = null;

// Build symptom chips
const chipsEl = document.getElementById("chips");
SYMPTOMS.forEach((s) => {
  const el = document.createElement("span");
  el.className = "chip";
  el.textContent = s;
  el.onclick = () => {
    if (selected.has(s)) { selected.delete(s); el.classList.remove("on"); }
    else { selected.add(s); el.classList.add("on"); }
  };
  chipsEl.appendChild(el);
});

// File preview
const fileInput = document.getElementById("file");
fileInput.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  fileObj = f;
  const img = document.getElementById("preview");
  img.src = URL.createObjectURL(f);
  img.style.display = "block";
  document.getElementById("dropHint").textContent = f.name;
});

// Drag & drop
const drop = document.getElementById("drop");
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.style.borderColor = "#e23b4e"; })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.style.borderColor = ""; })
);
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) {
    fileObj = f;
    fileInput.files = e.dataTransfer.files;
    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(f);
    img.style.display = "block";
    document.getElementById("dropHint").textContent = f.name;
  }
});

// Model status badge
fetch("/health").then((r) => r.json()).then((d) => {
  const m = d.model || {};
  const trained = m.trained ? "trained model" : "\u26a0 untrained demo weights";
  document.getElementById("status").innerHTML =
    `device: <b>${m.device || "?"}</b> \u00b7 ${trained}`;
}).catch(() => {
  document.getElementById("status").textContent = "model status unavailable";
});

function buildMetadata() {
  const meta = {
    age_months: parseInt(document.getElementById("age_months").value || "0", 10),
    sex: document.getElementById("sex").value,
    is_pregnant: document.getElementById("is_pregnant").value === "true",
    symptoms: Array.from(selected),
  };
  const hb = document.getElementById("measured_hb").value;
  if (hb !== "" && hb !== null) meta.measured_hb = parseFloat(hb);
  return meta;
}

async function analyze() {
  if (!fileObj) { alert("Please choose a conjunctiva photo first."); return; }
  const go = document.getElementById("go");
  const resEl = document.getElementById("result");
  go.disabled = true;
  go.innerHTML = '<span class="spin"></span> Analyzing\u2026';
  resEl.innerHTML = '<div class="muted">Running quality gate \u2192 model \u2192 Grad-CAM \u2192 triage\u2026</div>';

  const fd = new FormData();
  fd.append("image", fileObj);
  fd.append("metadata", JSON.stringify(buildMetadata()));
  fd.append("with_gradcam", document.getElementById("with_gradcam").checked ? "true" : "false");

  try {
    const r = await fetch("/analyze", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) {
      resEl.innerHTML = `<div class="banner b-None">Error ${r.status}</div>
        <pre class="muted">${escapeHtml(JSON.stringify(data.detail || data, null, 2))}</pre>`;
      return;
    }
    renderResult(data);
  } catch (err) {
    resEl.innerHTML = `<div class="banner b-None">Request failed: ${escapeHtml(String(err))}</div>`;
  } finally {
    go.disabled = false;
    go.textContent = "Analyze";
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderResult(data) {
  const t = data.triage || {};
  const v = data.vision;
  const q = data.quality || {};
  const urg = t.urgency || "None";
  const icon = { High: "\ud83d\udd34", Medium: "\ud83d\udfe0", Low: "\ud83d\udfe2", None: "\u26aa\ufe0f" }[urg] || "\u26aa\ufe0f";

  let html = `<div class="banner b-${urg}">${icon} ${escapeHtml(t.action_code || data.status)} \u00b7 ${urg} urgency</div>`;
  html += `<div style="font-size:14px;margin-bottom:10px">${escapeHtml(t.action || "")}</div>`;

  html += `<div class="kv">`;
  html += `<div class="k">Status</div><div>${escapeHtml(data.status)}</div>`;
  if (v) {
    html += `<div class="k">Risk class</div><div>${escapeHtml(v.risk_class)} (conf ${(v.confidence*100).toFixed(1)}%)</div>`;
    html += `<div class="k">P(anemic)</div><div>${(v.p_anemic*100).toFixed(1)}%</div>`;
  }
  if (t.hb_band) html += `<div class="k">Hb band</div><div>${escapeHtml(t.hb_band)}</div>`;
  if (t.recheck_in_weeks) html += `<div class="k">Re-check</div><div>in ${t.recheck_in_weeks} weeks</div>`;
  if (t.flags && t.flags.length) html += `<div class="k">Flags</div><div>${escapeHtml(t.flags.join(", "))}</div>`;
  html += `<div class="k">Rule fired</div><div><code>${escapeHtml(t.rule_fired || "")}</code></div>`;
  html += `<div class="k">Latency</div><div>${(data.processing_ms||0).toFixed(0)} ms</div>`;
  html += `</div>`;

  if (t.rationale) html += `<div class="muted" style="margin:8px 0">${escapeHtml(t.rationale)}</div>`;

  // Probability bars
  if (v && v.probabilities) {
    html += `<div class="bars">`;
    for (const [name, p] of Object.entries(v.probabilities)) {
      html += `<div class="bar"><span class="name">${escapeHtml(name)}</span>
        <span class="track"><span class="fill" style="width:${(p*100).toFixed(1)}%"></span></span>
        <span>${(p*100).toFixed(0)}%</span></div>`;
    }
    html += `</div>`;
  }

  // Quality
  if (q.reasons && q.reasons.length) {
    html += `<div class="reasons">Quality gate: ${q.reasons.map(escapeHtml).join("; ")}</div>`;
  } else {
    html += `<div class="muted">Quality gate passed \u00b7 blur ${q.blur_variance} \u00b7 brightness ${q.brightness_mean} \u00b7 glare ${((q.glare_fraction||0)*100).toFixed(1)}%</div>`;
  }

  // Grad-CAM
  if (data.gradcam_png_base64) {
    html += `<div class="heat"><label>Grad-CAM heatmap (where the model looked)</label>
      <img src="data:image/png;base64,${data.gradcam_png_base64}" /></div>`;
  }

  // Warnings
  if (data.warnings && data.warnings.length) {
    html += `<div class="warn">${data.warnings.map(escapeHtml).join("<br>")}</div>`;
  }

  document.getElementById("result").innerHTML = html;
}
