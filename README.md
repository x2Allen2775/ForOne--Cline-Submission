# 🩸 Anemia-Screen — AI Conjunctiva Anemia Screening & Triage Pipeline

A local-first, **free** (no paid APIs) backend pipeline that screens for anemia
from a photo of the **conjunctiva** (inner lower eyelid) and produces an
**actionable, WHO-aligned triage report**. Built for the scale of India's anemia
burden (NFHS-5: ~52% of pregnant women are anemic).

> **Status:** Pipeline runs end-to-end *today* on untrained demo weights. Drop a
> trained checkpoint at `ml/checkpoints/anemia_mobilenetv3.pt` and it is picked up
> automatically — no code changes needed.

---

## 🧠 What it does (pipeline stages)

```
Raw image + patient JSON
        │
        ▼
[1] Quality Gate      → blur (Laplacian var) + brightness/exposure + tissue check
        │  (reject bad frames early → prevents false positives)
        ▼
[2] Preprocessing     → OpenCV auto-crop conjunctiva ROI + colour normalization
        │
        ▼
[3] Inference         → MobileNetV3 (timm), 3-class: Normal / Mild-Moderate / Severe
        │
        ▼
[4] Grad-CAM          → base64 heatmap (proves it looks at the vascular bed)
        │
        ▼
[5] WHO Rules Engine  → combines vision score + symptoms/age/pregnancy → triage
        │
        ▼
Structured JSON Triage Report (+ CSV for batch)
```

### Clinical classes (WHO thresholds, see plan §4)
| Class | Hemoglobin |
|---|---|
| **Normal** | Hb > 11.0 g/dL |
| **Mild/Moderate** | 7.0 – 10.9 g/dL |
| **Severe** | Hb < 7.0 g/dL |

### Safety-first triage (WHO/MoHFW Rules Engine)
Implements the **Anemia Loop** decision table. Rules are evaluated
**most-urgent-first**, and crucially follow **"image is a screen, not a
diagnosis"** — an image alone never triggers treatment; only a *measured* Hb or a
danger sign does. Triage actions: `REFER_NOW`, `CONFIRM_48H`, `CONFIRM_1WEEK`,
`TREAT_AND_RECHECK`, `ROUTINE`, `RETAKE_PHOTO`.

| Priority | Condition | Action |
|---|---|---|
| 1 | Danger sign (breathless, fainting, chest pain, severe pallor…) | `REFER_NOW` |
| 2 | **Measured** Hb < 7 g/dL | `REFER_NOW` |
| 3 | **Measured** Hb mild/moderate | `TREAT_AND_RECHECK` (+ IFA, re-check 4–8 wks) |
| 4 | **Measured** Hb normal | `ROUTINE` |
| 5 | High image risk (p ≥ 0.60) | `CONFIRM_48H` (blood test first) |
| 6 | Medium risk (p ≥ 0.30) or ≥2 symptoms | `CONFIRM_1WEEK` |
| 7 | Low risk, no symptoms | `ROUTINE` |

Extra clinical guards:
- **Group-specific WHO cut-offs** (pregnant / child 6–59 mo / default), 4-level
  bands (Normal/Mild/Moderate/Severe) — all in `app/clinical_config.yaml`.
- **Pre-eclampsia flag** for pregnant women (swelling + headache/blurred vision).
- **Measured Hb overrides** the image when provided (confirmatory-test-first).

> **Config-driven:** every threshold lives in `app/clinical_config.yaml`, so you
> can switch to WHO-2024 trimester-specific cut-offs or MoHFW updates **without
> changing code**.

### Quality Gate checks
Blur (variance of Laplacian), brightness/exposure window, exposure clipping,
**specular-glare** detection, and minimum conjunctiva-tissue fraction — bad
frames are rejected with a clear reason before any inference runs.

---

## 🚀 Quick start

```bash
# 1) Create + activate the environment (conda)
conda env create -f environment.yml        # OR: conda create -n anemia python=3.11
conda activate anemia
pip install -r requirements.txt

# 2) Generate synthetic demo images (works before you download real data)
python -m scripts.make_sample_images

# 3) ⭐ Terminal analyzer — full summary in the terminal + pop-up Grad-CAM window
python -m scripts.analyze_cli --image data/sample/pale_pregnant.png
#   • prints a color-coded report (quality gate, model, WHO triage) to the terminal
#   • pops up a 3-panel window: original · preprocessed · Grad-CAM overlay
#   • also saves outputs/<image>_report.png
#   extra flags: --hb 9.5  --pregnant  --symptoms fatigue,fainting  --no-window  --json

# 4) (Optional) Run the API + web UI instead
uvicorn app.main:app --reload
# → open http://127.0.0.1:8000/        (self-contained web UI — upload & triage)
# → open http://127.0.0.1:8000/docs    (interactive Swagger, served offline)

# 5) Analyze one image via the API (micro demo)
curl -s -X POST http://127.0.0.1:8000/analyze \
  -F "image=@data/sample/pale_pregnant.png" \
  -F 'metadata={"age_months":312,"is_pregnant":true,"trimester":2,"symptoms":["tiredness"]}' \
  -F "with_gradcam=true" | python -m json.tool

# 6) Batch demo (macro) → CSV "ministry dashboard"
python -m scripts.batch_process --mode local --input data/sample --out outputs/triage.csv
```

---

## 🗂 Project structure

```
anemia-screen/
├── app/                      # FastAPI service + pipeline stages
│   ├── main.py               # web UI (/), POST /analyze, GET /health, offline /docs
│   ├── static/               # self-contained web UI (index.html, app.js) + vendored Swagger assets
│   ├── config.py             # all tunable thresholds (env-overridable)
│   ├── clinical_config.yaml  # WHO/MoHFW Hb bands + triage params (config-driven)
│   ├── clinical.py           # clinical config loader
│   ├── schemas.py            # Pydantic request/response contracts
│   └── pipeline/
│       ├── quality_gate.py   # blur/brightness/tissue checks
│       ├── preprocessing.py  # auto-crop + colour normalize + tensor
│       ├── inference.py      # MobileNetV3 + graceful demo-weights fallback
│       ├── gradcam.py        # Grad-CAM → base64 heatmap
│       ├── rules_engine.py   # WHO clinical triage logic
│       └── orchestrator.py   # wires the stages together
├── ml/                       # train / dataset / ONNX export
│   ├── dataset.py            # patient-wise split + albumentations
│   ├── train.py              # python -m ml.train --data-dir data/raw
│   └── export_onnx.py
├── scripts/
│   ├── batch_process.py      # folder → triage CSV (api or local mode)
│   └── make_sample_images.py # synthetic demo images + metadata
├── notebooks/walkthrough.ipynb
├── tests/                    # pytest: rules, quality gate, preprocessing, API
├── data/{raw,sample}/        # raw = your downloaded datasets (gitignored)
└── requirements.txt / environment.yml
```

---

## 🏋️ Training on your real data (done once, reproducible)

Datasets supported (auto-detected in the project root):
- **Eyes-defy-anemia** (`dataset anemia/` — India + Italy): per-subject folders, Hb in `India.xlsx`/`Italy.xlsx`.
- **CP-AnemiC** (conjunctiva, Ghana): extracted from its `.rar`; Hb in `Anemia_Data_Collection_Sheet.xlsx`.
- **Palm** (`Palm/`): palm/nail-bed PNGs labelled anemic / non-anemic in the filename.

```bash
# 1) Build unified manifests (data/manifests/eye.csv + palm.csv)
python -m ml.build_manifest

# 2) Train each modality (separate MobileNetV3 models, patient-wise splits)
python -m ml.train --modality eye  --epochs 12 --batch-size 32
python -m ml.train --modality palm --epochs  8 --batch-size 64

# 3) Evaluate honestly through the SERVING preprocessing path
python -m ml.evaluate --modality palm --max-images 200
python -m ml.evaluate --modality eye  --max-images 200
```

Checkpoints land in `ml/checkpoints/anemia_{eye,palm}_mobilenetv3.pt` and are
**auto-loaded** by the API/CLI/web UI on next start.

Key training design points:
- **Train/serve consistency** — training uses the *exact* serving preprocessing
  (modality-aware crop → gray-world + CLAHE → resize → ImageNet normalize); a
  train/serve mismatch silently degrades real-world accuracy.
- **Patient-wise splits** — no patient appears in both train and validation.
- **Class-weighted loss** for the imbalanced label distribution.
- **Data-calibrated quality gate** — thresholds were measured against the actual
  dataset distributions (conjunctiva crops are dark with black backgrounds;
  palms are smooth skin with naturally low edge content, so blur limits are
  modality-specific).

---

## ✅ Testing

```bash
pytest            # rules engine, quality gate, preprocessing, API (in-process)
```

---

## 🛠 How Cline was used in the build
This entire pipeline — architecture, modules, WHO rules engine, tests, and demo
tooling — was scaffolded and iterated with **Cline** as the AI coding partner,
from environment setup through validation.

---

## ⚠️ Disclaimer
This is a screening & triage **decision-support** prototype, **not** a diagnostic
device. All positive/severe findings must be confirmed with a laboratory
hemoglobin test by a qualified health worker.
