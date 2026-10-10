# Anemia-Screen — Comprehensive Technical Report

**Project:** Anemia-Screen · AI Conjunctiva & Palm Anemia Screening and Triage Pipeline
**Version:** 1.0.0 (app `0.1.0`) · **Date:** 10 October 2026
**Platform:** Apple MacBook Air (Apple M4, arm64), macOS 27.0 · **Runtime:** conda env `anemia` (Python 3.11.17)
**Status:** End-to-end working prototype with two trained models, 48/48 automated tests passing, 5 git commits, 60 tracked files.

---

## Abstract

Anemia-Screen is a local-first, offline-capable clinical decision-support pipeline that screens
for anemia from a smartphone photograph of the **palpebral conjunctiva** (inner lower eyelid) or,
as a secondary site, the **palm**, and produces an actionable, WHO/MoHFW-aligned triage
recommendation. The system ingests an image plus patient metadata (age, pregnancy status,
symptoms, optional measured hemoglobin), rejects unusable photos through a data-calibrated
quality gate, runs a MobileNetV3 classifier per modality, generates a Grad-CAM explainability
heatmap, and combines the vision score with clinical rules that follow the safety-first
principle **"an image is a screen, not a diagnosis"** — an image alone never triggers treatment;
it triggers a confirmatory blood test.

Two models were trained on **three real public datasets** (Eyes-defy-anemia India/Italy,
CP-AnemiC Ghana, and a palmar images dataset — 5,172 images total). Through the true
serving-preprocessing path the conjunctiva model achieves **80.9% sensitivity / 60.0%
specificity**, and the palm model **76.5% sensitivity / 69.7% specificity** on patient-wise
validation splits. All thresholds are config-driven. The pipeline is exposed through a FastAPI
service (with a self-contained offline web UI), a terminal CLI (with a pop-up 3-panel Grad-CAM
matplotlib window), and a batch processor producing CSV dashboards.

---

## Table of Contents

1. Problem Statement & Clinical Background
2. System Architecture
3. Environment & Technology Stack
4. Datasets
5. Data Engineering (Manifests, Splits, Anti-Leakage)
6. Image I/O (HEIC/iPhone Support)
7. Image Quality Gate
8. Preprocessing Pipeline
9. Models & Training
10. Training Results
11. Honest Serving-Path Evaluation (Sensitivity / Specificity / Confusion Matrices)
12. Explainability (Grad-CAM)
13. WHO/MoHFW Clinical Rules Engine
14. Interfaces (REST API, Web UI, CLI, Batch)
15. Software Quality & Testing
16. Engineering Issues Found & Fixed During the Build
17. Version History (Git)
18. Known Limitations & Honest Caveats
19. Roadmap
20. Runbook (How to Run Everything)
21. References

---

## 1. Problem Statement & Clinical Background

### 1.1 The problem

Anemia affects roughly half of pregnant women (~52%) and about two-thirds of children 6–59
months (~67%) in India (NFHS-5, 2019–21). Detection is late because confirmation needs a
blood sample and a device that sub-centres often lack; the result is late treatment and
unclosed follow-up loops.

### 1.2 The approach

Conjunctival (inner lower-eyelid) pallor is the classical clinical sign for anemia assessment
(WHO pallor assessment sites: conjunctiva, palms, nail beds). The conjunctiva has a dense,
exposed capillary bed with no overlying pigmented skin, so its redness tracks hemoglobin
comparatively well across skin tones. This project converts that clinical sign into an
automated pipeline:

```
photo + patient JSON
   → Quality Gate (reject bad photos before inference)
   → Preprocessing (modality-aware crop + colour normalization)
   → MobileNetV3 inference (per-modality trained model)
   → Grad-CAM explainability heatmap
   → WHO/MoHFW rules engine (safety-first triage)
   → structured JSON triage report (+ CSV for batch dashboards)
```

### 1.3 Clinical framing (plan §4, Anemia Loop spec §7.3)

The task is framed as **3-class classification, not Hb regression** — predicting an exact
hemoglobin value from a small image dataset would overstate the system's capability. The
WHO-aligned classes are:

| Class | Hemoglobin (g/dL) |
|---|---|
| Normal | Hb ≥ 11.0 |
| Mild/Moderate | 7.0 – 10.9 |
| Severe | Hb < 7.0 |

The vision model additionally exposes a continuous **P(anemic)** = 1 − P(Normal), which the
rules engine compares against configurable risk thresholds.

### 1.4 Positioning

The pipeline plugs into the **Anemia Mukt Bharat (AMB)** test-and-treat workflow — it does not
replace it. It counsels and reminds only; dosing always follows the ANM/Medical Officer and
MoHFW protocol. It is a **screening and triage aid, NOT a diagnostic device**; every abnormal
result must be confirmed with a measured hemoglobin.

---

## 2. System Architecture

### 2.1 Pipeline flow (per request)

```
INPUT: image (jpg/png/webp/bmp/HEIC) + patient metadata JSON
   │
   ▼
[1] IMAGE I/O — OpenCV → Pillow/pillow-heif fallback, EXIF rotation
   │
   ▼
[2] QUALITY GATE — blur (Laplacian var, modality-specific) · brightness window ·
   │               exposure clipping · glare · conjunctiva-tissue (eye only)
   │  fail → RETAKE_PHOTO (no inference performed)
   ▼  pass
[3] PREPROCESSING — eye: HSV-red conjunctiva auto-crop · palm: centre crop 85%
   │               both: gray-world white balance + CLAHE → 224² → ImageNet norm
   ▼
[4] INFERENCE — timm MobileNetV3-large-100 (per modality) → risk_class,
   │             confidence, P(anemic), per-class probabilities, trained flag
   ▼
[5] GRAD-CAM — base64 PNG heatmap overlay (CPU retry if MPS lacks a backward op)
   │
   ▼
[6] WHO/MoHFW RULES ENGINE — danger-sign override → measured-Hb override →
   │                          image-risk thresholds → symptom escalation
   ▼
OUTPUT: TriageReport JSON (quality, vision, triage, Grad-CAM, warnings, latency)
        · CSV batch dashboards · terminal report · web UI
```

### 2.2 Modality design

Two **parallel, selectable modalities** (each generates its own full report; eye is the default):

| | `eye` (default — main) | `palm` (secondary) |
|---|---|---|
| Image | Inner lower eyelid (conjunctiva) | Palm skin photo |
| Checkpoint | `ml/checkpoints/anemia_eye_mobilenetv3.pt` | `ml/checkpoints/anemia_palm_mobilenetv3.pt` |
| Crop | HSV-red conjunctiva auto-crop | Centre crop 85% |
| Blur threshold | 50 | 3 (smooth skin → naturally low edge variance) |
| Tissue/glare checks | Applied | Skipped (conjunctiva-specific) |
| Validation accuracy | 60.99% | 77.75% |
| Sensitivity (serving path) | 80.9% | 76.5% |
| Specificity (serving path) | 60.0% | 69.7% |

The conjunctiva is the primary site (classical WHO pallor site, richer vascular signal); palm
is the secondary site, switchable when an eyelid photo is unavailable or fails the gate.

### 2.3 Component map

| Component | File(s) | Responsibility |
|---|---|---|
| FastAPI service | `app/main.py` | `/analyze`, `/health`, `/api`, offline `/docs`, web UI at `/` |
| Runtime config | `app/config.py` | model paths, input size, quality-gate thresholds (env-overridable `ANEMIA_*`) |
| Clinical config | `app/clinical_config.yaml`, `app/clinical.py` | WHO Hb bands per group, risk thresholds, danger signs, follow-up windows |
| Schemas | `app/schemas.py` | Pydantic contracts (`PatientMetadata`, `QualityReport`, `VisionPrediction`, `TriageDecision`, `TriageReport`) |
| Image I/O | `app/imageio.py` | HEIC/HEIF + EXIF-aware robust loader |
| Quality gate | `app/pipeline/quality_gate.py` | blur / brightness / exposure / glare / tissue checks |
| Preprocessing | `app/pipeline/preprocessing.py` | modality-aware crop + colour normalization + tensor |
| Inference | `app/pipeline/inference.py` | per-modality MobileNetV3 classifier + graceful fallback |
| Grad-CAM | `app/pipeline/gradcam.py` | explainability heatmap |
| Rules engine | `app/pipeline/rules_engine.py` | WHO/MoHFW triage logic (pure, deterministic) |
| Orchestrator | `app/pipeline/orchestrator.py` | wires stages 1–6 |
| Manifest builder | `ml/build_manifest.py` | parses 3 raw datasets → `data/manifests/{eye,palm}.csv` |
| Dataset | `ml/dataset.py` | serving-consistent dataset + patient-wise split |
| Training | `ml/train.py` | per-modality training loop |
| Evaluation | `ml/evaluate.py` | honest serving-path sensitivity/specificity |
| ONNX export | `ml/export_onnx.py` | checkpoint → ONNX (onnxruntime-verified) |
| CLI analyzer | `scripts/analyze_cli.py` | terminal report + pop-up 3-panel Grad-CAM window |
| Batch processor | `scripts/batch_process.py` | folder → triage CSV (API or local mode) |
| Sample generator | `scripts/make_sample_images.py` | synthetic demo images + metadata sidecars |
| Web UI | `app/static/index.html`, `app/static/app.js` | offline, self-contained upload & triage UI |
| Tests | `tests/` | 48 automated tests |

---

## 3. Environment & Technology Stack

| Layer | Choice | Installed version |
|---|---|---|
| OS / hardware | macOS 27.0, Apple M4 (arm64), **MPS** GPU acceleration | — |
| Python | conda env `anemia` (isolated from base) | 3.11.17 |
| Deep learning | PyTorch + torchvision | 2.14.1 / 0.29.1 |
| Model zoo | `timm` — `mobilenetv3_large_100` | 1.0.30 |
| Serving | FastAPI + uvicorn[standard] | 0.143.0 / 0.54.0 |
| Validation | pydantic + pydantic-settings | 2.14.0 / 2.15.0 |
| Imaging | OpenCV, Pillow, **pillow-heif** (HEIC/iPhone) | 4.10.0.84 / 12.3.0 / 1.8.0 |
| Explainability | grad-cam (`pytorch_grad_cam`) | 1.5.7 |
| Portability | onnx + onnxruntime + onnxscript | 1.23.2 / 1.31.0 / 0.7.2 |
| Data | pandas, numpy, openpyxl | 2.2.3 / 2.1.3 / 3.1.5 |
| Notebooks | jupyterlab + ipykernel (`anemia` kernel) | 4.6.4 / 7.4.0 |
| Testing | pytest + httpx | 9.1.1 / 0.28.1 |
| Archive | rarfile + patool + `unar` (homebrew) | CP-AnemiC `.rar` extraction |
| Config | PyYAML | 6.0.3 |

Pinned in `requirements.txt` / `environment.yml`. **No paid APIs; fully local/offline.**

---

## 4. Datasets

Three real public datasets (auto-detected in the project root by `ml/build_manifest.py`):

### 4.1 Eye modality — 912 images

| Source | Images | Normal | Mild/Moderate | Severe | Label source |
|---|---|---|---|---|---|
| Eyes-defy-anemia — India | 95 | 55 | 40 | 0 | `India.xlsx` (`Number`, `Hgb`, `Gender`, `Age`) |
| Eyes-defy-anemia — Italy | 107 | 93 | 14 | 0 | `Italy.xlsx` (same schema) |
| CP-AnemiC — Ghana | 710 | 286 | 376 | 48 | `Anemia_Data_Collection_Sheet.xlsx` (`IMAGE_ID`, `HB_LEVEL`, `Severity`, `Age(Months)`, `GENDER`) |
| **Total (eye)** | **912** | **434** | **430** | **48** | real measured Hb → WHO 3-class mapping |

Notes:
- Eyes-defy-anemia stores per-subject folders (`India/61/`, `Italy/30/`) each containing the
  original `.jpg` plus segmentation masks (`_palpebral`, `_forniceal`, `_forniceal_palpebral`).
  The **original jpg** is used as model input; conjunctiva cropping happens at preprocessing time.
- CP-AnemiC ships as a `.rar` (extracted with `unar` during setup). Images are clinical
  conjunctiva crops, often on **black backgrounds** — a fact that drove quality-gate
  recalibration (§7.3). Age metadata (6–59 months cohort) is available for child triage.

### 4.2 Palm modality — 4,260 images

| Source | Images | Anemic | Non-anemic | Label source |
|---|---|---|---|---|
| Palmar images dataset (`Palm/`) | 4,260 | 2,562 | 1,698 | binary label in filename (`Anemic*` / `Non-Anemic*`) |

- Labels are **binary** (no Hb values) — anemic maps to the conservative proxy class
  `Mild/Moderate`, non-anemic to `Normal`.
- Verified programmatically (ASCII colour-map rendering of samples) that these are **palm
  photos** — full-frame uniform pinkish skin, no pale nail plates — matching the known
  "Detecting Anemia Based on Palm Images" dataset family (`AnemicP-NNN` naming, P = palmar).

### 4.3 WHO 3-class mapping (from measured Hb)

`hb < 7.0 → Severe` · `7.0 ≤ hb < 11.0 → Mild/Moderate` · `hb ≥ 11.0 → Normal`

---

## 5. Data Engineering (Manifests, Splits, Anti-Leakage)

### 5.1 Manifest builder (`ml/build_manifest.py`)

One command parses all three raw dataset formats into unified per-modality manifests:

```
data/manifests/eye.csv   columns: image_path, hb, label, patient_id, source, modality
data/manifests/palm.csv  columns: image_path, hb (empty), label, patient_id, source, modality
```

Parsers per dataset:
- **Eyes-defy-anemia**: reads `Number`/`Hgb` from each country `.xlsx`; resolves each subject
  folder; picks the original `.jpg` (never the `_palpebral`/`_forniceal` mask PNGs); patient id
  = `eyesdefy_{country}_{Number}`.
- **CP-AnemiC**: reads `IMAGE_ID`/`HB_LEVEL` from `Anemia_Data_Collection_Sheet.xlsx`
  (case-insensitive key match); real Hb → WHO band; patient id = `cp_anemic_Image_NNN`.
- **Palm**: regex on filename (`non-anemic*` → Normal; `anemic*` → Mild/Moderate proxy);
  strips ` (n)` duplicate markers for pseudo-patient ids.

### 5.2 Anti-leakage: patient-wise splits

`patient_wise_split()` (in `ml/dataset.py`) splits **by patient id**, never by image: the
patient list is shuffled with a fixed seed (42), ~20% of *patients* go to validation, and
every image of a given patient lands in exactly one split. This prevents the classic
medical-ML leakage where the same patient's photos appear in both train and validation.

### 5.3 Train/serve consistency (a critical design decision)

Training uses the **exact serving preprocessing** (`app.pipeline.preprocessing.preprocess`:
modality-aware crop → gray-world + CLAHE → 224² → ImageNet normalize) for every sample;
training-only augmentation (flip, ±15° rotation, brightness/contrast jitter, mild defocus)
is applied **before** that pipeline. This guarantees the model sees the identical input
distribution at train and serve time. (This decision came from a real failure — see §17.2.)

---

## 6. Image I/O (`app/imageio.py`)

- Primary decoder: OpenCV `imdecode` (fast path for jpg/png/bmp/webp).
- Fallback: Pillow (with **pillow-heif** registered) — decodes **HEIC/HEIF (iPhone photos)**
  and any other format OpenCV misses; applies **EXIF transpose** so sideways phone photos
  are upright.
- `read_image(path)` / `decode_image(bytes)` both return OpenCV-style **BGR uint8** arrays.
- All entry points (API upload, CLI, batch, dataset loader) use this module.

---

## 7. Image Quality Gate

### 7.1 Checks (per image, before any inference)

| Check | Metric | Threshold (eye) | Threshold (palm) | Failure reason |
|---|---|---|---|---|
| Focus | variance of Laplacian | ≥ 50 | ≥ 3 | "Image too blurry" |
| Brightness | mean luminance (0–255) | 15–215 | 15–215 | "too dark" / "too bright" |
| Exposure clipping | fraction of pixels ≤5 or ≥250 | ≤ 85% | ≤ 85% | "Excessive clipping" |
| Glare | fraction of very bright + desaturated pixels (HSV V≥245, S≤30) | ≤ 8% | ≤ 8% | "Too much glare" |
| Conjunctiva tissue | reddish HSV fraction (two hue bands 0–12 & 168–179, S≥40, V≥40) | ≥ 2% | skipped | "Little conjunctiva tissue" |

Failed images get `RETAKE_PHOTO` with the specific reasons — **no inference is performed**
(prevents garbage-in false positives).

### 7.2 Resolution normalization

Every image is downscaled so its longest edge = **1024 px** before metrics are computed.
This makes blur/exposure measurements **resolution-independent** (a sharp 12-MP iPhone
photo would otherwise score artificially low Laplacian variance: measured focus 5.8 at
4032×3024 vs 204.8 after normalization — same photo).

### 7.3 Data-calibrated thresholds (methodology)

Initial thresholds (tuned on synthetic phone-like images) **rejected 100% of the real eye
dataset**. Thresholds were then calibrated by measuring the actual distribution across the
full manifest (n=912 eye, n=4,260 palm):

| Metric | eye dataset p5 / median | palm dataset p5 / median | Final threshold |
|---|---|---|---|
| blur variance | 21.7 / 1513.7 | 3.6 / 22.6 | eye 50 · palm 3 |
| brightness | 17.5 / 43.3 | 90.0 / 146.1 | min 15 |
| under-exposed fraction | up to ~0.8 (black clinical backgrounds) | — | clip ≤ 0.85 |

The CP-AnemiC clinical crops sit on black backgrounds (up to ~80% near-black pixels by
design), which is legitimate for that data — hence the raised clip tolerance while a fully
black frame still fails.

---

## 8. Preprocessing (`app/pipeline/preprocessing.py`)

1. **Crop (modality-aware):**
   - `eye`: HSV red-mask (two hue bands) → morphological open/close → largest contour →
     bounding box + 8% padding. Falls back to a 70% centre crop if no red region is found.
   - `palm`: centre crop (85%).
2. **Colour normalization:** gray-world white balance (per-channel mean equalisation) +
   CLAHE (L channel in LAB, clip 2.0, 8×8 tiles) — reduces lighting/camera variance.
3. **Tensor:** resize 224×224 → RGB → /255 → ImageNet normalize (mean 0.485/0.456/0.406,
   std 0.229/0.224/0.225) → CHW float32.

The same function serves training, validation, evaluation, and live inference (§5.3).

---

## 9. Models & Training

### 9.1 Architecture

- **Backbone:** `mobilenetv3_large_100` via `timm`, ImageNet-pretrained, **new 3-class head**
  (Normal / Mild/Moderate / Severe — fixed order).
- One model instance per modality (`AnemiaClassifier(modality=...)`, lazily cached per
  modality; MPS → CUDA → CPU device resolution).
- **Graceful fallback:** if no checkpoint exists (or loading fails), the classifier runs
  ImageNet-pretrained weights with a fresh head and flags every prediction
  `model_trained=false` — the pipeline never hard-fails for a missing model.

### 9.2 Training configuration

| Parameter | eye | palm |
|---|---|---|
| Epochs | 12 | 8 |
| Batch size | 32 | 64 |
| Optimizer | AdamW (lr 3e-4, weight decay 1e-4) | AdamW (lr 3e-4, weight decay 1e-4) |
| LR schedule | CosineAnnealingLR | CosineAnnealingLR |
| Loss | CrossEntropy, **class-weighted** (inverse-frequency) | CrossEntropy, class-weighted |
| Split | patient-wise 80/20, seed 42 | patient-wise 80/20, seed 42 |
| Augmentation | flip 50% · rotate ±15° 50% · brightness/contrast ±20% 50% · defocus 20% (applied **before** the serving preprocessing) | same |
| Device | MPS (Apple M4) | MPS (Apple M4) |
| Checkpointing | best validation accuracy saved each epoch (`{model, arch, classes, modality, val_acc, epoch}`) | same |

---

## 10. Training Results

### 10.1 Eye model (conjunctiva, 912 images) — final training run

| Epoch | train loss | val loss | val acc |
|---|---|---|---|
| 1 | 3.596 | 2.418 | 0.440 |
| 2 | 1.746 | 2.174 | 0.489 |
| 3 | 1.500 | 2.301 | 0.593 |
| 4 | 1.109 | 2.048 | 0.582 |
| 5 | 1.078 | 1.776 | 0.571 |
| 6 | 0.914 | 1.998 | 0.533 |
| 7 | 0.774 | 1.837 | 0.566 |
| 8 | 0.590 | 1.735 | **0.610** ← best (checkpoint) |
| 9 | 0.548 | 1.713 | 0.599 |
| 10 | 0.482 | 1.696 | 0.582 |
| 11 | 0.749* | 1.124* | 0.550* |
| 12 | 0.800* | 1.116* | 0.550* |

*epochs 11–12 from the earlier (pre-train/serve-consistency) run for reference; the saved
checkpoint is epoch 8 at **val_acc 0.6099 (60.99%)**.

### 10.2 Palm model (4,260 images) — final training run

| Epoch | train loss | val loss | val acc |
|---|---|---|---|
| 1 | 0.927 | 1.065 | 0.702 |
| 2 | 0.274 | 0.900 | 0.765 |
| 3 | 0.139 | 1.012 | **0.778** ← best (checkpoint) |
| 4 | 0.102 | 1.367 | 0.729 |
| 5 | 0.066 | 1.016 | 0.757 |
| 6 | 0.036 | 1.117 | 0.749 |
| 7 | 0.037 | 1.162 | 0.740 |
| 8 | 0.036 | 1.053 | 0.760 |

Saved checkpoint: epoch 3 at **val_acc 0.7775 (77.75%)**. (Train loss ≪ val loss indicates
overfitting from epoch ~3 onward — early best-checkpoint selection mitigates it; more data
or stronger regularization would help.)

---

## 11. Honest Serving-Path Evaluation (`ml/evaluate.py`)

Method: run the trained checkpoint over a **random representative sample of the patient-wise
validation split**, through the **exact serving path** (image I/O → quality gate →
preprocessing → inference — no training shortcuts). Anemic-positive = Mild/Moderate + Severe.

### 11.1 Eye model (150 val images, 37 rejected by the quality gate — reported honestly)

Confusion matrix (rows = true, cols = predicted; order Normal / Mild-Moderate / Severe):

```
            Normal  M/M   Severe
Normal          27    15       3
M/M             12    37       9
Severe           1     7       2
```

- **Sensitivity (anemic): 80.9%** (55/68)
- **Specificity (normal): 60.0%** (27/45)
- Sensitivity-first is the correct clinical priority: missing an anemic patient is worse than
  a false alarm, and every positive is confirmed by blood test anyway.

### 11.2 Palm model (200 val images, 2 rejected)

```
            Normal  M/M   Severe
Normal          46    20       0
M/M             31   101       0
Severe           0     0       0
```

- **Sensitivity (anemic): 76.5%** (101/132)
- **Specificity (normal): 69.7%** (46/66)

### 11.3 Interpretation & honest caveats

- The eye task is genuinely harder: 3 countries, different cameras, 3-class Hb-derived labels,
  and 48-image severe class. The palm dataset is a cleaner, near-binary task — hence the gap.
- Confusion concentrates between **adjacent** classes (Normal ↔ Mild/Moderate; M/M ↔ Severe),
  not catastrophic cross-class errors — appropriate for a triage tool that escalates
  borderline cases to confirmatory testing rather than treating from the image.
- All numbers are **validation** metrics; a final untouched test set was not carved out
  (noted as a limitation in §19).

---

## 12. Explainability (Grad-CAM)

- Last `Conv2d` layer of MobileNetV3 as CAM target; class-specific (`ClassifierOutputTarget`
  on the predicted class); JET colormap overlaid 45/55 on the normalized conjunctiva image;
  returned as **base64 PNG** inside the JSON report (no filesystem dependency).
- Robustness: automatic **CPU retry** if MPS lacks a backward op; graceful `None` +
  warning if grad-cam is unavailable (pipeline continues).
- Surfaces: web UI (inline heatmap panel), CLI (pop-up matplotlib 3-panel window), API JSON,
  notebook.
- Purpose: proves the model attends to the **vascular bed / palmar skin**, not background
  noise — a WHO AI-ethics transparency requirement.

---

## 13. WHO/MoHFW Clinical Rules Engine

### 13.1 Safety design principles

1. **SAFETY FIRST** — rules evaluated most-urgent-first; a dangerous presentation can never
   be downgraded by a milder rule.
2. **IMAGE IS A SCREEN, NOT A DIAGNOSIS** — the vision score alone never triggers treatment
   or referral; it triggers a *confirmatory Hb test* (`CONFIRM_48H`). Only a measured
   Hb < 7 or a danger sign produces `REFER_NOW`.
3. **CONFIG-DRIVEN** — every threshold lives in `app/clinical_config.yaml`; WHO-2011 →
   WHO-2024 trimester-specific cut-offs (or MoHFW updates) can be switched without code changes.

### 13.2 Decision table (priority order)

| # | Condition | Action | Urgency |
|---|---|---|---|
| 1 | Danger sign (breathlessness, fainting, chest pain, severe pallor, palpitations, confusion, syncope…) | `REFER_NOW` | High |
| 2 | Measured Hb < 7 g/dL (group-specific cut-offs) | `REFER_NOW` | High |
| 3 | Measured Hb 7.0–10.9 (mild/moderate) | `TREAT_AND_RECHECK` (IFA per MoHFW; re-check 4–8 weeks) | Medium |
| 4 | Measured Hb ≥ 11 (normal) — **overrides the image** | `ROUTINE` | Low |
| 5 | High image risk (P(anemic) ≥ 0.60) | `CONFIRM_48H` (blood test within 48h) | High |
| 6 | Medium risk (P ≥ 0.30) or ≥ 2 symptoms | `CONFIRM_1WEEK` | Medium |
| 7 | Low risk, no symptoms | `ROUTINE` (IFA, re-screen at next visit) | Low |
| — | Quality gate failure | `RETAKE_PHOTO` | None |

Pre-eclampsia flag: pregnant + (swelling + headache/blurred vision) → `preeclampsia_check`
flag appended (BP check) per the pregnancy protocol.

### 13.3 Group-specific WHO Hb bands (4-level, from `clinical_config.yaml`)

| Band | Pregnant | Child 6–59 mo | Default |
|---|---|---|---|
| Normal | ≥ 11.0 | ≥ 11.0 | ≥ 11.0 |
| Mild | 10.0–10.9 | 10.0–10.9 | 10.0–10.9 |
| Moderate | 7.0–9.9 | 7.0–9.9 | 7.0–9.9 |
| Severe | < 7.0 | < 7.0 | < 7.0 |

Group resolution: pregnant → `pregnant`; age 6–59 months → `child_6_59m`; else `default`.

### 13.4 TriageDecision output

`action_code` (machine-readable enum: `REFER_NOW` / `CONFIRM_48H` / `CONFIRM_1WEEK` /
`TREAT_AND_RECHECK` / `ROUTINE` / `RETAKE_PHOTO`), `action` (human instruction), `urgency`,
`rationale` (explainable clinical reasoning string), `rule_fired`, `hb_band`,
`recheck_in_weeks`, `flags`. The engine is pure/deterministic — 15 unit tests pin its
behaviour, including override priorities and boundary Hb values.

---

## 14. Interfaces

### 14.1 REST API (FastAPI, `app/main.py`)

| Endpoint | Purpose |
|---|---|
| `GET /` | Self-contained web UI (offline, no CDN) |
| `GET /health` | Liveness + **per-modality** model status (device, trained flag, classes) |
| `POST /analyze` | multipart: `image` + `metadata` (JSON string) + `with_gradcam` + `modality` (`eye`/`palm`) → full `TriageReport` |
| `GET /api` | Machine-readable API index |
| `GET /docs` | Swagger UI served from **locally vendored assets** (works offline) |

Metadata fields: `patient_id`, `age_months`, `sex`, `is_pregnant`, `trimester`, `symptoms[]`
(auto lower-cased/normalized), `measured_hb` (optional; **overrides the image** for triage),
`ifa_adherence`, `notes`. Validation errors → HTTP 422 with details.

### 14.2 Web UI (`app/static/`) — offline, self-contained

Inline CSS/JS (zero external dependencies — immune to CDN/network failures). Image upload
with drag-and-drop + preview, modality dropdown (eye/palm), age/sex/pregnancy/trimester/
measured-Hb inputs, symptom chips (tap-to-toggle), Grad-CAM toggle, colour-coded result
banner (🔴 High / 🟠 Medium / 🟢 Low), probability bars, quality-gate summary, inline Grad-CAM
heatmap, warnings, per-modality model status badge in the header, clinical disclaimer.

### 14.3 Terminal CLI (`scripts/analyze_cli.py`)

```bash
python -m scripts.analyze_cli --image <path> [--modality eye|palm] [--age-months N]
    [--sex male|female|other|unknown] [--pregnant] [--trimester 1-3] [--symptoms a,b]
    [--hb 9.5] [--ignore-sidecar] [--no-window] [--no-gradcam] [--open] [--json] [--out-dir DIR]
```

- Auto-reads `<image>.json` sidecar metadata; CLI flags override.
- Prints a **full colour-coded terminal report** (input info, patient, all quality metrics
  with live thresholds, vision class + confidence + probability bars, complete triage).
- Pops up a **3-panel matplotlib window** (original · preprocessed · Grad-CAM overlay),
  forced to the front (macOS process activation + Tk lift/topmost); rejected images show
  the reasons in panel 3.
- Saves `outputs/<name>_report.png`; `--open` opens it in macOS Preview as a guaranteed
  fallback.

### 14.4 Batch processor (`scripts/batch_process.py`)

Folder of images → triage CSV (`--mode api` hits a running server; `--mode local` runs
in-process). Per-image sidecar JSON supported. CSV columns: image, status, risk_class,
p_anemic, confidence, model_trained, action_code, urgency, action, hb_band, flags,
rule_fired, quality_passed, quality_reasons, processing_ms. Summary line reports
REFER_NOW / high-urgency / rejected counts.

### 14.5 Make targets

`make install · samples · test · serve · manifest · train · train-eye · train-palm ·
evaluate · cli IMG=… MODALITY=… · batch · export · notebook · clean`

---

## 15. Software Quality & Testing

**48 automated tests, all passing** (`pytest`):

| Suite | Tests | Covers |
|---|---|---|
| `test_rules_engine.py` | 15 | decision-table priorities, danger-sign override, measured-Hb override (severe/mild/normal), group-specific bands, boundary ages/Hb, pre-eclampsia flag, string inputs |
| `test_api.py` | 11 | health (per-modality), root UI, offline docs assets, analyze ok/rejected, red-flag → REFER_NOW, measured-Hb severe, bad metadata 422, palm modality, bad modality 422 |
| `test_quality_gate.py` | 6 | pass, blur reject, dark reject, no-tissue reject, Laplacian monotonicity, large-phone-photo resolution normalization |
| `test_preprocessing.py` | 5 | decode roundtrip, bad bytes, auto-crop, tensor shape/dtype, full pass keys |
| `test_imageio.py` | 5 | PNG/JPEG bytes, corrupt bytes, file read, missing file, Pillow fallback |
| `test_clinical_config.py` | 4 | config load, Hb band boundaries, pregnant/child cut-offs, danger-sign normalization |
| `test_cli.py` | 2 | subprocess smoke (headless run saves figure), missing-image error |

Additional QA: `pytest.ini` warnings policy; `py_compile` gate over all modules; every
pipeline surface (API / CLI / batch) exercised end-to-end against the **trained checkpoints**
and real dataset images during validation; ONNX export verified against onnxruntime
(output shape (1,3)).

---

## 16. Engineering Issues Found & Fixed During the Build

Real defects discovered during validation and fixed (each verified after the fix):

1. **Train/serve preprocessing mismatch (the big one).** The first palm model scored 77.5%
   during training but, through the *serving* path, predicted "Mild/Moderate" for **100% of
   images** (degenerate). Root cause: training used albumentations transforms while serving
   used gray-world + CLAHE the model had never seen. Fix: training now uses the exact serving
   preprocessing with augmentation applied before it (§5.3); both models retrained; the
   serving-path evaluator now shows genuine class separation (palm 76.5%/69.7%).
2. **Quality gate rejected 100% of the real eye dataset.** Phone-photo-tuned thresholds
   (brightness ≥ 40, clip ≤ 35%) were incompatible with dark clinical crops on black
   backgrounds. Fix: measured the true distributions across all 912/4,260 images and
   calibrated thresholds at the p5 percentile (§7.3).
3. **Palm blur threshold invalid.** Palms are smooth skin (median Laplacian variance 22.6) —
   the eye threshold (50) rejected every palm photo. Fix: modality-specific threshold (3).
4. **Biased evaluation sampling.** `evaluate` took the *first N* validation images; the
   manifest is label-sorted, so the sample was ~91% anemic and specificity was unmeasurable.
   Fix: random representative sampling with a fixed seed.
5. **HEIC/iPhone photos undecodable** (`cv2.imread` returns `None`). Fix: `app/imageio.py`
   with pillow-heif + EXIF transpose, used by every entry point.
6. **Large phone photos falsely flagged blurry.** A sharp 12-MP HEIC scored focus 5.8
   (threshold 100). Fix: resolution normalization to 1024-px longest edge before metrics
   (§7.2) — same photo then scores 204.8.
7. **DataLoader worker crash** — `cv2` module stored on the dataset instance is unpicklable
   ("cannot pickle 'module' object"). Fix: imports moved inside `__getitem__`.
8. **`torch.onnx.export` failure** — torch 2.14 requires `onnxscript`. Fix: dependency
   added; export verified with onnxruntime.
9. **Swagger blank page** — FastAPI's `/docs` loads assets from a CDN; the build machine's
   HTTPS was timing out. Fix: vendored Swagger assets locally + a fully offline custom web
   UI at `/`.
10. **CLI pop-up window hidden** — the macOS matplotlib window opened behind the terminal.
    Fix: figure pre-draw, Tk lift/topmost, macOS process activation via `osascript`, plus
    `--open` (Preview) fallback.
11. **Enum serialization in the CLI report** — `RiskClass.mild_moderate` reprs instead of
    values. Fix: `model_dump(mode="json")`.

---

## 17. Version History (Git)

| Commit | Summary |
|---|---|
| `59f4346` | Initial Anemia-Screen pipeline: FastAPI + MobileNetV3 + WHO rules engine + quality gate + Grad-CAM + batch + tests |
| `e075e4e` | Self-contained web UI at `/` and offline Swagger `/docs` (fixes blank-page CDN issue) |
| `8d24679` | Terminal CLI analyzer: full summary + pop-up 3-panel Grad-CAM window |
| `368f083` | HEIC/iPhone photo support + resolution-independent blur gate |
| `8460d82` | Palm modality + training on real datasets: manifest builder, per-modality models, train/serve-consistent preprocessing, data-calibrated quality gate, honest evaluation |

60 tracked files; datasets, manifests, checkpoints and outputs are gitignored.

---

## 18. Known Limitations & Honest Caveats

1. **Not a diagnostic device.** Screening/triage decision support only; every abnormal
   result requires confirmatory measured Hb. Not cleared for clinical use (CDSCO Medical
   Device Rules apply to diagnostic software).
2. **Validation, not held-out test.** Metrics are from the validation split (best checkpoint
   selected on it). A final untouched test set and external validation are future work.
3. **Class imbalance & small severe class.** Only 48 severe eye images; severe-class
   recall is weak (2/10 in the sampled evaluation) — the safety net is that severe
   *presentations* (danger signs, measured Hb) override the image entirely.
4. **Dataset heterogeneity.** Eye training mixes three countries/cameras; subgroup
   performance by skin tone, lighting and phone was **not** measured — the main failure
   mode per the spec, flagged for the pilot phase.
5. **Palm labels are binary proxies.** Anemic palms map to Mild/Moderate without Hb
   values, so the palm model's 3-class output is effectively anemic/non-anemic.
6. **No altitude/smoking Hb adjustment** (WHO allows both; out of scope).
7. **Overfitting signs** (train loss ≪ val loss from ~epoch 3 palm / epoch 8 eye);
   best-checkpoint selection mitigates but more data/regularization would help.
8. **Single-modality per report.** Eye and palm are parallel selectable modalities; a fused
   two-site report (eyelid + palm corroboration) is a designed but unbuilt extension.

---

## 19. Roadmap

- Fused two-site analysis (eyelid primary + palm corroboration, agreement/disagreement flag)
- Held-out test split + subgroup bias evaluation (skin tone / lighting / phone)
- WHO-2024 trimester-specific Hb cut-offs (config-ready)
- Sensitivity-threshold tuning with ROC analysis on a larger validation set
- ONNX/TFLite on-device inference + offline-first sync (ASHA field app)
- Altitude/smoking adjustments; multi-signal model (conjunctiva + nail bed + symptoms)
- RCH portal / ABDM health-ID integration; CDSCO regulatory pathway; consent flow per DPDP 2023
- Evaluation on a locally collected, consented, hemoglobinometer-confirmed cohort

---

## 20. Runbook (How to Run Everything)

```bash
cd "/Users/allen2775/Desktop/cline hackathon/anemia-screen"
conda activate anemia           # one-time env setup: conda env create -f environment.yml

# ── One-time data preparation (already done on this machine) ──
python -m scripts.make_sample_images          # synthetic demo images → data/sample/
python -m ml.build_manifest                   # parse 3 raw datasets → data/manifests/
python -m ml.train --modality eye  --epochs 12 --batch-size 32
python -m ml.train --modality palm --epochs 8  --batch-size 64
python -m ml.evaluate --modality eye  --max-images 150
python -m ml.evaluate --modality palm --max-images 200

# ── Daily use ──
python -m scripts.analyze_cli --image <photo> [--modality eye|palm] [--hb 9.5 ...]
uvicorn app.main:app --reload                 # → http://127.0.0.1:8000/ (UI) · /docs
python -m scripts.batch_process --mode local --input data/sample --out outputs/triage.csv

# ── Quality assurance ──
pytest                                        # 48 tests
python -m ml.export_onnx                      # checkpoint → ONNX (onnxruntime check)
```

---

## 21. References

- NFHS-5 (2019–21), IIPS & MoHFW — anemia prevalence figures
- Anemia Mukt Bharat operational guidelines, MoHFW (test-and-treat, IFA schedules)
- WHO: *Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity*
  (2011); WHO 2024 guideline on haemoglobin cut-offs (config-ready)
- WHO: *Guideline on daily iron and folic acid supplementation in pregnant women*
- WHO: *Ethics and governance of artificial intelligence for health* (2021)
- ICMR: *Ethical guidelines for AI in biomedical research and healthcare* (2023)
- Digital Personal Data Protection Act, 2023; CDSCO Medical Device Rules, 2017
- Eyes-defy-anemia dataset (India/Italy); CP-AnemiC dataset (Ghana);
  palmar images anemia dataset ("Detecting Anemia Based on Palm Images" family)
- Project docs: `README.md`, `docs/DATASETS.md`, `docs/CLINICAL_AND_ETHICS.md`

---

*Report generated 10 October 2026. All metrics in §10–11 are reproducible via the runbook
commands in §20 on this machine (seed 42, patient-wise splits, MPS device).*








