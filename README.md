# 🩸 Anemia-Screen — Dual-Modality (Eyelid & Palm) AI Anemia Screening & Triage Pipeline

A local-first, **free** (no paid APIs, 100% offline-capable) clinical decision-support pipeline that screens for anemia from photographs of the **palpebral conjunctiva** (inner lower eyelid — primary factor, 70% weight) and the **palm** (palmar pallor — secondary factor, 30% weight), combines computer vision with **Clinical NLP symptom parsing**, and produces an **actionable, WHO/MoHFW-aligned triage recommendation**. Built for the scale of India's anemia burden (NFHS-5: ~52% of pregnant women and ~67% of young children).

> **Status:** Fully functional multimodal screening pipeline powered by **two real trained MobileNetV3 models** (trained on 5,172 real clinical images across India, Italy, Ghana, and palmar datasets), **Clinical NLP** symptom intake with negation handling, **spatial Grad-CAM** explainability, crash-free **native macOS file pickers**, **55/55 automated tests passing**, and mandatory clinical safety disclaimers.

---

## 🧠 What it does (Dual-Modality & Clinical NLP Architecture)

```
Eyelid Image + Palm Image + Patient Clinical Factors & Free-Text Symptoms
   │
   ├── [1] Clinical NLP Intake (app/clinical_nlp.py)
   │       → Lexicon normalization across 18 symptom domains
   │       → Negation analysis ("denies breathlessness", "no chest pain", "asymptomatic")
   │       → WHO/MoHFW danger signs detection (fainting, chest pain, palpitations, severe pallor…)
   │       → Automatic entity extraction: Pregnancy status, Trimester, Measured Hb (g/dL)
   │       → Pre-eclampsia risk detection (swelling + headache/blurred vision in pregnancy)
   │
   ├── [2] Quality Gate (app/pipeline/quality_gate.py)
   │       → Modality-calibrated blur gate (Laplacian variance: 50 for Eye, 3 for Palm)
   │       → Brightness window, exposure clipping, specular glare & conjunctiva tissue checks
   │       → Immediate rejection with RETAKE_PHOTO recommendation if unusable
   │
   ├── [3] Preprocessing (app/pipeline/preprocessing.py)
   │       → Eyelid: HSV-red conjunctiva auto-crop with defensive dimension safety guards
   │       → Palm: Center crop 85% with dimension safety guards
   │       → Both: Gray-world white balance + CLAHE local contrast → 224² resize → ImageNet normalization
   │
   ├── [4] MobileNetV3 Inference (app/pipeline/inference.py)
   │       → Per-modality timm MobileNetV3-large-100 classifiers → 3-class risk + P(anemic)
   │
   ├── [5] Spatial Grad-CAM Explainability (app/pipeline/gradcam.py)
   │       → Pre-pooling conv layer (model.blocks[-1]) targeting 224×224 spatial feature activations
   │       → Dynamic contrast heatmaps (JET colormap) overlaid on normalized tissue
   │       → Exports standalone heatmaps (eyelid_heatmap.png, palm_heatmap.png)
   │
   └── [6] Multimodal WHO/MoHFW Rules Synthesis (app/pipeline/rules_engine.py)
           → Danger-sign override (SAFETY FIRST: immediate REFER_NOW)
           → Measured-Hb override (laboratory confirmatory blood test priority)
           → Dual-site synthesis: Eyelid (70% primary weight) + Palm (30% secondary weight)
           → Cross-site concordance / discordance auditing & safety escalation
           → Quality failure fallbacks (eyelid fails → palm fallback with audit flag)
   │
OUTPUT ARTIFACTS:
   • Master Multimodal Dashboard (outputs/multimodal_triage_dashboard.png, 2-row × 3-col panel)
   • Isolated 3-Panel Figures (outputs/eyelid_3panel.png, outputs/palm_3panel.png)
   • Standalone Heatmaps (outputs/eyelid_heatmap.png, outputs/palm_heatmap.png)
   • Structured Clinical JSON (outputs/multimodal_triage_report.json) & CSV Dashboards
   • Color-coded Terminal Summary + MANDATORY CLINICAL DISCLAIMER
```

### Clinical classes (WHO thresholds)
| Class | Hemoglobin | Clinical Interpretation |
|---|---|---|
| **Normal** | Hb ≥ 11.0 g/dL | No anemia detected; routine preventive supplementation |
| **Mild/Moderate** | 7.0 – 10.9 g/dL | Mild to moderate anemia; IFA protocol & diet counselling |
| **Severe** | Hb < 7.0 g/dL | Severe anemia; urgent facility referral per AMB |

### Safety-first triage (WHO/MoHFW Rules Engine)
Implements the **Anemia Loop** decision table. Rules are evaluated **most-urgent-first**, strictly enforcing the medical principle:
**"An image is a screening tool, NOT a diagnosis"** — a vision score alone never prescribes medication or triggers transfusion; it triggers a *confirmatory laboratory hemoglobin test* (`CONFIRM_48H`). Only a *measured* Hb < 7 g/dL or a reported danger sign produces an immediate `REFER_NOW`.

| Priority | Clinical Presentation | Recommended Action | Urgency |
|---|---|---|---|
| 1 | WHO Danger Sign (breathlessness, fainting, chest pain, palpitations, severe pallor…) | `REFER_NOW` (same-day facility evaluation) | High |
| 2 | **Measured** Hb < 7 g/dL (severe anemia) | `REFER_NOW` (same-day referral per AMB) | High |
| 3 | **Measured** Hb 7.0–10.9 g/dL (mild/moderate) | `TREAT_AND_RECHECK` (IFA per MoHFW; re-check 4–8 weeks) | Medium |
| 4 | **Measured** Hb ≥ 11.0 g/dL (normal) | `ROUTINE` (confirmatory test overrides image) | Low |
| 5 | High image risk (P(anemic) ≥ 0.60) | `CONFIRM_48H` (blood test within 48 hours) | High |
| 6 | Medium risk (P ≥ 0.30) or ≥ 2 symptoms | `CONFIRM_1WEEK` (blood test within 1 week) | Medium |
| 7 | Low risk, no symptoms | `ROUTINE` (preventive IFA, re-screen next visit) | Low |
| — | Quality gate rejection | `RETAKE_PHOTO` (re-capture with guidance) | None |

### Key Clinical & Engineering Safeguards:
- **Eyelid Primary 70% / Palm Secondary 30% Weighting:** Conjunctival pallor is prioritized as the primary screening biomarker due to direct microvascular visualization with zero overlying melanin. Palmar pallor provides corroborating peripheral assessment.
- **Site Concordance Auditing:** Detects agreement (`concordant`) or discrepancies (`discordant`) between eyelid and palm models, appending `discordant_screening_sites` and clinical explanatory rationale.
- **Clinical NLP Negation Handling:** Disregards negated symptoms ("denies breathlessness", "no chest pain") to prevent false red-flag escalations.
- **Group-Specific WHO Cut-Offs:** Tailored 4-level bands for pregnant women, young children (6–59 months), and general adults defined in `app/clinical_config.yaml`.
- **Pre-eclampsia Flag:** Flags pregnant beneficiaries with peripheral edema (`swelling`) combined with neurological symptoms (`headache` or `blurred vision`).

---

## 🚀 Quick Start — Terminal Screening Pipeline

The primary workflow for Anemia-Screen is the **Terminal Screening Pipeline**. It runs 100% locally and offline directly in your terminal, providing live ANSI-colored reports, clinical NLP symptom analysis, and automated matplotlib visualization dashboards.

### 1) Environment Setup (conda)
```bash
conda env create -f environment.yml        # OR: conda create -n anemia python=3.11
conda activate anemia
pip install -r requirements.txt
```

### 2) Interactive Terminal Pipeline (Primary Workflow)
Run the pipeline launcher directly from your terminal:
```bash
python run_pipeline.py
# or: python -m scripts.analyze_cli
# or: make pipeline
```

**Terminal Pipeline Workflow:**
- **Individual Selectors:** Prompts you to pick the **Eyelid Image** (primary factor) and **Palm Image** (secondary factor) individually via crash-free native dialogs.
- **Clinical Intake:** Collects patient factors: Age in years, Biological Sex, Pregnancy status, Trimester, and optional measured Hb.
- **Terminal Symptom NLP Prompt:** Prompts you to enter patient symptoms in natural language (e.g. *"patient reports severe fatigue and dizziness, denies breathlessness, no chest pain"*). The built-in Clinical NLP engine parses negations, detects WHO danger signs, and extracts clinical entities in real time.
- **Color-Coded Terminal Output:** Prints an immediate ANSI-colored clinical summary in the terminal with live confidence bars, multimodal fused risk, WHO action codes, and the mandatory medical disclaimer.
- **Visual Dashboard Window:** Automatically opens the 2-row × 3-column Master Triage Dashboard (`multimodal_triage_dashboard.png`) and saves all figures to `outputs/`.

### 3) Direct Scripted CLI Command (One-Liner)
You can run the terminal pipeline non-interactively by passing image paths and patient metadata directly via command-line arguments:
```bash
python -m scripts.analyze_cli \
  --eye data/sample/pale_pregnant.png \
  --palm "Palm/Anemic-260 (10).png" \
  --age-years 28 --sex female --pregnant --trimester 2 \
  --symptoms "severe fatigue, shortness of breath, dizziness"
```

### 4) Pure Headless / Terminal-Only Mode (`--no-window`)
For servers, remote SSH sessions, or fast terminal-only inspection without opening GUI windows:
```bash
python -m scripts.analyze_cli \
  --eye data/sample/pale_pregnant.png \
  --palm "Palm/Anemic-260 (10).png" \
  --symptoms "severe fatigue, shortness of breath" \
  --no-window
```

### 5) Single-Image Terminal Mode (Legacy Backward Compatibility)
To screen an isolated eyelid or palm image:
```bash
python -m scripts.analyze_cli --image data/sample/pale_pregnant.png --modality eye
```

### 6) Terminal Batch Processor (Directory → CSV Report)
To triage an entire folder of patient images in bulk and produce a CSV dashboard:
```bash
python -m scripts.batch_process --mode local --input data/sample --out outputs/triage.csv
```

---

## 🌐 (Optional) Alternative Serving: REST API & Web UI

For deployments requiring a browser interface or microservices integration, an offline FastAPI service is also included:

```bash
uvicorn app.main:app --reload
# → Web UI: http://127.0.0.1:8000/
# → Offline Swagger API Docs: http://127.0.0.1:8000/docs
```


---

## 🖼️ Generated Visual Artifacts

Every multimodal screening run writes clean, publication-ready visual artifacts to `outputs/`:

| Artifact | Path | Description |
|---|---|---|
| **Master Multimodal Dashboard** | `outputs/multimodal_triage_dashboard.png` | **2-row × 3-column visual dashboard**: Row 1 presents Eyelid (Original, Cropped ROI, Grad-CAM Overlay with predicted risk and confidence); Row 2 presents Palm (Original, Preprocessed, Grad-CAM Overlay with predicted risk and confidence); bottom banner displays the mandatory Clinical Disclaimer. |
| **Eyelid 3-Panel Figure** | `outputs/eyelid_3panel.png` | Isolated 3-panel screening figure for the primary conjunctival site. |
| **Palm 3-Panel Figure** | `outputs/palm_3panel.png` | Isolated 3-panel screening figure for the secondary palmar site. |
| **Standalone Eyelid Heatmap** | `outputs/eyelid_heatmap.png` | High-contrast JET activation map of the eyelid microvascular capillary bed. |
| **Standalone Palm Heatmap** | `outputs/palm_heatmap.png` | High-contrast JET activation map of the palmar cutaneous circulation. |
| **Full Clinical Report (JSON)** | `outputs/multimodal_triage_report.json` | Complete machine-readable record combining patient data, symptom NLP entities, quality audits, vision metrics, multimodal synthesis, and triage actions. |

---

## 🗂 Project structure

```
anemia-screen/
├── run_pipeline.py           # Top-level interactive runner (GUI selector + symptom intake + dashboard)
├── app/                      # Serving application & pipeline stages
│   ├── main.py               # FastAPI endpoints (/analyze, /health, /api), offline /docs, web UI at /
│   ├── static/               # Self-contained web UI (index.html, app.js) + vendored offline Swagger assets
│   ├── config.py             # Runtime configuration & environment overrides
│   ├── clinical_config.yaml  # WHO/MoHFW Hb cut-offs & triage thresholds (config-driven)
│   ├── clinical.py           # Clinical config loader & helper utilities
│   ├── clinical_nlp.py       # Symptom parser: negation detection, WHO danger signs, pregnancy/Hb extraction
│   ├── schemas.py            # Pydantic schemas (TriageReport, PatientMetadata, CLINICAL_DISCLAIMER)
│   ├── imageio.py            # HEIC/HEIF & EXIF-aware robust image reader
│   └── pipeline/
│       ├── quality_gate.py   # Modality-calibrated blur / brightness / exposure / glare / tissue checks
│       ├── preprocessing.py  # Modality-aware auto-crop + dimension guards + CLAHE & Gray-World normalization
│       ├── inference.py      # timm MobileNetV3-large-100 classifiers + demo weights fallback
│       ├── gradcam.py        # Spatial 2D Grad-CAM (targeting model.blocks[-1]) & standalone heatmaps
│       ├── rules_engine.py   # WHO clinical triage logic + dual-modality 70/30 synthesis
│       └── orchestrator.py   # Wires pipeline stages 1–6 into a complete TriageReport
├── ml/                       # Training, evaluation & model export
│   ├── build_manifest.py     # Parses raw datasets → unified data/manifests/{eye,palm}.csv
│   ├── dataset.py            # Serving-consistent dataset loader + patient-wise train/val splits
│   ├── train.py              # Per-modality MobileNetV3 training with class-weighted loss
│   ├── evaluate.py           # Honest serving-path evaluation (sensitivity & specificity)
│   └── export_onnx.py        # Exports PyTorch checkpoints to ONNX (onnxruntime verified)
├── scripts/
│   ├── ui_selector.py        # Native macOS AppleScript individual image pickers & patient intake dialog
│   ├── analyze_cli.py        # Full CLI analyzer: master 2×3 dashboard, terminal report, standalone heatmaps
│   ├── batch_process.py      # Batch folder processing → CSV triage dashboard
│   └── make_sample_images.py # Synthetic sample images with metadata sidecars
├── tests/                    # 55 automated tests (pytest)
│   ├── test_multimodal.py    # Unit tests for multimodal synthesis, NLP symptom parsing, and disclaimer
│   ├── test_rules_engine.py  # Unit tests for WHO decision table & override priorities
│   ├── test_api.py           # FastAPI integration tests
│   ├── test_quality_gate.py  # Quality assessment & resolution normalization tests
│   ├── test_preprocessing.py # Auto-crop, dimension guards, and normalization tests
│   ├── test_imageio.py       # HEIC/JPEG/PNG decoding & fallback tests
│   ├── test_clinical_config.py# Clinical thresholds & beneficiary group resolution
│   └── test_cli.py           # Subprocess smoke tests
├── docs/
│   ├── TECHNICAL_REPORT.md   # Comprehensive engineering & clinical technical report (v1.1.0)
│   ├── DATASETS.md           # Dataset sources, license notes & manifest schemas
│   └── CLINICAL_AND_ETHICS.md# Clinical safety, AMB alignment & DPDP/CDSCO regulatory guidelines
├── data/sample/              # Synthetic test images + metadata sidecars
├── ml/checkpoints/           # Trained models (anemia_eye_mobilenetv3.pt, anemia_palm_mobilenetv3.pt)
├── Makefile                  # Convenience commands (make pipeline, test, serve, train, etc.)
└── requirements.txt / environment.yml
```

---

## 🏋️ Training on real clinical datasets (reproducible)

Datasets supported (auto-detected in the project root):
- **Eyes-defy-anemia** (`dataset anemia/` — India + Italy): 202 subjects, measured laboratory Hb.
- **CP-AnemiC** (`CP-AnemiC/` — Ghana): 710 subjects, pediatric clinical conjunctiva crops on black backgrounds.
- **Palmar Anemia Images** (`Palm/`): 4,260 palmar photos with binary labels.

```bash
# 1) Build unified manifests (data/manifests/eye.csv + palm.csv)
python -m ml.build_manifest

# 2) Train each modality (separate MobileNetV3 models, patient-wise splits)
python -m ml.train --modality eye  --epochs 12 --batch-size 32
python -m ml.train --modality palm --epochs  8 --batch-size 64

# 3) Evaluate honestly through the SERVING preprocessing path
python -m ml.evaluate --modality eye  --max-images 150
python -m ml.evaluate --modality palm --max-images 200
```

Trained checkpoints reside in `ml/checkpoints/anemia_{eye,palm}_mobilenetv3.pt` and are **auto-loaded** by the pipeline on startup.

---

## ✅ Software Quality & Automated Tests

The entire pipeline is protected by **55 automated tests** covering all clinical rules, NLP parsing, computer vision pipelines, quality checks, and API endpoints:

```bash
pytest
# Result: 55 passed in ~10.5s (100% pass rate)
```

| Suite | Tests | Scope |
|---|---|---|
| `tests/test_rules_engine.py` | 15 | Decision-table hierarchy, danger signs, measured Hb overrides, beneficiary groups |
| `tests/test_multimodal.py` | 7 | NLP danger signs, negation handling, pregnancy/Hb extraction, 70/30 fusion, disclaimer |
| `tests/test_api.py` | 11 | Health checks, analyze endpoints, offline docs, 422 validations, modality routing |
| `tests/test_quality_gate.py` | 6 | Blur, brightness, exposure, glare, tissue checks, resolution normalization |
| `tests/test_preprocessing.py` | 5 | ROI auto-crop, dimension safety guards, CLAHE, Gray-World, model tensor conversion |
| `tests/test_imageio.py` | 5 | HEIC/HEIF decoding, Pillow fallbacks, corrupt file resilience |
| `tests/test_clinical_config.py` | 4 | WHO cut-offs, YAML configuration loading, beneficiary group resolution |
| `tests/test_cli.py` | 2 | Headless execution, figure exports, error handling |

---

## 🛠 How Cline was used in the build

This entire pipeline — architecture, deep learning models, training scripts, WHO clinical rules engine, Clinical NLP parser, crash-free native intake UI, and test suites — was scaffolded, iterated, and hardened with **Cline** as the AI pair-programming partner.

---

## ⚠️ Mandatory Clinical Disclaimer

```
DISCLAIMER: This screening tool is an AI-assisted risk assessment and is NOT a medical diagnosis.
Non-invasive pallor screening cannot replace laboratory blood tests.
All results must be confirmed with a standard hemoglobin test (e.g. CBC or certified hemoglobinometer)
before initiating clinical treatment.
```

