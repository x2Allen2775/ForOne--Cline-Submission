# Datasets & Training Data Layout

Three public datasets are supported **out of the box** (auto-detected in the
project root by `ml/build_manifest.py`):

| Dataset (folder) | Modality | Labels | Notes |
|---|---|---|---|
| `dataset anemia/` (Eyes-defy-anemia, India + Italy) | eye | Real Hb (g/dL) from `India.xlsx` / `Italy.xlsx` | per-subject folders; the original `.jpg` is the model input |
| `CP-AnemiC .../` (Ghana) | eye | Real Hb from `Anemia_Data_Collection_Sheet.xlsx` (`IMAGE_ID`, `HB_LEVEL`) | ships as a `.rar` — extract with `unar` first |
| `Palm/` | palm | binary (Anemic / Non-Anemic in the filename) | palm/nail-bed photos |

## One-command workflow

```bash
python -m ml.build_manifest     # writes data/manifests/eye.csv + palm.csv
python -m ml.train --modality eye  --epochs 12   # → ml/checkpoints/anemia_eye_mobilenetv3.pt
python -m ml.train --modality palm --epochs 8    # → ml/checkpoints/anemia_palm_mobilenetv3.pt
python -m ml.evaluate --modality palm            # honest serving-path evaluation
```

### Manifest schema (what `build_manifest` writes)

```csv
image_path,hb,label,patient_id,source,modality
/Users/.../India/12/20200223_185602.jpg,9.4,Mild/Moderate,eyesdefy_india_12,eyesdefy_india,eye
/Users/.../Palm/AnemicP-012.png,,Mild/Moderate,palm_ghana_AnemicP-012,palm_ghana,palm
```

* `hb` may be empty (binary datasets) — `label` is then used directly.
* `label` uses the WHO 3-class mapping: Severe (<7) · Mild/Moderate (7–10.9) · Normal (≥11).
* Palm "anemic" maps to `Mild/Moderate` (a conservative proxy; no Hb value available).

## Legacy layout (still supported by `ml/train.py --data-dir`)

```
data/raw/
├── labels.csv      # columns: image_path, hb OR label, [patient_id]
└── images/
```

## Anti-leakage
Splits are **patient-wise**: every image from a given `patient_id` goes entirely
into train **or** validation, never both. Palm images use a pseudo-id from the
file stem (duplicates of the same palm share the stem prefix before " (n)").

## Extracting CP-AnemiC (one-time)

```bash
brew install unar   # or: pip install rarfile patool
unar -o . 'CP-AnemiC dataset.rar'
python -m ml.build_manifest   # re-run to pick it up
```
