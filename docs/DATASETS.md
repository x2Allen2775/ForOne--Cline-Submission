# Datasets & Training Data Layout

This project ships **synthetic** sample images so the pipeline runs before any real
data is downloaded. To train a real model, download public conjunctiva-anemia
datasets and arrange them as described below.

## Recommended public datasets (free, registration may be required)
- **Eyes-defy-anemia** — conjunctiva images with hemoglobin values.
- **CP-AnemiC** — conjunctival pallor anemia dataset.

Search for these on Kaggle / academic repositories and follow their licenses.

## Required on-disk layout

```
data/raw/
├── labels.csv
└── images/
    ├── patientA_01.jpg
    ├── patientA_02.jpg
    ├── patientB_01.jpg
    └── ...
```

## `labels.csv` schema

The loader (`ml/dataset.py`) accepts **either** a hemoglobin column **or** a direct
label column, plus an optional `patient_id` for leakage-free splits.

Option A — hemoglobin (preferred; class is derived with WHO thresholds):

```csv
image_path,hb,patient_id
images/patientA_01.jpg,9.4,patientA
images/patientA_02.jpg,9.4,patientA
images/patientB_01.jpg,12.8,patientB
images/patientC_01.jpg,6.2,patientC
```

Option B — explicit label (must be exactly one of the class names):

```csv
image_path,label,patient_id
images/patientA_01.jpg,Mild/Moderate,patientA
images/patientB_01.jpg,Normal,patientB
images/patientC_01.jpg,Severe,patientC
```

### WHO class thresholds (used when `hb` is provided)
| Hb (g/dL) | Class |
|---|---|
| `>= 11.0` | Normal |
| `7.0 – 10.9` | Mild/Moderate |
| `< 7.0` | Severe |

## Anti-leakage note
Splits are **patient-wise**: every image from a given `patient_id` goes entirely
into train **or** validation, never both. If you omit `patient_id`, the file stem
is used as a fallback (not recommended — it disables leakage control).

## Train
```bash
conda activate anemia
python -m ml.train --data-dir data/raw --epochs 15 --batch-size 32
```
The best checkpoint is written to `ml/checkpoints/anemia_mobilenetv3.pt` and is
auto-loaded by the API/pipeline on next start.
