"""
Build a unified training manifest (labels.csv) from the raw datasets you dropped
into the project, across BOTH modalities:

  * EYE  (conjunctiva)  — "Eyes-defy-anemia" (India + Italy): per-subject folders
                          with a .jpg + segmentation masks; Hb in an .xlsx.
  * PALM (palm/nail)    — flat folder of PNGs with the label in the filename
                          (Anemic* vs Non-Anemic*).
  * EYE  (conjunctiva)  — "CP-AnemiC" (Ghana): optional, used if extracted.

Output: one CSV per modality under ``data/manifests/``:
    data/manifests/eye.csv    columns: image_path, hb, label, patient_id, source, modality
    data/manifests/palm.csv   columns: image_path, label, patient_id, source, modality

WHO 3-class label (from Hb, g/dL):  Severe <7 ; Mild/Moderate 7-10.9 ; Normal >=11
For PALM (binary ground truth) we map: Anemic -> Mild/Moderate (proxy), Non -> Normal.

Usage:
    python -m ml.build_manifest            # auto-detect datasets in the project root
    python -m ml.build_manifest --root .   # explicit project root
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from app.config import PROJECT_ROOT

EYE_EXTS = {".jpg", ".jpeg", ".png"}
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"


def hb_to_label(hb: float) -> str:
    if hb < 7.0:
        return "Severe"
    if hb < 11.0:
        return "Mild/Moderate"
    return "Normal"


# ── Eyes-defy-anemia (India / Italy) ─────────────────────────────────────────────
def _pick_eye_image(subject_dir: Path) -> Path | None:
    """
    Prefer the palpebral-conjunctiva crop mask region's source — but the cleanest
    model input is the ORIGINAL jpg (the raw photo). We use the original .jpg;
    the preprocessing stage handles conjunctiva cropping at inference time.
    """
    jpgs = [p for p in subject_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg"} and "_" not in p.stem[-12:]]
    if not jpgs:
        # Fallback: any jpg that is NOT a mask (_palpebral/_forniceal).
        jpgs = [p for p in subject_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg"}
                and not any(m in p.stem for m in ("_palpebral", "_forniceal"))]
    return sorted(jpgs)[0] if jpgs else None


def build_eyes_defy(country_dir: Path, source: str) -> list[dict]:
    """Parse one Eyes-defy-anemia country folder (India or Italy)."""
    rows: list[dict] = []
    xlsx = next((p for p in country_dir.glob("*.xlsx")), None)
    if xlsx is None:
        print(f"[manifest] no .xlsx in {country_dir}; skipping.")
        return rows
    df = pd.read_excel(xlsx)
    # Normalise expected columns.
    cols = {c.lower(): c for c in df.columns}
    num_col = cols.get("number")
    hb_col = cols.get("hgb") or cols.get("hb")
    if num_col is None or hb_col is None:
        print(f"[manifest] {xlsx.name}: missing Number/Hgb columns; skipping.")
        return rows

    for _, r in df.iterrows():
        try:
            num = int(r[num_col])
            hb = float(r[hb_col])
        except (ValueError, TypeError):
            continue
        subject_dir = country_dir / str(num)
        if not subject_dir.is_dir():
            continue
        img = _pick_eye_image(subject_dir)
        if img is None:
            continue
        rows.append({
            "image_path": str(img.resolve()),
            "hb": hb,
            "label": hb_to_label(hb),
            "patient_id": f"{source}_{num}",
            "source": source,
            "modality": "eye",
        })
    print(f"[manifest] {source}: {len(rows)} eye samples")
    return rows


# ── Palm dataset (binary label in filename) ──────────────────────────────────────
def build_palm(palm_dir: Path, source: str = "palm_ghana") -> list[dict]:
    rows: list[dict] = []
    for p in sorted(palm_dir.rglob("*")):
        if p.suffix.lower() not in EYE_EXTS:
            continue
        name = p.name.lower()
        if name.startswith("non-anemic") or name.startswith("non-anemic") or "non-anemic" in name:
            label = "Normal"
        elif "anemic" in name:
            label = "Mild/Moderate"  # binary-positive proxy (no Hb value available)
        else:
            continue
        # Patient id = strip trailing "(n)" duplicate marker + extension.
        stem = re.sub(r"\s*\(\d+\)\s*$", "", p.stem)
        rows.append({
            "image_path": str(p.resolve()),
            "hb": "",
            "label": label,
            "patient_id": f"{source}_{stem}",
            "source": source,
            "modality": "palm",
        })
    n_anem = sum(1 for r in rows if r["label"] != "Normal")
    print(f"[manifest] palm: {len(rows)} samples ({n_anem} anemic / {len(rows)-n_anem} normal)")
    return rows


# ── CP-AnemiC (Ghana conjunctiva) — optional, only if extracted ──────────────────
def build_cp_anemic(cp_dir: Path, source: str = "cp_anemic") -> list[dict]:
    """
    Best-effort parser for an extracted CP-AnemiC folder. The dataset ships as a
    .rar; if you extract it, we look for images + a metadata CSV/XLSX with Hb.
    If no Hb table is found, we fall back to binary labels in file/folder names.
    """
    rows: list[dict] = []
    if not cp_dir.is_dir():
        return rows

    # Try a metadata table first (CP-AnemiC ships Anemia_Data_Collection_Sheet.xlsx).
    table = next((p for p in cp_dir.rglob("*.csv")), None) or \
            next((p for p in cp_dir.rglob("*.xlsx")), None)
    hb_map: dict[str, float] = {}
    if table is not None:
        try:
            df = pd.read_csv(table) if table.suffix == ".csv" else pd.read_excel(table)
            cols = {c.lower(): c for c in df.columns}
            hb_col = (cols.get("hb_level") or cols.get("hgb") or cols.get("hb")
                      or cols.get("hemoglobin"))
            id_col = (cols.get("image_id") or cols.get("id") or cols.get("number")
                      or cols.get("image") or cols.get("filename"))
            if hb_col and id_col:
                for _, r in df.iterrows():
                    try:
                        # Key by stem (e.g. 'Image_001'), case-insensitive.
                        hb_map[str(r[id_col]).strip().lower()] = float(r[hb_col])
                    except (ValueError, TypeError):
                        continue
        except Exception as exc:
            print(f"[manifest] CP-AnemiC table parse failed ({exc}); using name labels.")

    for p in sorted(cp_dir.rglob("*")):
        if p.suffix.lower() not in EYE_EXTS:
            continue
        hb = hb_map.get(p.stem.strip().lower())
        if hb is not None:
            label = hb_to_label(hb)
        else:
            name = p.name.lower()
            if "non" in name or "normal" in name or "healthy" in name:
                label = "Normal"
            elif "anemic" in name or "anaemic" in name or "pallor" in name:
                label = "Mild/Moderate"
            else:
                continue
        rows.append({
            "image_path": str(p.resolve()),
            "hb": hb if hb is not None else "",
            "label": label,
            "patient_id": f"{source}_{p.stem}",
            "source": source,
            "modality": "eye",
        })
    if rows:
        print(f"[manifest] CP-AnemiC: {len(rows)} eye samples")
    return rows


def _find(root: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        for p in root.glob(pat):
            if p.exists():
                return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build unified training manifests.")
    ap.add_argument("--root", default=str(PROJECT_ROOT),
                    help="Project root containing the dataset folders.")
    args = ap.parse_args()
    root = Path(args.root)

    eye_rows: list[dict] = []
    palm_rows: list[dict] = []

    # Eyes-defy-anemia (India + Italy)
    eda = _find(root, "dataset anemia", "*ataset*anemia*", "Eyes-defy-anemia*")
    if eda and eda.is_dir():
        for country in ("India", "Italy"):
            cdir = eda / country
            if cdir.is_dir():
                eye_rows += build_eyes_defy(cdir, source=f"eyesdefy_{country.lower()}")
    else:
        print("[manifest] 'dataset anemia' (Eyes-defy-anemia) not found.")

    # Palm
    palm = _find(root, "Palm", "palm", "*alm*")
    if palm and palm.is_dir():
        palm_rows += build_palm(palm)
    else:
        print("[manifest] 'Palm' dataset not found.")

    # CP-AnemiC (only if a folder exists, i.e. extracted)
    cp_parent = _find(root, "CP-AnemiC*")
    if cp_parent and cp_parent.is_dir():
        # Look for an extracted subfolder (not just the .rar).
        extracted = [d for d in cp_parent.rglob("*") if d.is_dir()]
        for d in [cp_parent, *extracted]:
            got = build_cp_anemic(d)
            if got:
                eye_rows += got
                break

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    wrote = []
    if eye_rows:
        p = MANIFEST_DIR / "eye.csv"
        pd.DataFrame(eye_rows).to_csv(p, index=False)
        wrote.append(p)
    if palm_rows:
        p = MANIFEST_DIR / "palm.csv"
        pd.DataFrame(palm_rows).to_csv(p, index=False)
        wrote.append(p)

    if not wrote:
        print("[manifest] No data found. Check dataset folders exist in the project root.")
        return 1
    print("\n[manifest] wrote:")
    for p in wrote:
        print(f"   {p}")
    # Label distribution summary.
    for p in wrote:
        df = pd.read_csv(p)
        print(f"   {p.name}: {len(df)} rows · labels={df['label'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
