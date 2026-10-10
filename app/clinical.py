"""
Clinical configuration loader.

Loads ``app/clinical_config.yaml`` so all WHO/MoHFW thresholds (Hb bands, image
risk cut-offs, danger signs, follow-up windows) are editable **without touching
code** — exactly as required by the Anemia Loop spec §4 ("make every threshold
config-driven so it can be updated without code changes").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from app.schemas import Group, HbBand

CONFIG_PATH = Path(__file__).resolve().parent / "clinical_config.yaml"


@dataclass(frozen=True)
class HbCutoffs:
    normal_min: float
    mild_min: float
    moderate_min: float

    def band(self, hb: float) -> HbBand:
        """Classify a measured Hb (g/dL) into a WHO 4-level band."""
        if hb < self.moderate_min:
            return HbBand.severe
        if hb < self.mild_min:
            return HbBand.moderate
        if hb < self.normal_min:
            return HbBand.mild
        return HbBand.normal


@dataclass(frozen=True)
class ClinicalConfig:
    standard: str
    hb_bands: dict[str, HbCutoffs]
    high_threshold: float
    low_threshold: float
    danger_signs: frozenset[str]
    preeclampsia_swelling: frozenset[str]
    preeclampsia_neuro: frozenset[str]
    recheck_weeks_min: int
    recheck_weeks_max: int
    ifa_missed_days_alert: int
    symptom_escalation_count: int

    def cutoffs_for(self, group: Group) -> HbCutoffs:
        key = group.value if group.value in self.hb_bands else "default"
        return self.hb_bands[key]


def _norm_set(items) -> frozenset[str]:
    return frozenset(str(s).strip().lower() for s in (items or []))


@lru_cache
def get_clinical_config() -> ClinicalConfig:
    with open(CONFIG_PATH, "r") as fh:
        raw = yaml.safe_load(fh)

    hb_bands = {
        key: HbCutoffs(
            normal_min=float(val["normal_min"]),
            mild_min=float(val["mild_min"]),
            moderate_min=float(val["moderate_min"]),
        )
        for key, val in raw["hb_bands"].items()
    }
    pre = raw.get("preeclampsia_signs", {})
    return ClinicalConfig(
        standard=raw.get("standard", "WHO-2011"),
        hb_bands=hb_bands,
        high_threshold=float(raw["image_risk"]["high_threshold"]),
        low_threshold=float(raw["image_risk"]["low_threshold"]),
        danger_signs=_norm_set(raw.get("danger_signs")),
        preeclampsia_swelling=_norm_set(pre.get("swelling")),
        preeclampsia_neuro=_norm_set(pre.get("neuro")),
        recheck_weeks_min=int(raw["followup"]["recheck_weeks_min"]),
        recheck_weeks_max=int(raw["followup"]["recheck_weeks_max"]),
        ifa_missed_days_alert=int(raw["followup"]["ifa_missed_days_alert"]),
        symptom_escalation_count=int(raw.get("symptom_escalation_count", 2)),
    )
