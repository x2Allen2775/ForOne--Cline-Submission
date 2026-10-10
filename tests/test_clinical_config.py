"""Tests for the config-driven clinical thresholds (WHO Hb bands, etc.)."""
from app.clinical import get_clinical_config
from app.schemas import Group, HbBand


def test_config_loads():
    cfg = get_clinical_config()
    assert cfg.standard
    assert 0.0 < cfg.low_threshold < cfg.high_threshold < 1.0
    assert cfg.danger_signs  # non-empty
    assert "default" in cfg.hb_bands


def test_hb_band_boundaries_default():
    cut = get_clinical_config().cutoffs_for(Group.default)
    assert cut.band(6.9) is HbBand.severe
    assert cut.band(7.0) is HbBand.moderate
    assert cut.band(9.9) is HbBand.moderate
    assert cut.band(10.0) is HbBand.mild
    assert cut.band(10.9) is HbBand.mild
    assert cut.band(11.0) is HbBand.normal
    assert cut.band(13.5) is HbBand.normal


def test_pregnant_and_child_cutoffs_present():
    cfg = get_clinical_config()
    assert cfg.cutoffs_for(Group.pregnant).band(10.5) is HbBand.mild
    assert cfg.cutoffs_for(Group.child_6_59m).band(6.5) is HbBand.severe


def test_danger_signs_are_lowercased():
    cfg = get_clinical_config()
    assert all(s == s.lower() for s in cfg.danger_signs)
