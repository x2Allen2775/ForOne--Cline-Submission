"""
Clinical NLP Symptom Analyzer.

Parses free-text symptom descriptions entered by health workers or patients,
normalizes medical terms, recognizes WHO/MoHFW danger signs, extracts clinical
modifiers (pregnancy, trimester, measured Hb), and handles common negations
(e.g., "no breathlessness", "denies fainting").
"""
from __future__ import annotations

import re
from typing import Any

from app.clinical import get_clinical_config


# Canonical symptom dictionary mapping: canonical_name -> list of keywords/phrases
SYMPTOM_LEXICON: dict[str, list[str]] = {
    # ── Danger signs (WHO / MoHFW red flags) ──
    "breathlessness": [
        "breathlessness", "breathless", "shortness of breath", "short of breath",
        "difficulty breathing", "dyspnea", "trouble breathing", "out of breath",
        "gasping", "breathless at rest", "breathless_rest"
    ],
    "fainting": [
        "fainting", "fainted", "syncope", "passed out", "blackout", "black out",
        "loss of consciousness", "collapsing", "collapse", "swooning"
    ],
    "chest pain": [
        "chest pain", "pain in chest", "chest tightness", "angina", "chest pressure",
        "chest discomfort", "chest_pain"
    ],
    "severe pallor": [
        "severe pallor", "severe_pallor", "extremely pale", "very pale",
        "ghostly pale", "paper white", "blanched"
    ],
    "palpitations": [
        "palpitations", "rapid heartbeat", "fast heartbeat", "racing heart",
        "heart racing", "heart pounding", "fluttering heart", "irregular heartbeat"
    ],
    "confusion": [
        "confusion", "confused", "disoriented", "delirium", "altered mental state",
        "brain fog"
    ],
    "unable to walk": [
        "unable to walk", "cannot walk", "difficulty walking", "unsteady",
        "unable to stand", "prostration"
    ],

    # ── General Anemia Symptoms ──
    "fatigue": [
        "fatigue", "tired", "tiredness", "exhaustion", "exhausted", "lethargy",
        "lethargic", "low energy", "worn out", "drowsy", "drowsiness", "malaise"
    ],
    "dizziness": [
        "dizziness", "dizzy", "lightheaded", "lightheadedness", "vertigo",
        "woozy", "giddy", "feeling faint"
    ],
    "weakness": [
        "weakness", "weak", "feeling weak", "muscle weakness", "feeble",
        "lack of strength"
    ],
    "headache": [
        "headache", "head aches", "migraine", "head pain", "throbbing head"
    ],
    "pallor": [
        "pallor", "pale", "paleness", "pale skin", "pale eyes", "pale eyelid",
        "pale conjunctiva", "pale hands", "pale palms", "pale nails"
    ],
    "cold extremities": [
        "cold hands", "cold feet", "cold extremities", "cold limbs", "chills"
    ],
    "brittle nails": [
        "brittle nails", "spoon nails", "koilonychia", "cracked nails"
    ],
    "pica": [
        "pica", "craving ice", "eating ice", "ice craving", "craving dirt",
        "eating chalk", "clay craving"
    ],
    "poor appetite": [
        "poor appetite", "loss of appetite", "not eating", "anorexia"
    ],
    "swelling": [
        "swelling", "swollen", "swollen feet", "swollen legs", "swollen ankles",
        "edema", "oedema", "puffy face"
    ],
    "blurred vision": [
        "blurred vision", "blurry vision", "vision problems", "visual disturbance",
        "blurred_vision"
    ],
}

# Negation prefixes to avoid false matches (e.g. "no chest pain")
NEGATION_PATTERNS = [
    r"\bno\s+",
    r"\bnot\s+",
    r"\bdenies\s+",
    r"\bwithout\s+",
    r"\bnever\s+",
    r"\bfree of\s+",
    r"\bnegative for\s+",
]


def _is_negated(text: str, match_start: int) -> bool:
    """Check if a match is preceded within ~25 characters by a negation word."""
    preceding = text[max(0, match_start - 25):match_start]
    for pat in NEGATION_PATTERNS:
        if re.search(pat, preceding, re.IGNORECASE):
            return True
    return False


def analyze_symptom_text(raw_text: str | None) -> dict[str, Any]:
    """
    Extract structured clinical findings from free-form text.

    Returns:
      {
        "raw_text": str,
        "matched_symptoms": list[str],
        "danger_signs": list[str],
        "has_danger_signs": bool,
        "preeclampsia_risk": bool,
        "detected_pregnant": bool,
        "detected_trimester": int | None,
        "detected_hb": float | None,
        "clinical_urgency": "High" | "Medium" | "Low",
        "summary": str,
      }
    """
    if not raw_text or not raw_text.strip():
        return {
            "raw_text": "",
            "matched_symptoms": [],
            "danger_signs": [],
            "has_danger_signs": False,
            "preeclampsia_risk": False,
            "detected_pregnant": False,
            "detected_trimester": None,
            "detected_hb": None,
            "clinical_urgency": "Low",
            "summary": "No symptoms reported.",
        }

    cfg = get_clinical_config()
    clean_text = raw_text.lower().strip()

    # Check for blanket negation (e.g. "none", "nil", "healthy", "no symptoms")
    if clean_text in ("none", "nil", "no", "no symptoms", "asymptomatic", "healthy", "normal"):
        return {
            "raw_text": raw_text,
            "matched_symptoms": [],
            "danger_signs": [],
            "has_danger_signs": False,
            "preeclampsia_risk": False,
            "detected_pregnant": False,
            "detected_trimester": None,
            "detected_hb": None,
            "clinical_urgency": "Low",
            "summary": "Asymptomatic (patient explicitly reports no symptoms).",
        }

    matched_set: set[str] = set()

    for canonical, phrase_list in SYMPTOM_LEXICON.items():
        for phrase in phrase_list:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            for m in re.finditer(pattern, clean_text):
                if not _is_negated(clean_text, m.start()):
                    matched_set.add(canonical)
                    break

    matched_symptoms = sorted(matched_set)

    # Danger signs check
    danger_signs = [
        s for s in matched_symptoms if s in cfg.danger_signs or s in (
            "breathlessness", "fainting", "chest pain", "severe pallor",
            "palpitations", "confusion", "unable to walk"
        )
    ]

    # Pre-eclampsia signs check (swelling + headache/blurred vision)
    s_set = set(matched_symptoms)
    has_swelling = bool(s_set & {"swelling"})
    has_neuro = bool(s_set & {"headache", "blurred vision"})
    preeclampsia_signs = has_swelling and has_neuro

    # Detect pregnancy mentions
    detected_pregnant = False
    detected_trimester = None
    if re.search(r"\bpregnant\b|\bpregnancy\b|\bgestation\b", clean_text):
        detected_pregnant = True
    if re.search(r"\b(1st|first)\s+trimester\b|trimester\s*1\b|\bt1\b", clean_text):
        detected_pregnant = True
        detected_trimester = 1
    elif re.search(r"\b(2nd|second)\s+trimester\b|trimester\s*2\b|\bt2\b", clean_text):
        detected_pregnant = True
        detected_trimester = 2
    elif re.search(r"\b(3rd|third)\s+trimester\b|trimester\s*3\b|\bt3\b", clean_text):
        detected_pregnant = True
        detected_trimester = 3

    # Detect measured hemoglobin mentions (e.g., "hb 8.5", "hemoglobin is 9.2", "hb: 7.1 g/dl")
    detected_hb = None
    hb_match = re.search(
        r"\b(?:hb|hemo(?:globin)?|haemo(?:globin)?)\s*(?:is|=|:)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:g/?dl)?\b",
        clean_text,
    )
    if hb_match:
        try:
            val = float(hb_match.group(1))
            if 2.0 <= val <= 22.0:
                detected_hb = val
        except ValueError:
            pass

    # Urgency assessment
    if danger_signs:
        urgency = "High"
        summary = (
            f"ALERT: Danger sign(s) detected [{', '.join(danger_signs)}]. "
            "Requires immediate facility referral per WHO/MoHFW protocol."
        )
    elif len(matched_symptoms) >= 2 or preeclampsia_signs:
        urgency = "Medium"
        summary = (
            f"Multiple symptoms reported ({len(matched_symptoms)}): {', '.join(matched_symptoms)}. "
            "Escalates triage review within 1 week."
        )
    elif len(matched_symptoms) == 1:
        urgency = "Low"
        summary = f"Single symptom reported: {matched_symptoms[0]}."
    else:
        # User entered text but no recognized medical symptoms matched
        urgency = "Low"
        summary = f"Custom text recorded: '{clean_text[:60]}...' (no specific anemia danger signs detected)."

    return {
        "raw_text": raw_text,
        "matched_symptoms": matched_symptoms,
        "danger_signs": danger_signs,
        "has_danger_signs": bool(danger_signs),
        "preeclampsia_risk": preeclampsia_signs,
        "detected_pregnant": detected_pregnant,
        "detected_trimester": detected_trimester,
        "detected_hb": detected_hb,
        "clinical_urgency": urgency,
        "summary": summary,
    }
