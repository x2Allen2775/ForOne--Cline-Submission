# Clinical Basis, Safety & Ethics

This prototype is a **screening and triage decision-support aid — NOT a diagnostic
device.** Every positive/abnormal result must be confirmed with a measured
hemoglobin (digital hemoglobinometer / lab) by a qualified health worker before any
treatment decision.

## 1. Clinical standards implemented
- **WHO 2011 haemoglobin classification** (used by Anemia Mukt Bharat), 4 bands:
  Normal (≥11.0), Mild (10.0–10.9), Moderate (7.0–9.9), Severe (<7.0) g/dL.
- **Group-specific cut-offs** for pregnant women, children 6–59 months, and a
  default adult group — all editable in `app/clinical_config.yaml`.
- **Anemia Mukt Bharat (AMB)** test-and-treat flow: the app plugs into AMB; it
  does not replace it. IFA counselling/reminders only — never prescribing.
- Thresholds are **config-driven** so WHO-2024 trimester-specific cut-offs or
  MoHFW updates can be applied without code changes.

> Verify the exact current thresholds and IFA dosing against the latest WHO and
> MoHFW AMB operational guidelines before any real deployment.

## 2. Safety design
- **Image is a screen, not a diagnosis.** The vision model never triggers
  treatment or referral on its own; it recommends a *confirmatory Hb test*.
- **Danger-sign override**: breathlessness at rest, fainting, chest pain, severe
  pallor, etc. → `REFER_NOW`, regardless of the image.
- **Measured Hb overrides the image** when provided.
- **Sensitivity-first**: thresholds should be tuned so missing an anemic patient
  (false negative) is minimised; report the resulting specificity honestly.
- **Quality gate** rejects blurry / dark / bright / glare / off-target photos so
  the model is not fed garbage.
- **Human-in-the-loop**: an ASHA/ANM/MO makes the final decision.

## 3. Explainability
Every prediction can return a **Grad-CAM heatmap** showing the region that drove
the decision, so workers and reviewers can sanity-check that the model looked at
the vascular bed, not the background.

## 4. Ethics & governance (to follow for any real use)
- **WHO Ethics & Governance of AI for Health (2021)** — six principles:
  autonomy; well-being & safety; transparency & explainability; accountability;
  inclusiveness & equity; responsive & sustainable AI.
- **ICMR Ethical Guidelines for AI in Biomedical Research & Healthcare (2023).**
- **ICMR National Ethical Guidelines (2017)** if collecting volunteer images.
- **Digital Personal Data Protection Act, 2023** — consent, purpose limitation,
  data minimisation (store a cropped eye image only; no face; encrypt in transit
  and at rest).
- **CDSCO Medical Device Rules, 2017** — diagnostic software may be regulated as a
  medical device. This prototype is **for research / screening support only and is
  NOT cleared for clinical use.**

## 5. Known limitations
- Trained on limited and possibly non-Indian-dominant data → bias risk.
- No altitude / smoking Hb adjustment (out of scope for the prototype).
- Lighting, camera and **skin-tone** variation are the main failure modes; report
  subgroup performance and collect a diverse local validation set before piloting.
- The synthetic sample images shipped in `data/sample/` are **not** real medical
  images — they only exercise the pipeline plumbing.
