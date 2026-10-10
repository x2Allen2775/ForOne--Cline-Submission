"""
Individual Image & Patient Clinical Intake Selector UI for Anemia-Screen.

Features:
  1. Individual file selectors for Eyelid (primary) and Palm (secondary) images.
     Uses native macOS dialogs (100% crash-free, no Cocoa Tkinter segfaults).
  2. Patient factor inputs: Age (years), Biological Sex, Pregnancy status,
     Trimester, measured Hemoglobin (optional), and Symptoms.
  3. Interactive terminal intake fallback.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def choose_image_file(prompt: str) -> Path | None:
    """
    Safely open a file picker for image selection.
    On macOS, uses native osascript to prevent the known Tk 8.6 Cocoa
    setAllowedFileTypes NSInvalidArgumentException crash.
    """
    if platform.system() == "Darwin":
        try:
            cmd = [
                "osascript", "-e",
                f'try\n'
                f'  set f to POSIX path of (choose file with prompt "{prompt}")\n'
                f'  return f\n'
                f'on error\n'
                f'  return ""\n'
                f'end try'
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            out = res.stdout.strip()
            if out:
                p = Path(out)
                if p.is_file():
                    return p
            return None
        except Exception as exc:
            print(f"[selector] notice: native file picker fallback ({exc})")

    # Cross-platform / Tkinter fallback without semicolon filetype patterns
    try:
        from tkinter import filedialog
        f = filedialog.askopenfilename(title=prompt)
        if f:
            p = Path(f)
            if p.is_file():
                return p
    except Exception:
        pass

    return None


def select_images_and_patient_gui() -> dict[str, Any] | None:
    """
    Display a GUI dialog for individual image selection (Eyelid & Palm)
    and patient clinical factors (Age, Sex, Pregnancy, Trimester, Hb, Symptoms).

    Returns a dict with:
      {
        "eye_path": Path,
        "palm_path": Path,
        "age_years": float,
        "sex": str,
        "is_pregnant": bool,
        "trimester": int | None,
        "measured_hb": float | None,
        "symptoms_text": str,
      }
    or None if canceled.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError:
        return None

    selected_eye: Path | None = None
    selected_palm: Path | None = None
    submitted = False
    result: dict[str, Any] = {}

    root = tk.Tk()
    root.title("Anemia-Screen · Dual-Modality Intake")
    root.geometry("680x560")
    root.minsize(620, 520)

    # Bring window to front safely
    try:
        root.lift()
    except Exception:
        pass

    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    header = ttk.Label(
        main_frame,
        text="🩸 Anemia-Screen: Dual-Modality Screening Intake",
        font=("Helvetica", 15, "bold"),
    )
    header.pack(anchor=tk.W, pady=(0, 2))

    subheader = ttk.Label(
        main_frame,
        text="Select the Eyelid image (primary factor) and Palm image (secondary factor), then enter patient clinical factors.",
        font=("Helvetica", 10),
    )
    subheader.pack(anchor=tk.W, pady=(0, 14))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 1: Image Selectors (Two Separate, Individual Options)
    # ──────────────────────────────────────────────────────────────────────────
    img_group = ttk.LabelFrame(main_frame, text=" 1. Screening Images (Select Individually) ", padding=12)
    img_group.pack(fill=tk.X, pady=(0, 14))

    eye_var = tk.StringVar(value="No file selected (Required)")
    palm_var = tk.StringVar(value="No file selected (Required)")
    status_var = tk.StringVar(value="Please select both Eyelid and Palm images above.")

    def update_submit_btn():
        if selected_eye and selected_palm:
            submit_btn.config(state=tk.NORMAL)
            status_var.set("✓ Both images selected. Ready to run analysis.")
        elif selected_eye:
            submit_btn.config(state=tk.DISABLED)
            status_var.set("⚠ Eyelid selected. Please also select the Palm image.")
        elif selected_palm:
            submit_btn.config(state=tk.DISABLED)
            status_var.set("⚠ Palm selected. Please also select the Eyelid image (primary).")
        else:
            submit_btn.config(state=tk.DISABLED)
            status_var.set("Please select both Eyelid and Palm images above.")

    def on_choose_eye():
        nonlocal selected_eye
        p = choose_image_file("Select Eyelid (Inner Lower Conjunctiva) Image")
        if p and p.is_file():
            selected_eye = p
            eye_var.set(f"✓ {p.name}")
            update_submit_btn()
        # Bring intake window back to front after file chooser closes
        root.lift()

    def on_choose_palm():
        nonlocal selected_palm
        p = choose_image_file("Select Palm / Hand Image")
        if p and p.is_file():
            selected_palm = p
            palm_var.set(f"✓ {p.name}")
            update_submit_btn()
        root.lift()

    # Eye Button & Label
    eye_row = ttk.Frame(img_group)
    eye_row.pack(fill=tk.X, pady=5)
    eye_btn = ttk.Button(eye_row, text="👁️ Select Eyelid Image", width=22, command=on_choose_eye)
    eye_btn.pack(side=tk.LEFT, padx=(0, 10))
    ttk.Label(eye_row, textvariable=eye_var, font=("Helvetica", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Palm Button & Label
    palm_row = ttk.Frame(img_group)
    palm_row.pack(fill=tk.X, pady=5)
    palm_btn = ttk.Button(palm_row, text="✋ Select Palm Image", width=22, command=on_choose_palm)
    palm_btn.pack(side=tk.LEFT, padx=(0, 10))
    ttk.Label(palm_row, textvariable=palm_var, font=("Helvetica", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Section 2: Patient Factors (Age, Sex, Pregnancy, Trimester, Hb, Symptoms)
    # ──────────────────────────────────────────────────────────────────────────
    pat_group = ttk.LabelFrame(main_frame, text=" 2. Patient Clinical Factors ", padding=12)
    pat_group.pack(fill=tk.X, pady=(0, 10))

    # Row 1: Age & Sex
    row1 = ttk.Frame(pat_group)
    row1.pack(fill=tk.X, pady=3)

    ttk.Label(row1, text="Age (years):", width=14).pack(side=tk.LEFT)
    age_entry = ttk.Entry(row1, width=10)
    age_entry.insert(0, "25")
    age_entry.pack(side=tk.LEFT, padx=(0, 30))

    ttk.Label(row1, text="Biological Sex:", width=14).pack(side=tk.LEFT)
    sex_combo = ttk.Combobox(row1, values=["female", "male", "other", "unknown"], width=12, state="readonly")
    sex_combo.set("female")
    sex_combo.pack(side=tk.LEFT)

    # Row 2: Pregnancy & Trimester
    row2 = ttk.Frame(pat_group)
    row2.pack(fill=tk.X, pady=4)

    ttk.Label(row2, text="Pregnant?:", width=14).pack(side=tk.LEFT)
    preg_combo = ttk.Combobox(row2, values=["No", "Yes"], width=10, state="readonly")
    preg_combo.set("No")
    preg_combo.pack(side=tk.LEFT, padx=(0, 30))

    ttk.Label(row2, text="Trimester (if yes):", width=14).pack(side=tk.LEFT)
    tri_combo = ttk.Combobox(row2, values=["None", "1", "2", "3"], width=12, state="readonly")
    tri_combo.set("None")
    tri_combo.pack(side=tk.LEFT)

    def on_preg_change(event=None):
        if preg_combo.get() == "Yes":
            if tri_combo.get() == "None":
                tri_combo.set("2")
        else:
            tri_combo.set("None")

    preg_combo.bind("<<ComboboxSelected>>", on_preg_change)

    # Row 3: Measured Hb (Optional)
    row3 = ttk.Frame(pat_group)
    row3.pack(fill=tk.X, pady=4)

    ttk.Label(row3, text="Measured Hb (g/dL):", width=18).pack(side=tk.LEFT)
    hb_entry = ttk.Entry(row3, width=10)
    hb_entry.pack(side=tk.LEFT, padx=(0, 10))
    ttk.Label(row3, text="(Optional: leave blank if blood test not yet done)", font=("Helvetica", 8, "italic")).pack(side=tk.LEFT)

    # Row 4: Symptoms
    row4 = ttk.Frame(pat_group)
    row4.pack(fill=tk.X, pady=4)

    ttk.Label(row4, text="Symptoms / Notes:", width=18).pack(side=tk.LEFT)
    sym_entry = ttk.Entry(row4)
    sym_entry.insert(0, "severe fatigue, shortness of breath, dizziness")
    sym_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Status & Submit
    # ──────────────────────────────────────────────────────────────────────────
    status_label = ttk.Label(main_frame, textvariable=status_var, font=("Helvetica", 9, "italic"), foreground="#0066cc")
    status_label.pack(anchor=tk.W, pady=(4, 10))

    action_row = ttk.Frame(main_frame)
    action_row.pack(fill=tk.X, side=tk.BOTTOM)

    def on_submit():
        nonlocal submitted, result
        if not (selected_eye and selected_palm):
            messagebox.showwarning("Images Required", "Please select both Eyelid and Palm images before proceeding.")
            return

        try:
            age_y = float(age_entry.get().strip())
        except ValueError:
            age_y = 25.0

        sex_val = sex_combo.get().strip().lower()
        is_preg = preg_combo.get() == "Yes"

        tri_str = tri_combo.get().strip()
        trimester = int(tri_str) if tri_str in ("1", "2", "3") else (2 if is_preg else None)

        measured_hb = None
        hb_str = hb_entry.get().strip()
        if hb_str:
            try:
                measured_hb = float(hb_str)
            except ValueError:
                pass

        symptoms_str = sym_entry.get().strip()

        result = {
            "eye_path": selected_eye,
            "palm_path": selected_palm,
            "age_years": age_y,
            "sex": sex_val,
            "is_pregnant": is_preg,
            "trimester": trimester,
            "measured_hb": measured_hb,
            "symptoms_text": symptoms_str,
        }
        submitted = True
        root.destroy()

    def on_cancel():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cancel)

    submit_btn = ttk.Button(
        action_row,
        text="✓ Run Multimodal Analysis",
        command=on_submit,
        state=tk.DISABLED,
    )
    submit_btn.pack(side=tk.RIGHT, padx=4)

    cancel_btn = ttk.Button(action_row, text="Cancel", command=on_cancel)
    cancel_btn.pack(side=tk.RIGHT, padx=4)

    root.mainloop()

    return result if submitted else None


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Terminal Intake Fallback
# ──────────────────────────────────────────────────────────────────────────────
def prompt_patient_intake_cli(defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Terminal clinical intake prompts for Age, Sex, Pregnancy, Trimester,
    measured Hb, and Symptoms.
    """
    d = defaults or {}
    print("\n" + "═" * 63)
    print("         PATIENT CLINICAL INTAKE (Terminal)")
    print("═" * 63)

    # 1. Age
    def_age = d.get("age_years", 25.0)
    age_in = input(f"1. Patient Age in years [default: {def_age:.0f}]: ").strip()
    try:
        age_years = float(age_in) if age_in else float(def_age)
    except ValueError:
        age_years = 25.0

    # 2. Sex
    def_sex = d.get("sex", "female")
    sex_in = input(f"2. Biological Sex (female/male/other/unknown) [default: {def_sex}]: ").strip().lower()
    sex = sex_in if sex_in in ("female", "male", "other", "unknown") else def_sex

    # 3. Pregnancy
    def_preg = "yes" if d.get("is_pregnant", False) else "no"
    preg_in = input(f"3. Is the patient currently pregnant? (yes/no) [default: {def_preg}]: ").strip().lower()
    is_pregnant = preg_in in ("y", "yes", "true", "1") if preg_in else (def_preg == "yes")

    # 4. Trimester
    trimester = None
    if is_pregnant:
        def_tri = d.get("trimester", 2) or 2
        tri_in = input(f"   Which pregnancy trimester? (1, 2, or 3) [default: {def_tri}]: ").strip()
        try:
            trimester = int(tri_in) if tri_in in ("1", "2", "3") else def_tri
        except ValueError:
            trimester = 2

    # 5. Measured Hb
    def_hb = d.get("measured_hb")
    def_hb_str = str(def_hb) if def_hb is not None else "none"
    hb_in = input(f"4. Measured Hemoglobin in g/dL if known [default: {def_hb_str}]: ").strip()
    measured_hb = None
    if hb_in and hb_in.lower() not in ("none", "no", "n/a"):
        try:
            measured_hb = float(hb_in)
        except ValueError:
            pass
    elif def_hb is not None:
        measured_hb = def_hb

    # 6. Symptoms
    def_sym = d.get("symptoms_text", "")
    sym_hint = f" [default: '{def_sym}']" if def_sym else ""
    print("5. Patient Symptoms or clinical notes:")
    print("   (e.g., severe fatigue, shortness of breath, pale skin, dizziness, fainting)")
    sym_in = input(f"   Symptoms >{sym_hint}: ").strip()
    symptoms_text = sym_in if sym_in else def_sym

    return {
        "age_years": age_years,
        "sex": sex,
        "is_pregnant": is_pregnant,
        "trimester": trimester,
        "measured_hb": measured_hb,
        "symptoms_text": symptoms_text,
    }


def prompt_images_cli() -> tuple[Path | None, Path | None]:
    """Terminal file path entry for Eyelid and Palm."""
    print("\n[selector] Enter image file paths individually:")
    eye_in = input("Enter Eyelid image path (e.g. data/sample/pale_pregnant.png): ").strip()
    palm_in = input("Enter Palm image path   (e.g. Palm/Anemic-260 (10).png): ").strip()

    eye_p = Path(eye_in).expanduser() if eye_in else None
    palm_p = Path(palm_in).expanduser() if palm_in else None

    if eye_p and not eye_p.exists():
        print(f"  ✗ Eyelid file not found: {eye_p}")
        eye_p = None
    if palm_p and not palm_p.exists():
        print(f"  ✗ Palm file not found: {palm_p}")
        palm_p = None

    return eye_p, palm_p
