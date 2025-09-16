#!/usr/bin/env python3

# ================= GUI BACKEND (pop-up window) =================
import os, importlib, matplotlib
os.environ.pop("MPLBACKEND", None)                 # avoid inline
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")    # helps Qt on Wayland

def _has(mod):
    try: importlib.import_module(mod); return True
    except Exception: return False

# Prefer Qt -> then Tk. If neither exists, raise a helpful error.
if _has("PySide6") or _has("PyQt6") or _has("PyQt5") or _has("PySide2"):
    matplotlib.use("QtAgg", force=True)
elif _has("tkinter"):
    matplotlib.use("TkAgg", force=True)
else:
    raise RuntimeError(
        "No GUI backend found. Install one of: "
        "`conda install -c conda-forge pyqt` (recommended) or `conda install tk`."
    )
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

print("Matplotlib backend:", matplotlib.get_backend())

# -------- reset styles (no empty color cycles!) ----------
mpl.rcdefaults()
plt.style.use("default")
plt.close('all')

# ---- Your palette (5s MUST be ORANGE) ----
COLORS = {
    5:  '#ff7f0e',   # orange
    6:  '#1f77b4',   # blue
    7:  '#2ca02c',   # green
    8:  'olive',
    9:  'navy',
    10: 'purple',
}

# === Load TR-ENF summary (real ROC curves + AUCs) ===
summary = pd.read_csv("./results/summary_tr.csv")

# Convert whitespace-separated strings -> numpy arrays
def str_to_array(s: str) -> np.ndarray:
    s = str(s).strip().strip("[]")
    if not s:
        return np.array([], dtype=float)
    return np.array([float(x) for x in s.split()])

summary["fpr"] = summary["fpr"].apply(str_to_array)
summary["tpr"] = summary["tpr"].apply(str_to_array)

# ---------------------------
# PCHIP smoothing helper
# ---------------------------
def smooth_curve(x, y, num=1000):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux, idx = np.unique(x, return_index=True)
    uy = y[idx]
    if ux.size < 2:
        return ux, uy
    f = PchipInterpolator(ux, uy)
    xs = np.linspace(ux.min(), ux.max(), num)
    ys = f(xs)
    return xs, ys

# ---------------------------
# ROC (TR-ENF only) — 5 s
# ---------------------------
row5 = summary.loc[summary["duration_s"] == 5]
if not row5.empty:
    row5 = row5.iloc[0]
    fpr5, tpr5, auc5 = row5["fpr"], row5["tpr"], float(row5["auc"])
    fpr5_s, tpr5_s = smooth_curve(fpr5, tpr5, num=1000)

    plt.figure(figsize=(8,8))
    # FORCE 5s = ORANGE
    plt.plot(fpr5_s*100, tpr5_s*100, lw=3.5, color=COLORS[5],
             label=f"TR-ENF, 5s (AUC={auc5*100:.1f}%)")
    plt.plot([0,100],[0,100],'k--', lw=1.5)

    plt.xlabel("False Positive Rate (%)", fontsize=16)
    plt.ylabel("True Positive Rate (%)", fontsize=16)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.axis("square"); plt.xlim([0,100]); plt.ylim([0,100])
    plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("roc_curve_trenf_5s.png", dpi=300)
else:
    print("Warning: No 5s row found in summary_tr.csv")

# ---------------------------
# DET (TR-ENF only) — 5..10 s
# ---------------------------
plt.figure(figsize=(8,8))

for d in [5,6,7,8,9,10]:
    row = summary.loc[summary["duration_s"] == d]
    if row.empty:
        print(f"Warning: No {d}s row found in summary_tr.csv")
        continue
    row = row.iloc[0]
    fpr, tpr = row["fpr"], row["tpr"]
    if len(fpr) == 0 or len(tpr) == 0:
        print(f"Warning: Empty fpr/tpr for {d}s")
        continue

    fnr = 1 - tpr

    # Smooth in % domain
    fpr_s, fnr_s = smooth_curve(fpr*100, fnr*100, num=1000)

    # --- Raw EER (unsmoothed arrays) ---
    diff_raw = np.abs(fpr - fnr)
    idx_raw = int(np.argmin(diff_raw))
    eer_raw = float((fpr[idx_raw] + fnr[idx_raw]) / 2) * 100.0

    # --- Smoothed EER (continuous) ---
    diff_s = np.abs(fpr_s - fnr_s)
    idx_s = int(np.argmin(diff_s))
    eer_s = float((fpr_s[idx_s] + fnr_s[idx_s]) / 2.0)
    eer_fpr = float(fpr_s[idx_s]); eer_fnr = float(fnr_s[idx_s])

    print(f"[{d}s] Raw EER = {eer_raw:.2f}%, Smoothed EER = {eer_s:.2f}%")

    col = COLORS[int(d)]  # FORCE your color
    plt.plot(fpr_s, fnr_s, lw=2.5, color=col,
             label=f"TR-ENF, {d}s (EER={eer_s:.2f}%)")
    plt.scatter([eer_fpr], [eer_fnr], s=40, color=col, zorder=5)

# Diagonal EER line
plt.plot([0,100],[0,100],'k--', lw=1.5, label="EER line")
#plt.xlabel("False Positive Rate (%)", fontsize=16)
#plt.ylabel("True Positive Rate (%)", fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.axis("square"); plt.xlim([0,100]); plt.ylim([0,100])
#plt.legend(loc="upper right", fontsize=12, frameon=True, fancybox=True)
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("det_curves_trenf_5_10s.png", dpi=300)
plt.show()  # opens a native pop-up window
