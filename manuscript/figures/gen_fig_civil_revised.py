# -*- coding: utf-8 -*-
"""Render the revised clean-window civil event-study figures.

Inputs are fixed-name CSV outputs from analysis/code/110_primary_civil_revised.py.
Each figure is written first with a timestamp and then copied to its fixed
manuscript filename.  Both vector PDF and 300-DPI PNG are produced.
"""

# Replication-package paths
from pathlib import Path as _ReplicationPath
import os as _ReplicationOS
_REP_PROJECT = _ReplicationPath(__file__).resolve().parents[1]
_REP_PACKAGE = _REP_PROJECT.parent
_REP_RESTRICTED = _REP_PACKAGE / 'restricted_data'
_REP_JUDGMENTS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_JUDGMENT_ARCHIVE', _REP_RESTRICTED / 'judgment_archive'))
_REP_CASE_ARCHIVE = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CASE_LEVEL_ARCHIVE', _REP_RESTRICTED / 'case_level_archive.parquet'))
_REP_MORTALITY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CDC_SOURCE_ROOT', _REP_RESTRICTED / 'mortality_volumes'))
_REP_REGISTRY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_REGISTRY_ROOT', _REP_RESTRICTED / 'firm_registry'))
_REP_BAIDU = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_BAIDU_ROOT', _REP_RESTRICTED / 'baidu'))
_REP_INTERVIEWS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_INTERVIEWS_ROOT', _REP_RESTRICTED / 'interviews'))
import os
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = str(_REP_PACKAGE)
RESULTS = f"{ROOT}/analysis/output/ext2124"
OUTPUT_FIGURES = f"{ROOT}/analysis/output/figures"
PAPER_FIGURES = f"{ROOT}/manuscript/figures"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSIONED_OUTPUTS = os.environ.get("HWIH_REPLICATION", "0") != "1"

os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(PAPER_FIGURES, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.35",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": "0.35",
    "ytick.color": "0.35",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "0.90",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK, WHISK, SHADE, GUIDE = "0.10", "0.50", "0.955", "0.55"
BIN_LABELS = ["[-20,-13]", "[-12,-7]", "[-6,-1]", "[0,6]"]


def with_reference(frame):
    rows = []
    by_start = {int(row["bin_start"]): row for _, row in frame.iterrows()}
    for start, end in [(-20, -13), (-12, -7), (-6, -1), (0, 6)]:
        if start == -6:
            rows.append({"coefficient": 0.0, "std_error_crv1": 0.0, "reference": True})
        else:
            row = by_start[start]
            rows.append({
                "coefficient": float(row["coefficient"]),
                "std_error_crv1": float(row["std_error_crv1"]),
                "reference": False,
            })
    return pd.DataFrame(rows)


def style_ticklabels(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("0.15")


def render(frame, stem, ylabel, ylim):
    data = with_reference(frame)
    x = np.arange(4, dtype=float)
    estimate = data["coefficient"].to_numpy(float)
    se = data["std_error_crv1"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(5.8, 3.15))
    ax.axvspan(2.5, 3.5, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=GUIDE, lw=0.7, zorder=1)
    ax.axvline(2.5, color=GUIDE, lw=0.7, ls=(0, (4, 3)), zorder=1)

    nonref = ~data["reference"].to_numpy(bool)
    ax.vlines(
        x[nonref], estimate[nonref] - 1.96 * se[nonref], estimate[nonref] + 1.96 * se[nonref],
        color=WHISK, lw=0.8, zorder=3,
    )
    ax.vlines(
        x[nonref], estimate[nonref] - 1.645 * se[nonref], estimate[nonref] + 1.645 * se[nonref],
        color=WHISK, lw=2.0, zorder=3,
    )
    ax.plot(
        x[nonref], estimate[nonref], "o", color=INK, ms=5.0,
        mec="white", mew=0.6, zorder=4,
    )
    ax.plot(
        [2], [0], marker="s", ms=5.2, mfc="white", mec=INK, mew=1.0, zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Calendar months relative to September 2018")
    ax.set_ylabel(ylabel)
    ax.text(0.02, 0.97, "Pre", transform=ax.transAxes, va="top", color="0.40", fontsize=8)
    ax.text(0.88, 0.97, "Post", transform=ax.transAxes, va="top", color="0.40", fontsize=8)
    style_ticklabels(ax)

    fixed_pdf = os.path.join(PAPER_FIGURES, f"{stem}.pdf")
    fixed_png = os.path.join(PAPER_FIGURES, f"{stem}.png")
    fig.savefig(fixed_pdf, pad_inches=0.03)
    fig.savefig(fixed_png, dpi=300, pad_inches=0.03)
    plt.close(fig)
    versioned = []
    if VERSIONED_OUTPUTS:
        timestamped_pdf = os.path.join(PAPER_FIGURES, f"{stem}_{STAMP}.pdf")
        timestamped_png = os.path.join(PAPER_FIGURES, f"{stem}_{STAMP}.png")
        shutil.copyfile(fixed_pdf, timestamped_pdf)
        shutil.copyfile(fixed_png, timestamped_png)
        versioned = [timestamped_pdf, timestamped_png]
    shutil.copyfile(fixed_pdf, os.path.join(OUTPUT_FIGURES, f"{stem}.pdf"))
    shutil.copyfile(fixed_png, os.path.join(OUTPUT_FIGURES, f"{stem}.png"))
    return [*versioned, fixed_pdf, fixed_png]


composition = pd.read_csv(f"{RESULTS}/composition_eventstudy_revised.csv")
flow = pd.read_csv(f"{RESULTS}/clean_flow_eventstudy_revised.csv")

outputs = []
outputs.extend(render(
    composition,
    "fig_es_composition",
    "Acquaintance minus stranger\nasinh(cases) per SD",
    (-0.19, 0.31),
))
outputs.extend(render(
    flow,
    "fig_es_civil_clean",
    "Relational-cause asinh(cases) per SD",
    (-0.23, 0.32),
))

print("generated figures:")
for path in outputs:
    print(path)
