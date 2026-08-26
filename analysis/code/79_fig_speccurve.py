# -*- coding: utf-8 -*-
"""6B step 79 — specification forest for the judicialization estimate.

All estimates already computed elsewhere in the pipeline (no new estimation):
the clean-window family, geographic designs, robustness variants, estimator
validation, and different-support benchmarks, drawn with two-tier CIs.
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
import sys, os, shutil
sys.path.insert(0, str(_REP_PACKAGE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hz_figstyle as hz

hz.apply()
FIGD = str(_REP_PROJECT / 'output' / 'figures').replace('\\', '/')
MANUSCRIPT = str(_REP_PACKAGE / "manuscript" / "figures")

# (label, est, se, group)
SPECS = [
    ("Clean window, judgment clock",        0.1565, 0.0789, "design"),
    ("Clean window, filing clock",          0.1385, 0.0596, "design"),
    ("Judgment clock, matched horizon",     0.137,  0.081,  "design"),
    ("Contiguous provinces",                0.2172, 0.0769, "geo"),
    ("Neighbour sample ($\\leq$200 km)",    0.1908, 0.1269, "geo"),
    ("Neighbour pairs (pair FE)",           0.3171, 0.1190, "geo"),
    ("Neighbour pairs, $\\leq$150 km",      0.401,  0.159,  "geo"),
    ("Neighbour pairs, $\\leq$250 km",      0.371,  0.092,  "geo"),
    ("+ Post$\\times$DFI-credit control",   0.1698, 0.0824, "robust"),
    ("Drop top-quartile DFI prefectures",   0.1438, 0.0865, "robust"),
    ("Exposure from 2014--15 only",         0.1169, 0.0796, "robust"),
    ("Exposure from 2016--17 only",         0.1680, 0.0762, "robust"),
    ("Poisson, filing clock",               0.188,  0.088,  "robust"),
    ("Poisson, neighbour pairs",            0.619,  0.123,  "robust"),
    ("Callaway--Sant'Anna dose slope",      0.162,  0.083,  "estimator"),
    ("Full staggered sample (DDD)",         0.0275, 0.0223, "atten"),
    ("Poisson, judgment clock (staggered)", 0.087,  0.100,  "atten"),
]
GROUPS = [("design", "Clean-window designs"), ("geo", "Geographic designs"),
          ("robust", "Robustness variants"), ("estimator", "Estimator validation"),
          ("atten", "Different-support benchmarks")]

rows, labels, heading_positions = [], [], []
y = 0
for gkey, gname in GROUPS:
    heading_positions.append(y)
    labels.append(gname)
    y += 1
    for lab, est, se, g in SPECS:
        if g == gkey:
            rows.append((y, est, se)); labels.append(lab); y += 1

fig, ax = plt.subplots(figsize=(6.4, 6.2))
ys = np.array([r[0] for r in rows])
est = np.array([r[1] for r in rows])
se = np.array([r[2] for r in rows])
ax.axvline(0, color=hz.GUIDE, lw=0.7, zorder=1)
ax.axvline(0.1565, color=hz.GUIDE, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.hlines(ys, est - 1.96 * se, est + 1.96 * se, color=hz.WHISK, lw=0.8, zorder=3)
ax.hlines(ys, est - 1.645 * se, est + 1.645 * se, color=hz.WHISK, lw=2.0, zorder=3)
ax.plot(est, ys, "o", color=hz.INK, ms=4.6, mec="white", mew=0.6, zorder=4)
for pos in heading_positions[1:]:
    ax.axhline(pos - 0.5, color="0.85", lw=0.5, zorder=0)
ax.set_yticks(np.arange(y))
ax.set_yticklabels(labels, fontsize=7.4)
ax.set_ylim(-0.5, y - 0.5)
ax.invert_yaxis()
ax.grid(False)
ax.set_xlabel("asinh(cases) per SD of exposure")
ax.margins(x=0.04)
hz.style_ticklabels(ax)
for pos in heading_positions:
    tick = ax.get_yticklabels()[pos]
    tick.set_fontweight("bold")
    tick.set_color("0.35")
hz.save(fig, f"{FIGD}/fig_speccurve.pdf")
shutil.copy(f"{FIGD}/fig_speccurve.pdf", f"{MANUSCRIPT}/fig_speccurve.pdf")
print("fig_speccurve.pdf written and copied")
