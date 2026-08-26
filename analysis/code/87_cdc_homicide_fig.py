# -*- coding: utf-8 -*-
"""Descriptive figure: DSP homicide mortality through the campaign window.
Panel (a): national homicide death rate by sex, 2014-2021 (2016 volume is
scanned-only; the line breaks). Panel (b): deaths indexed to 2017=100 --
adult-male homicide against female homicide, suicide, and road-traffic
deaths, showing the campaign-window acceleration concentrated in adult men.
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
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(_REP_PACKAGE))
import hz_figstyle as hz

hz.apply()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "output", "cdc_homicide")
df = pd.read_csv(os.path.join(OUT, "cdc_homicide_panel.csv"))

nat = df[(df.region == "全国") & (df.urbrur == "城乡合计")]
years = list(range(2014, 2022))


def series(sex, col):
    s = nat[nat.sex == sex].set_index("year")[col]
    return np.array([s.get(y, np.nan) for y in years], dtype=float)


fig, axes = plt.subplots(1, 2, figsize=hz.DOUBLE)

# ---- (a) homicide death rate by sex --------------------------------------
ax = axes[0]
ax.axvspan(2017.5, 2020.5, color=hz.SHADE, zorder=0)
ax.axvline(2017.5, color=hz.GUIDE, lw=0.7, ls=(0, (4, 3)))
for sex, ls, lab in [("合计", "-", "All"), ("男性", (0, (5, 2)), "Male"),
                     ("女性", (0, (1, 1.6)), "Female")]:
    y = series(sex, "homicide_rate")
    ax.plot(years, y, ls=ls, color=hz.INK, lw=1.1)
    ax.plot(years, y, "o", color=hz.INK, ms=2.4)
    ax.annotate(lab, (years[-1] + 0.12, y[-1]), fontsize=7.5, color="0.15",
                va="center")
ax.set_xlim(2013.6, 2022.3)
ax.set_xticks(years)
ax.set_ylabel("Homicide deaths per 100,000 (DSP)")
ax.text(0.03, 1.01, "(a) Homicide mortality by sex", transform=ax.transAxes,
        fontsize=9, color="0.15", va="bottom")
hz.style_ticklabels(ax)

# ---- (b) indexed to 2017 = 100 -------------------------------------------
ax = axes[1]
ax.axvspan(2017.5, 2020.5, color=hz.SHADE, zorder=0)
ax.axvline(2017.5, color=hz.GUIDE, lw=0.7, ls=(0, (4, 3)))
ax.axhline(100, color=hz.GUIDE, lw=0.6)


def indexed(vals):
    base = vals[years.index(2017)]
    return vals / base * 100.0


male1559 = indexed(series("男性", "homicide_15_59_n"))
fem = indexed(series("女性", "homicide_n"))
sui = indexed(series("合计", "suicide_n"))
tra = indexed(series("合计", "traffic_acc_n"))
for y, ls, lab in [(male1559, "-", "Male 15–59 homicide"),
                   (fem, (0, (5, 2)), "Female homicide"),
                   (sui, (0, (1, 1.6)), "Suicide"),
                   (tra, (0, (3, 1.5, 1, 1.5)), "Road traffic")]:
    ax.plot(years, y, ls=ls, color=hz.INK, lw=1.1)
    ax.plot(years, y, "o", color=hz.INK, ms=2.4)
    ax.annotate(lab, (years[-1] + 0.12, y[-1]), fontsize=7.5, color="0.15",
                va="center")
ax.plot(2017, 100, "s", mfc="white", mec=hz.INK, ms=4.5, mew=0.8, zorder=5)
ax.set_xlim(2013.6, 2023.6)
ax.set_xticks(years)
ax.set_ylabel("Deaths, 2017 = 100")
ax.text(0.03, 1.01, "(b) Deaths indexed to 2017", transform=ax.transAxes,
        fontsize=9, color="0.15", va="bottom")
hz.style_ticklabels(ax)

hz.save(fig, os.path.join(OUT, "fig_cdc_homicide.pdf"))
fig.savefig(os.path.join(OUT, "fig_cdc_homicide.png"), bbox_inches="tight",
            pad_inches=0.03, dpi=200)
print("wrote", os.path.join(OUT, "fig_cdc_homicide.pdf"))
