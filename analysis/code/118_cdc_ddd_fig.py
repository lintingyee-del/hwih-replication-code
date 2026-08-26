# -*- coding: utf-8 -*-
"""Figure: triple-difference event study on the death registry.

Panel (a) plots the year-by-year post x male x homicide coefficients from
117_cdc_ddd.py, reference 2017, with the six region-by-residence strata drawn
individually behind the pooled path. The strata are shown because there are only
six of them: clustered standard errors are not usable at that count, so the
figure displays the dispersion the inference actually rests on rather than
implying an asymptotic interval. The plotted interval is heteroskedasticity
robust and does not account for stratum-level dependence; inference for this
margin is the 2^6 stratum sign test.

Panel (b) retains the raw indexed series from 87_cdc_homicide_fig.py so the
reader can see the underlying data.
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
import sys

import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt

sys.path.insert(0, str(_REP_PACKAGE))
import hz_figstyle as hz

hz.apply()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "output", "cdc_homicide")
YEARS = list(range(2014, 2022))
REF = 2017

d = pd.read_csv(os.path.join(OUT, "cdc_homicide_panel.csv"), encoding="utf-8-sig")
nat = d[(d.region == "全国") & (d.urbrur == "城乡合计")]
a = d[(d.region != "全国") & (d.urbrur != "城乡合计") & (d.sex != "合计")].copy()
a["stratum"] = a["region"] + "_" + a["urbrur"]
a["male"] = (a["sex"] == "男性").astype(int)

# ---- pooled triple-difference event study ---------------------------------
CAUSES = [("homicide_15_59_n", "homicide", 1), ("suicide_n", "suicide", 0),
          ("traffic_acc_n", "traffic", 0)]
rows = []
for col, name, focal in CAUSES:
    t = a[["year", "stratum", "male", "sex", "pop_implied", col]].copy()
    t = t.rename(columns={col: "deaths"})
    t["cause"], t["focal"] = name, focal
    rows.append(t)
s = pd.concat(rows, ignore_index=True)
s = s[s["deaths"] > 0].copy()
s["y"] = np.log(s["deaths"] / s["pop_implied"])
s["cell"] = s["stratum"] + "_" + s["sex"] + "_" + s["cause"]
s["stratum_year"] = s["stratum"] + "_" + s["year"].astype(str)
s["sex_year"] = s["sex"] + "_" + s["year"].astype(str)
s["cause_year"] = s["cause"] + "_" + s["year"].astype(str)
s["sex_cause"] = s["sex"] + "_" + s["cause"]
terms = []
for yr in YEARS:
    if yr == REF:
        continue
    s[f"T_{yr}"] = ((s["year"] == yr) * s["male"] * s["focal"]).astype(float)
    terms.append(f"T_{yr}")
m = pf.feols("y ~ " + " + ".join(terms)
             + " | cell + stratum_year + sex_year + cause_year + sex_cause",
             data=s, vcov="hetero")
est = np.array([0.0 if y == REF else float(m.coef()[f"T_{y}"]) for y in YEARS])
se = np.array([0.0 if y == REF else float(m.se()[f"T_{y}"]) for y in YEARS])

# ---- per-stratum profiles --------------------------------------------------
def strat_profile(g):
    """(male-female) homicide minus (male-female) comparator, in logs, by year."""
    piv = g.pivot_table(index="year", columns="sex",
                        values=["homicide_15_59_n", "suicide_n",
                                "traffic_acc_n"], aggfunc="sum")
    out = []
    for y in YEARS:
        try:
            hm = np.log(piv.loc[y, ("homicide_15_59_n", "男性")]
                        / piv.loc[y, ("homicide_15_59_n", "女性")])
            cm = np.log((piv.loc[y, ("suicide_n", "男性")]
                         + piv.loc[y, ("traffic_acc_n", "男性")])
                        / (piv.loc[y, ("suicide_n", "女性")]
                           + piv.loc[y, ("traffic_acc_n", "女性")]))
            out.append(hm - cm)
        except (KeyError, ZeroDivisionError, ValueError):
            out.append(np.nan)
    v = np.array(out, float)
    return v - v[YEARS.index(REF)]


profiles = {st: strat_profile(g) for st, g in a.groupby("stratum")}

# ---- draw ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=hz.DOUBLE)

ax = axes[0]
ax.axvspan(2017.5, 2021.4, color=hz.SHADE, lw=0, zorder=0)
ax.axhline(0, color=hz.GUIDE, lw=0.7, zorder=1)
ax.axvline(2017.5, color=hz.GUIDE, lw=0.7, ls=(0, (4, 3)), zorder=1)
for st, v in profiles.items():
    ax.plot(YEARS, v, "-", color="0.78", lw=0.7, zorder=2)
mask = np.array(YEARS) != REF
ax.vlines(np.array(YEARS)[mask], (est - 1.96 * se)[mask],
          (est + 1.96 * se)[mask], color=hz.WHISK, lw=0.8, zorder=3)
ax.vlines(np.array(YEARS)[mask], (est - 1.645 * se)[mask],
          (est + 1.645 * se)[mask], color=hz.WHISK, lw=2.0, zorder=3)
ax.plot(np.array(YEARS)[mask], est[mask], "o", color=hz.INK, ms=4.6,
        mec="white", mew=0.6, zorder=4)
ax.plot([REF], [0], marker="s", ms=5.0, mfc="white", mec=hz.INK, mew=1.0,
        zorder=5)
ax.set_xlim(2013.6, 2021.6)
ax.set_xticks(YEARS)
ax.set_ylabel("Male $\\times$ homicide, log points")
ax.text(0.03, 1.01, "(a) Triple-difference profile", transform=ax.transAxes,
        fontsize=9, color="0.15", va="bottom")
hz.style_ticklabels(ax)

ax = axes[1]
ax.axvspan(2017.5, 2020.5, color=hz.SHADE, lw=0, zorder=0)
ax.axvline(2017.5, color=hz.GUIDE, lw=0.7, ls=(0, (4, 3)))
ax.axhline(100, color=hz.GUIDE, lw=0.6)


def series(sex, col):
    ss = nat[nat.sex == sex].set_index("year")[col]
    return np.array([ss.get(y, np.nan) for y in YEARS], dtype=float)


def indexed(v):
    return v / v[YEARS.index(REF)] * 100.0


for v, ls, lab in [(indexed(series("男性", "homicide_15_59_n")), "-",
                    "Male 15–59 homicide"),
                   (indexed(series("女性", "homicide_n")), (0, (5, 2)),
                    "Female homicide"),
                   (indexed(series("合计", "suicide_n")), (0, (1, 1.6)),
                    "Suicide"),
                   (indexed(series("合计", "traffic_acc_n")),
                    (0, (3, 1.5, 1, 1.5)), "Road traffic")]:
    ax.plot(YEARS, v, ls=ls, color=hz.INK, lw=1.1)
    ax.plot(YEARS, v, "o", color=hz.INK, ms=2.4)
    ax.annotate(lab, (YEARS[-1] + 0.12, v[-1]), fontsize=7.5, color="0.15",
                va="center")
ax.plot(REF, 100, "s", mfc="white", mec=hz.INK, ms=4.5, mew=0.8, zorder=5)
ax.set_xlim(2013.6, 2023.6)
ax.set_xticks(YEARS)
ax.set_ylabel("Deaths, 2017 = 100")
ax.text(0.03, 1.01, "(b) Deaths indexed to 2017", transform=ax.transAxes,
        fontsize=9, color="0.15", va="bottom")
hz.style_ticklabels(ax)

hz.save(fig, os.path.join(OUT, "fig_cdc_ddd.pdf"))
fig2 = plt.figure()
print("wrote", os.path.join(OUT, "fig_cdc_ddd.pdf"))
for y, e, s_ in zip(YEARS, est, se):
    print(f"  {y}: {e:+.4f} (robust se {s_:.4f})")
