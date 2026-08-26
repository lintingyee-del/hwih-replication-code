# -*- coding: utf-8 -*-
"""Exhibit: the market for private debt collection through the campaign.
Panel (a): registered collection-scope firms, quarterly foundings and
deregistration approvals. Panel (b): national Baidu search demand for
collection services, quarterly."""

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
OUT = os.path.join(HERE, "output", "collection_firms")
BOUT = os.path.join(HERE, "output", "baidu_index")

fig, axes = plt.subplots(1, 2, figsize=hz.DOUBLE)

# ---- (a) registry -----------------------------------------------------------
d = pd.read_csv(os.path.join(OUT, "national_quarterly.csv"))
d["t"] = pd.PeriodIndex(d["quarter"], freq="Q").to_timestamp() \
    + pd.Timedelta(days=45)
d = d[(d["quarter"] >= "2014Q1") & (d["quarter"] <= "2022Q4")]
ax = axes[0]
ax.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31"),
           color=hz.SHADE, zorder=0)
ax.axvline(pd.Timestamp("2018-01-01"), color=hz.GUIDE, lw=0.7, ls=(0, (4, 3)))
ax.plot(d["t"], d["entries"], "-", color=hz.INK, lw=1.1)
ax.plot(d["t"], d["exits"], ls=(0, (5, 2)), color=hz.INK, lw=1.1)
ax.annotate("Foundings", (pd.Timestamp("2016-06-01"), 175), fontsize=7.5,
            color="0.15")
ax.annotate("Deregistrations", (pd.Timestamp("2019-01-01"), 185), fontsize=7.5,
            color="0.15")
ax.set_xlim(pd.Timestamp("2013-10-01"), pd.Timestamp("2023-06-30"))
ax.set_ylabel("Collection-scope firms per quarter")
ax.text(0.03, 1.01, "(a) Registered firms", transform=ax.transAxes,
        fontsize=9, color="0.15", va="bottom")
hz.style_ticklabels(ax)

# ---- (b) search demand ------------------------------------------------------
b = pd.read_csv(os.path.join(BOUT, "national_monthly.csv"), dtype={"ym": str})
b["q"] = pd.PeriodIndex(pd.to_datetime(b["ym"] + "-01"), freq="Q")
bq = b.groupby(["keyword", "q"], as_index=False)["mean"].sum()
bq["t"] = bq["q"].dt.to_timestamp() + pd.Timedelta(days=45)
ax = axes[1]
ax.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31"),
           color=hz.SHADE, zorder=0)
ax.axvline(pd.Timestamp("2018-01-01"), color=hz.GUIDE, lw=0.7, ls=(0, (4, 3)))
for kw, ls, lab, xpos, ypos in [
        ("讨债公司", "-", "``collection company''",
         pd.Timestamp("2020-03-01"), 9000),
        ("讨债", (0, (5, 2)), "``collect a debt''",
         pd.Timestamp("2015-01-01"), 3100)]:
    s = bq[bq["keyword"] == kw]
    ax.plot(s["t"], s["mean"], ls=ls, color=hz.INK, lw=1.1)
    ax.annotate(lab, (xpos, ypos), fontsize=7.5, color="0.15")
ax.set_xlim(pd.Timestamp("2013-10-01"), pd.Timestamp("2022-06-30"))
ax.set_ylabel("Search index, national quarterly sum")
ax.text(0.03, 1.01, "(b) Search demand", transform=ax.transAxes,
        fontsize=9, color="0.15", va="bottom")
hz.style_ticklabels(ax)

hz.save(fig, os.path.join(OUT, "fig_collection_firms.pdf"))
fig.savefig(os.path.join(OUT, "fig_collection_firms.png"),
            bbox_inches="tight", pad_inches=0.03, dpi=200)
print("wrote two-panel fig_collection_firms")
