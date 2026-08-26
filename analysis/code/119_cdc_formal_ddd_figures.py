# -*- coding: utf-8 -*-
"""House-style figures for the formal death-registry DDD.

Main figure:
  (a) pooled annual DDD coefficients with CRV3 t(5) intervals;
  (b) age-matched male/female death ratios for the three causes.

Appendix figure:
  the six East/Central/West by urban/rural DDD trajectories, each normalized
  to zero in 2017, with their equal-weight mean.
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

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "output" / "cdc_homicide"
sys.path.insert(0, str(WORKSPACE))
import hz_figstyle as hz

hz.apply()

YEARS = np.arange(2014, 2022)
REFERENCE = 2017
CRIT_95 = stats.t.ppf(0.975, 5)
CRIT_90 = stats.t.ppf(0.950, 5)


def event_panel(ax, event):
    """Pooled annual profile with six-cluster CRV3 intervals."""
    event = event.sort_values("year")
    years = event["year"].to_numpy(dtype=int)
    estimate = event["coefficient"].to_numpy(dtype=float)
    se = event["crv3_se"].to_numpy(dtype=float)
    reference = event["reference"].astype(str).str.lower().eq("true").to_numpy()
    plotted = ~reference

    ax.axvspan(2017.5, 2020.5, color=hz.SHADE, lw=0, zorder=0)
    ax.axhline(0, color=hz.GUIDE, lw=0.7, zorder=1)
    ax.axvline(2017.5, color=hz.GUIDE, lw=0.7,
               ls=(0, (4, 3)), zorder=1)
    ax.vlines(
        years[plotted],
        estimate[plotted] - CRIT_95 * se[plotted],
        estimate[plotted] + CRIT_95 * se[plotted],
        color=hz.WHISK, lw=0.8, zorder=3,
    )
    ax.vlines(
        years[plotted],
        estimate[plotted] - CRIT_90 * se[plotted],
        estimate[plotted] + CRIT_90 * se[plotted],
        color=hz.WHISK, lw=2.0, zorder=3,
    )
    ax.plot(years[plotted], estimate[plotted], "o", color=hz.INK,
            ms=4.6, mec="white", mew=0.6, zorder=4)
    ax.plot([REFERENCE], [0], marker="s", ms=5.0, mfc="white",
            mec=hz.INK, mew=1.0, zorder=5)
    ax.set_xlim(2013.6, 2021.4)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Triple difference, log points")
    hz.tag(ax, "(a) Annual triple difference")
    hz.style_ticklabels(ax)


def ratio_panel(ax, panel):
    """Age-15-59 male/female death ratios, indexed to 2017."""
    national = panel[
        panel["region"].eq("全国")
        & panel["urbrur"].eq("城乡合计")
        & panel["sex"].isin(["男性", "女性"])
    ].copy()
    labels = {
        "homicide": "Homicide",
        "suicide": "Suicide",
        "traffic": "Road traffic",
    }
    line_styles = {
        "homicide": "-",
        "suicide": (0, (1, 1.6)),
        "traffic": (0, (5, 2)),
    }

    indexed = {}
    for cause in ("homicide", "suicide", "traffic"):
        wide = national.pivot(
            index="year", columns="sex", values=f"{cause}_15_59_n")
        ratio = wide["男性"] / wide["女性"]
        indexed[cause] = ratio / ratio.loc[REFERENCE] * 100

    ax.axvspan(2017.5, 2020.5, color=hz.SHADE, lw=0, zorder=0)
    ax.axvline(2017.5, color=hz.GUIDE, lw=0.7,
               ls=(0, (4, 3)), zorder=1)
    ax.axhline(100, color=hz.GUIDE, lw=0.6, zorder=1)
    label_offsets = {"homicide": -1.0, "suicide": 1.7, "traffic": -1.7}
    for cause in ("homicide", "suicide", "traffic"):
        values = indexed[cause].reindex(YEARS).to_numpy(dtype=float)
        ax.plot(YEARS, values, ls=line_styles[cause], color=hz.INK,
                lw=1.1, zorder=2)
        ax.plot(YEARS, values, "o", color=hz.INK, ms=2.4, zorder=3)
        ax.annotate(
            labels[cause],
            (2021.12, values[-1] + label_offsets[cause]),
            fontsize=7.5, color="0.15", va="center",
        )
    ax.plot(REFERENCE, 100, "s", mfc="white", mec=hz.INK,
            ms=4.5, mew=0.8, zorder=5)
    ax.set_xlim(2013.6, 2023.0)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Male/female deaths, 2017 = 100")
    hz.tag(ax, "(b) Age-matched mortality ratios")
    hz.style_ticklabels(ax)


def save_main(event, panel):
    figure, axes = plt.subplots(1, 2, figsize=hz.DOUBLE)
    event_panel(axes[0], event)
    ratio_panel(axes[1], panel)
    figure.savefig(
        OUT / "fig_cdc_homicide_formal.pdf",
        bbox_inches="tight", pad_inches=0.03,
    )
    figure.savefig(
        OUT / "fig_cdc_homicide_formal.png",
        bbox_inches="tight", pad_inches=0.03, dpi=220,
    )
    plt.close(figure)


def spread_positions(values, minimum_gap=0.075):
    """Spread right-edge labels while preserving their vertical order."""
    order = np.argsort(values)
    adjusted = np.asarray(values, dtype=float).copy()
    for previous, current in zip(order[:-1], order[1:]):
        adjusted[current] = max(
            adjusted[current], adjusted[previous] + minimum_gap)
    shift = adjusted.mean() - np.asarray(values).mean()
    return adjusted - shift


def save_strata(stratum_year):
    wide = stratum_year.pivot(
        index="year", columns="stratum", values="ddd_combined")
    profiles = wide.subtract(wide.loc[REFERENCE], axis=1)
    pooled = profiles.mean(axis=1)
    name_map = {
        "东部 / 城市": "East urban",
        "东部 / 农村": "East rural",
        "中部 / 城市": "Central urban",
        "中部 / 农村": "Central rural",
        "西部 / 城市": "West urban",
        "西部 / 农村": "West rural",
    }

    figure, ax = plt.subplots(figsize=hz.SINGLE)
    ax.axvspan(2017.5, 2020.5, color=hz.SHADE, lw=0, zorder=0)
    ax.axhline(0, color=hz.GUIDE, lw=0.7, zorder=1)
    ax.axvline(2017.5, color=hz.GUIDE, lw=0.7,
               ls=(0, (4, 3)), zorder=1)

    columns = list(profiles.columns)
    styles = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (3, 1.5, 1, 1.5)),
              (0, (7, 2, 1, 2)), (0, (2, 2))]
    endpoints = []
    for column, line_style in zip(columns, styles):
        values = profiles[column].reindex(YEARS).to_numpy(dtype=float)
        ax.plot(YEARS, values, ls=line_style, color="0.72",
                lw=0.8, zorder=2)
        ax.plot(YEARS, values, "o", color="0.72", ms=1.8, zorder=2)
        endpoints.append(values[-1])
    pooled_values = pooled.reindex(YEARS).to_numpy(dtype=float)
    ax.plot(YEARS, pooled_values, "-", color=hz.INK, lw=1.25, zorder=3)
    ax.plot(YEARS, pooled_values, "o", color=hz.INK, ms=3.3, zorder=4)
    ax.plot(REFERENCE, 0, "s", mfc="white", mec=hz.INK,
            ms=4.5, mew=0.8, zorder=5)

    label_y = spread_positions(np.asarray(endpoints), minimum_gap=0.075)
    for column, y0, y1 in zip(columns, endpoints, label_y):
        ax.plot([2021.02, 2021.20], [y0, y1], color="0.65", lw=0.55,
                clip_on=False)
        ax.annotate(
            name_map[column], (2021.23, y1), fontsize=6.8,
            color="0.35", va="center", clip_on=False,
        )
    ax.annotate(
        "Equal-weight mean", (2021.23, pooled_values[-1] - 0.035),
        fontsize=7.0, color=hz.INK, va="center", clip_on=False,
    )
    ax.set_xlim(2013.6, 2023.0)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Triple difference relative to 2017, log points")
    hz.style_ticklabels(ax)
    figure.savefig(
        OUT / "fig_cdc_ddd_strata.pdf",
        bbox_inches="tight", pad_inches=0.03,
    )
    figure.savefig(
        OUT / "fig_cdc_ddd_strata.png",
        bbox_inches="tight", pad_inches=0.03, dpi=220,
    )
    plt.close(figure)


def main():
    event = pd.read_csv(
        OUT / "cdc_formal_ddd_eventstudy.csv", encoding="utf-8-sig")
    panel = pd.read_csv(
        OUT / "cdc_age1559_panel.csv", encoding="utf-8-sig")
    stratum_year = pd.read_csv(
        OUT / "cdc_formal_ddd_stratum_year.csv", encoding="utf-8-sig")
    save_main(event, panel)
    save_strata(stratum_year)
    print("wrote", OUT / "fig_cdc_homicide_formal.pdf")
    print("wrote", OUT / "fig_cdc_ddd_strata.pdf")


if __name__ == "__main__":
    main()
