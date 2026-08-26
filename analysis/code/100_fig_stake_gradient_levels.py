# -*- coding: utf-8 -*-
"""Draw the clean-window stake gradient in case-count levels.

The input is the five-band stacked regression produced by
99f_stake_five_band.py.  Both panels use the same specification:
prefecture-by-band, province-by-month, and month-by-band fixed effects.

Run with ``--install`` to copy the vector PDF into the submission figures
directory after visual inspection.
"""

from __future__ import annotations

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

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/output"
FIG = OUT / "figures"
PAPER_FIG = ROOT / "manuscript/figures"
sys.path.insert(0, str(ROOT))
import hz_figstyle as hz


LABELS = ["<20", "20–50", "50–200", "200–1,000", ">1,000"]
BANDS = ["q1", "q2", "q3", "q4", "q5"]


def panel_data(data: pd.DataFrame, clock: str) -> pd.DataFrame:
    d = data[(data["clock"] == clock) & (data["transformation"] == "level")].copy()
    return d.set_index("band").loc[BANDS].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    source = OUT / "stake_five_band_coefficients.csv"
    data = pd.read_csv(source)
    judgment = panel_data(data, "judgment")
    filing = panel_data(data, "filing")

    hz.apply()
    fig, axes = plt.subplots(1, 2, figsize=hz.DOUBLE, sharey=True)
    for ax, d, tag in [
        (axes[0], judgment, "(a) Judgment month"),
        (axes[1], filing, "(b) Filing month"),
    ]:
        hz.coefpanel(
            ax,
            LABELS,
            d["coefficient"].to_numpy(),
            d["std_error_crv1"].to_numpy(),
            connect=True,
        )
        hz.tag(ax, tag)
        ax.set_ylim(-12.0, 20.0)

    axes[0].set_ylabel("Cases per prefecture and month\nper SD of exposure")
    fig.supxlabel("Claim size (thousand yuan)", fontsize=9, y=-0.025, color="0.15")
    fig.subplots_adjust(wspace=0.16)

    FIG.mkdir(parents=True, exist_ok=True)
    pdf = FIG / "fig_c4_invertedu.pdf"
    png = FIG / "fig_c4_invertedu.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    if args.install:
        PAPER_FIG.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, PAPER_FIG / pdf.name)
        print(f"installed {PAPER_FIG / pdf.name}")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
