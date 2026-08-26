# -*- coding: utf-8 -*-
"""Influence and scale diagnostics for the stake-gradient level result.

This is a companion to 99_stake_gradient.py.  It keeps the treatment,
window, sample, and five original amount bins fixed.  The only purpose is to
determine whether the positive level-count contrast is robust to influential
clusters, upper-tail counts, fixed effects, and the filing clock.

No paper file or existing pipeline output is modified.
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

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
OUT = ROOT / "analysis/output"
REPS_PERM = 999
SEED = 314159


def load_search_module():
    path = CODE / "99_stake_gradient.py"
    spec = importlib.util.spec_from_file_location("stake_gradient", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xg = load_search_module()


def level_group_panel(bin_panel: pd.DataFrame, value_col: str = "n") -> pd.DataFrame:
    d = bin_panel.copy()
    d["z"] = d[value_col].astype(float)
    d["middle"] = d["band"].isin(xg.MIDDLE).astype(int)
    keys = [
        "prefecture_code", "province", "month", "inspection_round", "H", "treat",
        "postc", "prov_id", "pt", "pth", "ph", "middle",
    ]
    g = d.groupby(keys, as_index=False, observed=True)["z"].mean().rename(columns={"z": "y"})
    g["pref_group"] = g["prefecture_code"] + "_" + g["middle"].astype(str)
    g["month_group"] = g["month"] + "_" + g["middle"].astype(str)
    g["prov_month_group"] = g["province"] + "_" + g["month"] + "_" + g["middle"].astype(str)
    for name in ["pth", "ph", "pt"]:
        g[f"{name}_mid"] = g[name] * g["middle"]
    return g


def fit_shape(
    spec_id: str,
    panel: pd.DataFrame,
    formula: str = xg.FML_BASE,
    note: str = "",
    run_wild: bool = True,
) -> dict:
    m1 = pf.feols(formula, data=panel, vcov={"CRV1": "prov_id"})
    try:
        m3 = pf.feols(formula, data=panel, vcov={"CRV3": "prov_id"})
        se3 = float(m3.se()["pth_mid"])
        p3 = float(m3.pvalue()["pth_mid"])
    except Exception as exc:
        print(f"[CRV3 failed] {spec_id}: {exc}", flush=True)
        se3 = p3 = np.nan
    if run_wild:
        try:
            pwild = float(
                xg.wild_score_p(
                    formula,
                    panel,
                    "pth_mid",
                    cluster="prov_id",
                    reps=xg.REPS_WILD,
                    seed=SEED,
                )
            )
        except Exception as exc:
            print(f"[wild failed] {spec_id}: {exc}", flush=True)
            pwild = np.nan
    else:
        pwild = np.nan

    tail = float(m1.coef()["pth"])
    diff = float(m1.coef()["pth_mid"])
    row = {
        "spec_id": spec_id,
        "coefficient": diff,
        "std_error_crv1": float(m1.se()["pth_mid"]),
        "p_crv1": float(m1.pvalue()["pth_mid"]),
        "p_wild": pwild,
        "std_error_crv3": se3,
        "p_crv3": p3,
        "tail_response": tail,
        "middle_response": tail + diff,
        "n_obs": int(m1._N),
        "note": note,
    }
    print(
        f"{spec_id:30s} diff={diff:+.4f}; CRV1 se={row['std_error_crv1']:.4f}, "
        f"p={row['p_crv1']:.3f}; wild={pwild:.3f}; CRV3 p={p3:.3f}; "
        f"tail={tail:+.3f}; middle={tail + diff:+.3f}",
        flush=True,
    )
    return row


def cap_from_preperiod(bin_panel: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, dict]:
    d = bin_panel.copy()
    caps = (
        d.loc[d["postc"] == 0]
        .groupby("band", observed=True)["n"]
        .quantile(quantile)
        .to_dict()
    )
    d["n_capped"] = np.minimum(d["n"], d["band"].map(caps).astype(float))
    return d, caps


def leave_one_province_out(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for province in sorted(panel["province"].unique()):
        d = panel[panel["province"] != province].copy()
        m = pf.feols(xg.FML_BASE, data=d, vcov="iid")
        rows.append(
            {
                "province_omitted": province,
                "coefficient": float(m.coef()["pth_mid"]),
                "n_obs": int(m._N),
            }
        )
        print(f"LOPO omit {province}: {rows[-1]['coefficient']:+.4f}", flush=True)
    return pd.DataFrame(rows)


def wave_permutation(panel: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    treat_map = panel[["province", "treat"]].drop_duplicates().set_index("province")["treat"].to_dict()
    provinces = np.array(sorted(treat_map))
    treated_n = int(sum(treat_map.values()))

    def relabel(source: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
        d = source.copy()
        d["treat"] = d["province"].map(mapping).astype(int)
        d["pt"] = d["postc"] * d["treat"]
        d["pth"] = d["pt"] * d["H"]
        d["ph"] = d["postc"] * d["H"]
        for name in ["pth", "ph", "pt"]:
            d[f"{name}_mid"] = d[name] * d["middle"]
        return d

    observed = relabel(panel, treat_map)
    beta = float(pf.feols(xg.FML_BASE, data=observed, vcov="iid").coef()["pth_mid"])
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPS_PERM)
    for r in range(REPS_PERM):
        labels = np.zeros(len(provinces), dtype=int)
        labels[rng.choice(len(provinces), treated_n, replace=False)] = 1
        d = relabel(panel, dict(zip(provinces, labels)))
        draws[r] = float(pf.feols(xg.FML_BASE, data=d, vcov="iid").coef()["pth_mid"])
        if (r + 1) % 100 == 0:
            print(f"level wave permutations {r + 1}/{REPS_PERM}", flush=True)
    p = float((1 + np.sum(np.abs(draws) >= abs(beta))) / (REPS_PERM + 1))
    return p, pd.DataFrame({"draw": np.arange(REPS_PERM), "estimate": draws})


def per_band_level(bin_panel: pd.DataFrame, clock: str) -> pd.DataFrame:
    rows = []
    for band in xg.BANDS:
        d = bin_panel[bin_panel["band"] == band].copy()
        d["y"] = d["n"].astype(float)
        d["pref"] = d["prefecture_code"]
        formula = "y ~ pth + ph + pt | pref + month"
        m1 = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
        try:
            pwild = float(
                xg.wild_score_p(
                    formula,
                    d,
                    "pth",
                    cluster="prov_id",
                    reps=xg.REPS_WILD,
                    seed=SEED,
                )
            )
        except Exception as exc:
            print(f"[per-band wild failed] {clock} {band}: {exc}", flush=True)
            pwild = np.nan
        rows.append(
            {
                "clock": clock,
                "band": band,
                "estimate": float(m1.coef()["pth"]),
                "std_error_crv1": float(m1.se()["pth"]),
                "p_crv1": float(m1.pvalue()["pth"]),
                "p_wild": pwild,
                "n_obs": int(m1._N),
            }
        )
    return pd.DataFrame(rows)


def baseline_counts(bin_panel: pd.DataFrame, clock: str) -> pd.DataFrame:
    pre = bin_panel[bin_panel["postc"] == 0]
    return (
        pre.groupby("band", observed=True)["n"]
        .agg(mean="mean", median="median", p90=lambda s: s.quantile(0.90), p99=lambda s: s.quantile(0.99))
        .reset_index()
        .assign(clock=clock)
    )


def main() -> None:
    print("== level-count diagnostics under the frozen stake design ==", flush=True)
    pref = xg.load_prefecture_frame()
    cases = xg.read_lending_cases()
    judgment = xg.balanced_bin_panel(cases, pref, "judgment_month", "judgment")
    level = level_group_panel(judgment)
    level["clock"] = "judgment"

    results = [
        fit_shape(
            "D1_level_baseline",
            level,
            note="Same positive level-count result from the complete search",
        ),
        fit_shape(
            "D2_level_strict_fe",
            level,
            formula=xg.FML_STRICT,
            note="Prefecture by group and province by month by group fixed effects",
        ),
    ]

    for q, tag in [(0.99, "D3_level_cap_pre_p99"), (0.995, "D4_level_cap_pre_p995")]:
        capped, caps = cap_from_preperiod(judgment, q)
        capped_level = level_group_panel(capped, "n_capped")
        capped_level["clock"] = "judgment"
        results.append(
            fit_shape(
                tag,
                capped_level,
                note=f"Counts capped at band-specific pre-period p{q * 100:g}: {caps}",
            )
        )

    filing_cases = xg.filing_clock(cases)
    filing = xg.balanced_bin_panel(filing_cases, pref, "filing_month", "filing")
    filing_level = level_group_panel(filing)
    filing_level["clock"] = "filing"
    results.append(
        fit_shape(
            "D5_filing_level",
            filing_level,
            note="Same level-count contrast dated by filing month",
        )
    )

    lopo = leave_one_province_out(level)
    p_perm, perm = wave_permutation(level)
    for row in results:
        if row["spec_id"] == "D1_level_baseline":
            row["p_wave_permutation"] = p_perm
            row["lopo_min"] = float(lopo["coefficient"].min())
            row["lopo_max"] = float(lopo["coefficient"].max())
            row["lopo_positive_share"] = float((lopo["coefficient"] > 0).mean())

    band = pd.concat(
        [per_band_level(judgment, "judgment"), per_band_level(filing, "filing")],
        ignore_index=True,
    )
    base = pd.concat(
        [baseline_counts(judgment, "judgment"), baseline_counts(filing, "filing")],
        ignore_index=True,
    )

    pd.DataFrame(results).to_csv(OUT / "stake_level_diagnostics.csv", index=False)
    lopo.to_csv(OUT / "stake_level_lopo.csv", index=False)
    perm.to_csv(OUT / "stake_level_permutation.csv", index=False)
    band.to_csv(OUT / "stake_level_per_band.csv", index=False)
    base.to_csv(OUT / "stake_level_baseline_counts.csv", index=False)

    print(f"Level wave-label permutation p={p_perm:.3f}", flush=True)
    print(
        f"LOPO range [{lopo.coefficient.min():+.4f}, {lopo.coefficient.max():+.4f}], "
        f"positive in {(lopo.coefficient > 0).sum()}/{len(lopo)} omissions",
        flush=True,
    )
    print("Saved diagnostic outputs; no paper file was modified.", flush=True)


if __name__ == "__main__":
    main()
