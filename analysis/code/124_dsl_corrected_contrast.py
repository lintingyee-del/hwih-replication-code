# -*- coding: utf-8 -*-
"""6B step 124 — misclassification-corrected acquaintance-minus-stranger contrast.

The headline contrast (110_primary_civil_revised.py, coefficient pthA) counts cases
by the regex rel_txn flag. The gold-standard audit (App B) shows the flag is noisy:
among lending-cause audit cases, P(true acq | flag=1) = p1 ~ 0.88 and
P(true acq | flag=0) = p0 ~ 0.14, so cell-count contamination attenuates the
measured contrast by roughly (p1 - p0).

Under log-scale mixing, the measured flagged-side responses satisfy
    b_flag1 = p1*bA + (1-p1)*bS,   b_flag0 = p0*bA + (1-p0)*bS,
so the measured gap identifies (p1 - p0) * (bA - bS). Two corrections:

1. Coefficient deconvolution (primary): gap_true = pthA / kappa with
   kappa = p1 - p0. The wild/CRV t-statistics are scale-invariant, so the test of
   gap = 0 is unchanged; label-sampling uncertainty in kappa enters the corrected
   point estimate and total SE via the delta method plus a stratified bootstrap
   over the audit subsample (resampling within flag strata, matching the audit's
   stratified design).
2. Count-level matrix unmixing (robustness): convert (p1, p0) to
   sensitivity/false-positive rates via Bayes using the estimation sample's flag
   prevalence q, invert the 2x2 mixing matrix cell by cell (clipping negative
   solutions at zero), rebuild y = asinh(n*), and re-estimate the identical
   specification; also run with period-specific (pre/post) rates as a drift
   sensitivity.

NOTE a naive posterior reweighting nA* = p1*n1 + p0*n0 is NOT a correction: it is
a convex mixture of the measured series and mechanically shrinks the contrast by
roughly kappa (verified: 0.104 vs 0.182 uncorrected). Assumes nondifferential
misclassification across cells within period; audit is stratified by flag and
anchor month only.

Outputs: output/ext2124/dsl_corrected_contrast.json / .csv. No paper file touched.
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
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyfixest as pf

from _wild import wild_score_p

DATA = str(_REP_PROJECT / "data")
VAL = str(_REP_PROJECT / "output" / "validation")
OUT = str(_REP_PROJECT / "output" / "ext2124")
START, END, POST0 = "2017-01", "2019-03", "2018-09"
SUPPORT_START, SUPPORT_END = "2014-01", "2017-12"
B_LABEL = 299
rng = np.random.RandomState(42)

# ---- audit error rates ------------------------------------------------------
fr = pd.read_parquet(f"{VAL}/frame_civ_rel_txn.parquet")
rows = []
for f in glob.glob(f"{VAL}/labels_civ_rel_txn*.jsonl"):
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
audit = (
    fr.merge(pd.DataFrame(rows)[["case_no", "gold_rel_txn"]], on="case_no")
    .dropna(subset=["gold_rel_txn"])
    .assign(
        gold=lambda d: d.gold_rel_txn.astype(int),
        flag=lambda d: d.flag.astype(int),
        period=lambda d: np.where(d.ym.str[:4].astype(int) < 2018, "pre", "post"),
    )
)
audit_lend = audit[audit.cause == "民间借贷纠纷"].reset_index(drop=True)


def rates(d):
    return d[d.flag == 1].gold.mean(), d[d.flag == 0].gold.mean()


p1_lend, p0_lend = rates(audit_lend)
p1_all, p0_all = rates(audit)
period_rates = {per: rates(audit_lend[audit_lend.period == per]) for per in ("pre", "post")}
audit_ns = {
    "lending": {"n1": int((audit_lend.flag == 1).sum()), "n0": int((audit_lend.flag == 0).sum())},
    "all": {"n1": int((audit.flag == 1).sum()), "n0": int((audit.flag == 0).sum())},
    "lending_pre": {
        "n1": int(((audit_lend.flag == 1) & (audit_lend.period == "pre")).sum()),
        "n0": int(((audit_lend.flag == 0) & (audit_lend.period == "pre")).sum()),
    },
    "lending_post": {
        "n1": int(((audit_lend.flag == 1) & (audit_lend.period == "post")).sum()),
        "n0": int(((audit_lend.flag == 0) & (audit_lend.period == "post")).sum()),
    },
}

# ---- rebuild the balanced classified cell panel (as in 110), keeping n1/n0 --
schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "province", "exposure_v2_z"
]].drop_duplicates()
months = pd.DataFrame({"month": pd.period_range(START, END, freq="M").astype(str)})

case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
case["month"] = case["jmonth"].astype(str).str[:7]
lending = case[case["cause"].eq("民间借贷纠纷") & case["rel_txn"].notna()].copy()
lending["acq"] = lending["rel_txn"].astype(int)

comp_support = (
    lending[lending["month"].between(SUPPORT_START, SUPPORT_END)][
        ["prefecture_code", "province"]
    ].drop_duplicates()
    .merge(exposure[["prefecture_code", "province"]],
           on=["prefecture_code", "province"], how="inner")
)
counts = (
    lending[lending["month"].between(START, END)]
    .groupby(["prefecture_code", "province", "month", "acq"])
    .size().rename("n").reset_index()
    .pivot_table(index=["prefecture_code", "province", "month"],
                 columns="acq", values="n", fill_value=0)
    .rename(columns={0: "n0", 1: "n1"}).reset_index()
)
base = (
    comp_support.merge(months, how="cross")
    .merge(counts, on=["prefecture_code", "province", "month"], how="left")
)
base[["n0", "n1"]] = base[["n0", "n1"]].fillna(0).astype(float)
base = (
    base.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)


def build_panel(nacq, nstr, base_frame):
    a = base_frame.copy()
    a["acq"], a["n"] = 1, nacq
    s = base_frame.copy()
    s["acq"], s["n"] = 0, nstr
    d = pd.concat([a, s], ignore_index=True)
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d["month"] >= POST0).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["y"] = np.arcsinh(d["n"])
    d["prefA"] = d["prefecture_code"] + "_" + d["acq"].astype(str)
    d["monthA"] = d["month"] + "_" + d["acq"].astype(str)
    for term in ("pth", "ph", "pt"):
        d[f"{term}A"] = d[term] * d["acq"]
    return d


FORMULA = "y ~ pthA + phA + ptA + pth + ph + pt | prefA + monthA"


def fit_full(d):
    m1 = pf.feols(FORMULA, data=d, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(FORMULA, data=d, vcov={"CRV3": "prov_id"})
    return {
        "coefficient": float(m1.coef()["pthA"]),
        "se_crv1": float(m1.se()["pthA"]),
        "p_crv1": float(m1.pvalue()["pthA"]),
        "se_crv3": float(m3.se()["pthA"]),
        "p_crv3": float(m3.pvalue()["pthA"]),
        "p_wild": float(wild_score_p(FORMULA, d, "pthA")),
        "n_obs": int(m1._N),
    }


def fit_point(d):
    return float(pf.feols(FORMULA, data=d, vcov={"CRV1": "prov_id"}).coef()["pthA"])


def bayes_sf(p1, p0, q):
    """Row-conditional (p1, p0) + flag prevalence q -> sensitivity s = P(flag1|true1)
    and false-positive rate f = P(flag1|true0)."""
    pi = q * p1 + (1 - q) * p0
    s = q * p1 / pi
    f = q * (1 - p1) / (1 - pi)
    return s, f


def unmix_counts(n1, n0, s, f):
    """Invert measured (n1, n0) = M (A, S) with M = [[s, f], [1-s, 1-f]]; clip at 0."""
    det = s - f
    A = ((1 - f) * n1 - f * n0) / det
    S = (s * n0 - (1 - s) * n1) / det
    return np.clip(A, 0, None), np.clip(S, 0, None)


results = {"audit_rates": {
    "lending": {"p1": p1_lend, "p0": p0_lend},
    "all_cause": {"p1": p1_all, "p0": p0_all},
    "lending_pre": dict(zip(("p1", "p0"), period_rates["pre"])),
    "lending_post": dict(zip(("p1", "p0"), period_rates["post"])),
    "n": audit_ns,
}}

# uncorrected baseline (must reproduce 110's composition_static)
results["uncorrected"] = fit_full(build_panel(base["n1"], base["n0"], base))

# flag prevalence in the estimation window (classified lending cases)
win = base  # balanced support, window months already imposed
q = float(win["n1"].sum() / (win["n1"].sum() + win["n0"].sum()))
results["flag_prevalence_q"] = q

# ---- 1. coefficient deconvolution (primary) --------------------------------
from scipy import stats as st

g = results["uncorrected"]["coefficient"]
se_g = results["uncorrected"]["se_crv1"]
g1 = audit_lend[audit_lend.flag == 1].gold.values
g0 = audit_lend[audit_lend.flag == 0].gold.values
kappa = p1_lend - p0_lend
kdraws = np.array([
    rng.choice(g1, len(g1), replace=True).mean()
    - rng.choice(g0, len(g0), replace=True).mean()
    for _ in range(B_LABEL)
])
sd_kappa = float(kdraws.std(ddof=1))
gap_corr = g / kappa
se_total = float(np.sqrt(se_g ** 2 / kappa ** 2 + g ** 2 * sd_kappa ** 2 / kappa ** 4))
results["deconvolved"] = {
    "kappa": kappa,
    "sd_kappa": sd_kappa,
    "kappa_draws_p025": float(np.percentile(kdraws, 2.5)),
    "kappa_draws_p975": float(np.percentile(kdraws, 97.5)),
    "gap_corrected": gap_corr,
    "se_total_delta": se_total,
    "p_total_normal": float(2 * st.norm.sf(abs(gap_corr) / se_total)),
    "p_wild_scale_invariant": results["uncorrected"]["p_wild"],
    "note": "test of gap=0 identical to uncorrected wild p by scale invariance",
}
kappa_all = p1_all - p0_all
results["deconvolved_allcause"] = {"kappa": kappa_all, "gap_corrected": g / kappa_all}

# ---- 2. count-level matrix unmixing (robustness) ---------------------------
s_hat, f_hat = bayes_sf(p1_lend, p0_lend, q)
results["bayes_rates"] = {"sensitivity": s_hat, "false_positive_rate": f_hat}
A, S = unmix_counts(base["n1"].values, base["n0"].values, s_hat, f_hat)
results["unmixed_counts"] = fit_full(build_panel(A, S, base))
results["unmixed_counts"]["clipped_share_A"] = float(
    (((1 - f_hat) * base["n1"] - f_hat * base["n0"]) / (s_hat - f_hat) < 0).mean())

# period-specific rates sensitivity (drift check), same q within period
is_post = (base["month"] >= POST0).values
Av = np.empty(len(base))
Sv = np.empty(len(base))
for per, mask in (("pre", ~is_post), ("post", is_post)):
    p1p, p0p = period_rates[per]
    sp, fp = bayes_sf(p1p, p0p, q)
    Av[mask], Sv[mask] = unmix_counts(
        base["n1"].values[mask], base["n0"].values[mask], sp, fp)
results["unmixed_periodspecific"] = fit_full(build_panel(Av, Sv, base))

# label-uncertainty bootstrap for the unmixed-count estimator
draws = []
for b in range(B_LABEL):
    p1b = rng.choice(g1, len(g1), replace=True).mean()
    p0b = rng.choice(g0, len(g0), replace=True).mean()
    if p1b - p0b < 0.2:
        continue
    sb, fb = bayes_sf(p1b, p0b, q)
    Ab, Sb = unmix_counts(base["n1"].values, base["n0"].values, sb, fb)
    draws.append(fit_point(build_panel(Ab, Sb, base)))
draws = np.array(draws)
uc = results["unmixed_counts"]
sd_label = float(draws.std(ddof=1))
se_tot_u = float(np.sqrt(uc["se_crv1"] ** 2 + sd_label ** 2))
results["unmixed_label_bootstrap"] = {
    "B_effective": int(len(draws)),
    "sd_label": sd_label,
    "coef_draws_p025": float(np.percentile(draws, 2.5)),
    "coef_draws_p975": float(np.percentile(draws, 97.5)),
    "se_total": se_tot_u,
    "p_total_normal": float(2 * st.norm.sf(abs(uc["coefficient"]) / se_tot_u)),
}

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/dsl_corrected_contrast.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=float)
flat = []
for spec in ("uncorrected", "unmixed_counts", "unmixed_periodspecific"):
    flat.append({"spec": spec, **{k: v for k, v in results[spec].items()
                                  if not isinstance(v, str)}})
pd.DataFrame(flat).to_csv(f"{OUT}/dsl_corrected_contrast.csv", index=False)

for spec in flat:
    print(f"{spec['spec']:>24}: b={spec['coefficient']:.4f} "
          f"se1={spec['se_crv1']:.4f} pw={spec['p_wild']:.4f} "
          f"p3={spec['p_crv3']:.4f}")
d = results["deconvolved"]
print(f"deconvolved: gap={d['gap_corrected']:.4f} kappa={d['kappa']:.4f} "
      f"(sd {d['sd_kappa']:.4f}) se_total={d['se_total_delta']:.4f} "
      f"p_total={d['p_total_normal']:.4f}")
print(f"q={q:.4f} s={s_hat:.4f} f={f_hat:.4f} "
      f"clipped_A_share={results['unmixed_counts']['clipped_share_A']:.4f}")
lb = results["unmixed_label_bootstrap"]
print(f"unmixed label sd={lb['sd_label']:.4f} se_total={lb['se_total']:.4f} "
      f"p_total={lb['p_total_normal']:.4f} "
      f"draw CI [{lb['coef_draws_p025']:.4f}, {lb['coef_draws_p975']:.4f}]")
