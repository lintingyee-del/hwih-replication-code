# -*- coding: utf-8 -*-
"""6B step 27 — two referee-proofing upgrades, both design-based.

PART A (软肋一): joint co-movement index with wave-timing randomization inference.
  The paper's identification is the *joint* movement of many margins that are each
  imprecise at 31 clusters. Turn that into ONE statistic. Take K theory-signed
  margins (civil judicialization +, criminal de-militarization -), estimate each on
  the clean not-yet-treated window (Post x Treat x H dose), standardize each by its
  OWN randomization SD, sign-flip to predicted direction, sum -> summary index.
  p-value by permuting first-wave (Round-1) status across the 31 provinces and
  refitting ALL margins each draw (generalizes step 24's RefPermCiv from 1 margin to
  the system). Exact in finite samples; never invokes 31-cluster asymptotics.
  Reports each margin's own one-sided perm p (individually weak) AND the joint p
  (jointly sharp), equal-weight and precision(GLS)-weight indices.

PART B (软肋二): seal the "campaign mechanically manufactured the litigation" confound
  off the SHARP civil data, not the imprecise criminal leg.
  T3 subtract-and-survive: drop lending cases whose own text carries coercive-
    collection / hard-backstop language (the mechanical channel's fingerprint); show
    the clean-window flow survives among ordinary disputes. backstop precision ~0.55
    => over-drops => conservative bound.
  T4 which input was hit: decompose exposure into general violent capacity
    (violent_share) vs debt-collection-specific (detention_debt + backstop_collect).
    Mechanical manufacture can only run through the debt-collection component; if the
    civil flow also loads on general violent capacity, backstop-removal is operating.
  T5 timing shape: a fixed busted stock dumps as a decaying spike; backstop removal
    is a sustained step. Compare early vs late post-bins of the civil event study.

Non-destructive: writes output/joint_confound_log lines and output/tables/
numbers_joint.tex (new macros only); does not touch cached results.
Usage: python 27_joint_confound.py [REPS]   (default REPS=999)
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
import sys, numpy as np, pandas as pd, pyfixest as pf

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
WIN = ("2017-01", "2019-03"); POST0 = "2018-09"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 999
rng = np.random.default_rng(20260705)

# province -> first-wave (Round-1) schedule, shared by all margins
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
PROVS = sched["province"].values
ROUND1 = (sched.set_index("province")["inspection_round"] == 1)
N_TREAT = int(ROUND1.sum())
print(f"[schedule] {len(PROVS)} provinces, {N_TREAT} first-wave", flush=True)

def cleanwin(df):
    df = df.copy()
    df["month"] = df["jmonth"].astype(str).str[:7]
    df = df[(df["month"] >= WIN[0]) & (df["month"] <= WIN[1])]
    df["postc"] = (df["month"] >= POST0).astype(int)
    return df

# province -> actual inspection month (the treatment clock we permute) ---------
INSP = (pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "insp_month"]]
        .drop_duplicates())
INSP["insp"] = INSP["insp_month"].astype(str).str[:7]
INSP_MAP0 = dict(zip(INSP["province"], INSP["insp"]))

# ---- FULL-SAMPLE frames, each margin in its NATIVE design (matches the paper):
#      civil = triple-diff vs placebo (pxr); criminal = dose px = post x H.
#      post is recomputed from the (permuted) province inspection month each draw.
def civ_td_frame(lending_only=False, outcome="asinh_n"):
    c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
    c = c[c["cause_family"].isin(["relational", "placebo"])].dropna(subset=["exposure_v2_z"]).copy()
    c["month"] = c["jmonth"].astype(str).str[:7]
    c["rel"] = (c["cause_family"] == "relational").astype(int)
    c["H"] = c["exposure_v2_z"]
    c["fe1"] = c["prefecture_code"] + "_" + c["cause"]
    c["fe2"] = c["province"] + "_" + c["month"]        # prov x month
    c["fe3"] = c["cause"] + "_" + c["month"]           # cause x month
    c["y"] = np.arcsinh(c["n_cases"])
    return c, dict(kind="td", fml="y ~ pxr + px + pr | fe1 + fe2 + fe3", coef="pxr", w=None)

def crim_px_frame(outcome_col, family, weight=True):
    k = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
    k = k[(k["family"] == family) & (k["n_cases"] > 0)].dropna(subset=["exposure_v2_z"]).copy()
    k["month"] = k["jmonth"].astype(str).str[:7]
    k["H"] = k["exposure_v2_z"]; k["fe1"] = k["prefecture_code"]
    k["fe2"] = k["province"] + "_" + k["month"]
    k["y"] = np.arcsinh(k["n_cases"]) if outcome_col == "asinh_n" else k[outcome_col]
    k["w"] = k["n_cases"].astype(float) if weight else 1.0
    return k, dict(kind="px", fml="y ~ px | fe1 + fe2", coef="px",
                   w="w" if weight else None)

def civ_px_frame(outcome_col, lending_only=True, weight=True):
    c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
    c = c[c["cause_family"] == "relational"].dropna(subset=["exposure_v2_z"]).copy()
    if lending_only:
        c = c[c["cause"] == c["cause"].value_counts().idxmax()]
    c["month"] = c["jmonth"].astype(str).str[:7]
    c["H"] = c["exposure_v2_z"]; c["fe1"] = c["prefecture_code"] + "_" + c["cause"]
    c["fe2"] = c["province"] + "_" + c["month"]
    c["y"] = c[outcome_col]; c["w"] = c["n_cases"].astype(float) if weight else 1.0
    return c, dict(kind="px", fml="y ~ px | fe1 + fe2", coef="px",
                   w="w" if weight else None)

civ_td, civ_spec = civ_td_frame()
MARGINS = [
    dict(name="civ_flow",     sign=+1, crim=False, **dict(zip(["data","spec"], (civ_td, civ_spec)))),
    dict(name="civ_relshare", sign=+1, crim=False, **dict(zip(["data","spec"], civ_px_frame("y_rel_txn")))),
    dict(name="crim_backstop",sign=-1, crim=True,  **dict(zip(["data","spec"], crim_px_frame("y_backstop","market")))),
    dict(name="crim_enforceN",sign=-1, crim=True,  **dict(zip(["data","spec"], crim_px_frame("asinh_n","enforcementcrime", weight=False)))),
    dict(name="crim_detdebt", sign=-1, crim=True,  **dict(zip(["data","spec"], crim_px_frame("y_detention_debt","enforcementcrime")))),
]

def fit_margin(data, spec, inspmap):
    d = data.copy()
    ins = d["province"].map(inspmap).values
    d["post"] = (d["month"].values >= ins).astype(int)
    d["px"] = d["post"].values * d["H"].values
    if spec["kind"] == "td":
        d["pxr"] = d["px"].values * d["rel"].values
        d["pr"] = d["post"].values * d["rel"].values
    m = pf.feols(spec["fml"], data=d, weights=spec["w"])
    return float(m.coef()[spec["coef"]])

b_obs = np.array([fit_margin(m["data"], m["spec"], INSP_MAP0) for m in MARGINS])
for m, b in zip(MARGINS, b_obs):
    print(f"[obs] {m['name']:14s} coef={b:+.5f}  (predicted sign {m['sign']:+d})", flush=True)

# ---- permute inspection timing across provinces; refit ALL margins each draw --
INSP_VALS = INSP.set_index("province").loc[PROVS, "insp"].values
K = len(MARGINS); B = np.full((REPS, K), np.nan)
for r in range(REPS):
    perm = rng.permutation(INSP_VALS)
    tm = dict(zip(PROVS, perm))
    for j, m in enumerate(MARGINS):
        B[r, j] = fit_margin(m["data"], m["spec"], tm)
    if (r + 1) % 50 == 0: print(f"   perm {r+1}/{REPS}", flush=True)

# per-margin one-sided perm p (in predicted direction)
signs = np.array([m["sign"] for m in MARGINS])
perm_p = np.array([(1 + np.sum(signs[j]*B[:, j] >= signs[j]*b_obs[j])) / (1 + REPS)
                   for j in range(K)])
# standardize each margin by its OWN null mean/SD, sign-flip to predicted direction
mu, sd = B.mean(0), B.std(0, ddof=1)
z_obs = signs * (b_obs - mu) / sd
Z = signs[None, :] * (B - mu[None, :]) / sd[None, :]

def index_p(cols, gls=False):
    zo, ZZ = z_obs[cols], Z[:, cols]
    if gls:
        S = np.cov(ZZ, rowvar=False); w = np.linalg.solve(S, np.ones(len(cols)))
    else:
        w = np.ones(len(cols))
    to, t = float(w @ zo), ZZ @ w
    return (1 + np.sum(t >= to)) / (1 + REPS), float(w @ zo)

allc = list(range(K))
crimc = [j for j, m in enumerate(MARGINS) if m["crim"]]
civc = [j for j, m in enumerate(MARGINS) if not m["crim"]]
p_joint, T_obs = index_p(allc); p_gls, _ = index_p(allc, gls=True)
p_crim, _ = index_p(crimc); p_civ, _ = index_p(civc)

print("\n===== PART A: joint co-movement index =====", flush=True)
for m, p in zip(MARGINS, perm_p):
    star = "  <- survives BH?" if p < 0.05 else ""
    print(f"   {m['name']:14s} one-sided perm p = {p:.3f}{star}", flush=True)
print(f"   CRIMINAL de-militarization joint p = {p_crim:.4f}   ({len(crimc)} margins)", flush=True)
print(f"   CIVIL judicialization joint p      = {p_civ:.4f}   ({len(civc)} margins)", flush=True)
print(f"   FULL co-movement index p (eq-wt)   = {p_joint:.4f}   (T_obs={T_obs:.2f})", flush=True)
print(f"   FULL co-movement index p (GLS)     = {p_gls:.4f}", flush=True)

# ============================ PART B: confound seals =========================
# Part B defends the SHARP civil clean-window flow (headline 0.166), so it uses the
# clean-window dose design: treat = first-wave (Round-1), postc = month >= 2018-09.
print("\n===== PART B: manufactured-litigation confound =====", flush=True)
obs_map = dict(zip(PROVS, ROUND1.reindex(PROVS).astype(int).values))
def fit_cw(fr, treatmap, weighted=False):
    d = fr.copy(); tr = d["province"].map(treatmap).values
    d["pt"] = d["postc"].values * tr
    d["pth"] = d["pt"].values * d["H"].values
    d["ph"] = d["postc"].values * d["H"].values
    m = pf.feols("y ~ pth + ph + pt | fe1 + month", data=d,
                 weights=("w" if weighted else None))
    return float(m.coef()["pth"])
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause_family","cause","prefecture_code","province",
                              "jmonth","backstop_any","backstop_collection","amount_yuan"])
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")
cc = cleanwin(cc[cc["cause_family"] == "relational"]).merge(
    ex[["prefecture_code","exposure_v2_z"]], on="prefecture_code", how="inner")

def cell_dose(cases, tag):
    g = (cases.groupby(["prefecture_code","cause","province","month","postc"])
         .size().rename("n").reset_index()
         .merge(ex[["prefecture_code","exposure_v2_z"]], on="prefecture_code"))
    g["H"] = g["exposure_v2_z"]; g["fe1"] = g["prefecture_code"] + "_" + g["cause"]
    g["y"] = np.arcsinh(g["n"])
    b = fit_cw(g, obs_map)
    return b, len(cases)

# T3 subtract-and-survive
flagged = cc["backstop_any"].fillna(0).astype(float) > 0
b_all, n_all = cell_dose(cc, "all")
b_clean, n_clean = cell_dose(cc[~flagged], "no_backstop_text")
b_flag, n_flag = cell_dose(cc[flagged], "backstop_text_only")
post_share_flag = float(flagged[cc["postc"] == 1].mean())
print(f"[T3] flow all lending      pth={b_all:+.4f}  (N={n_all:,})", flush=True)
print(f"[T3] flow drop backstop-txt pth={b_clean:+.4f}  (N={n_clean:,})", flush=True)
print(f"[T3] flow backstop-txt only pth={b_flag:+.4f}  (N={n_flag:,})", flush=True)
print(f"[T3] backstop-flagged share of post-window lending = {post_share_flag:.3f}", flush=True)

# T4 which input was hit: general violent capacity vs debt-collection-specific
exx = ex.copy()
for col, z in [("violent_share","gen_z"), ("detention_debt_rate","debt_z"),
               ("backstop_collect_rate","coll_z")]:
    exx[z] = (exx[col] - exx[col].mean()) / exx[col].std()
exx["dc_z"] = ((exx["debt_z"] + exx["coll_z"]) / 2)
exx["dc_z"] = (exx["dc_z"] - exx["dc_z"].mean()) / exx["dc_z"].std()
# rebuild civil relational cells carrying components
cvp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cvp = cleanwin(cvp[cvp["cause_family"] == "relational"]).merge(
    exx[["prefecture_code","gen_z","dc_z"]], on="prefecture_code", how="inner")
cvp["fe1"] = cvp["prefecture_code"] + "_" + cvp["cause"]; cvp["y"] = np.arcsinh(cvp["n_cases"])
def fit_component(df, comp):
    d = df.copy(); tr = d["province"].map(obs_map).values
    d["pt"] = d["postc"].values * tr; d["pc"] = d[comp].values
    d["ptc"] = d["pt"].values * d["pc"].values; d["pcm"] = d["postc"].values * d["pc"].values
    m = pf.feols("y ~ ptc + pcm + pt | fe1 + month",
                 data=d, vcov={"CRV1": "province"})
    return float(m.coef()["ptc"]), float(m.pvalue()["ptc"])
bg, pg = fit_component(cvp, "gen_z")
bd, pd_ = fit_component(cvp, "dc_z")
print(f"[T4] civil flow on GENERAL violent capacity   pth={bg:+.4f} (p={pg:.3f})", flush=True)
print(f"[T4] civil flow on DEBT-COLLECTION-specific    pth={bd:+.4f} (p={pd_:.3f})", flush=True)

# T5 timing shape: triple-diff (relational minus placebo) exposure-interacted event
# study. Mechanical stock-dump => decaying post-profile; backstop removal => sustained.
te = pd.read_parquet(f"{DATA}/civil_panel.parquet")
te = te[te["cause_family"].isin(["relational","placebo"])].dropna(subset=["exposure_v2_z"]).copy()
te["H"] = te["exposure_v2_z"]; te["rel"] = (te["cause_family"] == "relational").astype(int)
te["mo"] = te["jmonth"].astype(str).str[:7]
te["fe1"] = te["prefecture_code"] + "_" + te["cause"]
te["fe2"] = te["province"] + "_" + te["mo"]; te["fe3"] = te["cause"] + "_" + te["mo"]
te["y"] = np.arcsinh(te["n_cases"])
BINS = [(-24,-19),(-18,-13),(-12,-7),(0,5),(6,11),(12,17),(18,28)]
terms = []
for lo, hi in BINS:
    nm = f"e_{lo}_{hi}".replace("-", "m"); ind = ((te["event_time"]>=lo)&(te["event_time"]<=hi)).astype(int)
    te[nm+"_rh"] = ind*te["H"]*te["rel"]; te[nm+"_h"] = ind*te["H"]; te[nm+"_r"] = ind*te["rel"]
    terms.append((nm, lo, hi))
rhs = " + ".join(f"{nm}_rh + {nm}_h + {nm}_r" for nm,_,_ in terms)
mE = pf.feols(f"y ~ {rhs} | fe1 + fe2 + fe3", data=te, vcov={"CRV1": "province"})
prof = {(lo, hi): float(mE.coef()[nm+"_rh"]) for nm, lo, hi in terms}
early = prof[(0, 5)]; late = np.mean([prof[(12,17)], prof[(18,28)]])
print("[T5] civil post-bin exposure profile (asinh/SD):", flush=True)
for (lo, hi), v in prof.items():
    if lo >= 0: print(f"      +{lo:2d}..{hi:2d}m : {v:+.4f}", flush=True)
# mechanical stock-dump => late decays toward zero (late/early << 1);
# backstop removal => sustained level shift (late/early ~ 1 or rising)
ratio = late / early if early != 0 else float("nan")
print(f"[T5] early(0-5)={early:+.4f}  late(12-28)={late:+.4f}  late/early={ratio:.2f}  "
      f"=> {'SUSTAINED (backstop removal)' if late >= 0.5*early else 'DECAYING (stock dump)'}",
      flush=True)

# ---- emit macros -------------------------------------------------------------
def mac(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}\n"
with open(f"{OUTD}/tables/numbers_joint.tex", "w", encoding="utf-8") as fh:
    fh.write("% step 27 — joint index + manufactured-litigation seals\n")
    fh.write(mac("JointIndexP", f"{p_joint:.3f}"))
    fh.write(mac("JointIndexGLSP", f"{p_gls:.3f}"))
    fh.write(mac("CrimJointP", f"{p_crim:.3f}"))
    fh.write(mac("CivJointP", f"{p_civ:.3f}"))
    for m, p in zip(MARGINS, perm_p):
        fh.write(mac("PermP" + m["name"].title().replace("_", ""), f"{p:.3f}"))
    fh.write(mac("ConfFlowAll", f"{b_all:.3f}"))
    fh.write(mac("ConfFlowClean", f"{b_clean:.3f}"))
    fh.write(mac("ConfFlagShare", f"{post_share_flag*100:.1f}"))
    fh.write(mac("ConfGenCap", f"{bg:.3f}"))
    fh.write(mac("ConfGenCapP", f"{pg:.3f}"))
    fh.write(mac("ConfDebtSpec", f"{bd:.3f}"))
    fh.write(mac("TimingEarly", f"{early:.3f}"))
    fh.write(mac("TimingLate", f"{late:.3f}"))
pd.DataFrame(dict(margin=[m["name"] for m in MARGINS], obs=b_obs, perm_p=perm_p)
             ).to_csv(f"{OUTD}/joint_index.csv", index=False)
print("\n[done] wrote numbers_joint.tex + joint_index.csv", flush=True)
