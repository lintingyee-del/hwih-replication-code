# -*- coding: utf-8 -*-
"""6B step 70 — referee shore-up battery (compute only; NO tex writes).

P0  Cross-check: current results.csv / results_v2.csv wild p's vs the wild p's
    hardcoded in the paper (Tables C1/C3/8), to catalogue stale pre-_wild-fix
    cells.
P1  Wave-timing randomization inference for the geographic designs: permute
    first-wave treatment across the 31 provinces (count held fixed), REBUILD the
    border sample/pairs per draw, refit; 999 draws. Designs: contiguous
    provinces, prefecture-border (<=200 km), DLR border pairs (pair FE).
P2  Callaway-Sant'Anna upgrade: continuous-dose slope (per SD of H), i.e.
    per-(g,e) [slope of dy on H among treated cohort g] minus [slope among
    strictly not-yet-treated], weighted by treated count, pooled over e in
    [0,8]; quartile top-bottom contrast as a coarser check; the existing
    tercile contrast is replicated first as a validation of the machinery.
    Province block bootstrap (999) for SEs and joint lead tests.
P3  MDEs (80% power, 5% two-sided, t(30)) for the audited criminal margins of
    Table 3, expressed against pre-period baselines, with the v1-proportional
    benchmark.
P4  Canonical v1 market-backstop wild p: rerun the exact 02/22 specification
    with 9,999 draws from the corrected shared _wild routine.
P5  Clean-window calendar event study (62-D spec): equality test of the two
    lead bins and joint lead test.

Outputs: output/referee_shoreup.csv, output/cs_dose_dynamics.csv, stdout log.
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
import os, sys, io, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps
from _wild import wild_score_p

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"; D_KM = 200.0
R_PERM = 999; R_BOOT = 999
rows = []


def log(msg):
    print(msg, flush=True)


# ============================================================================
# P0 — stale-cell cross-check
# ============================================================================
log("== P0: cross-check paper wild p cells vs current CSVs ==")
res1 = pd.read_csv(f"{OUTD}/results.csv").set_index("tag")
res2 = pd.read_csv(f"{OUTD}/results_v2.csv").set_index("tag")
PAPER = {  # tag -> (paper est, paper se, paper wild p, where)
 "A_triplediff_y_backstop":      (-0.0202, 0.0103, 0.001, "C1-A"),
 "A_triplediff_y_relational":    ( 0.0014, 0.0087, 0.928, "C1-A"),
 "A_triplediff_y_rel_failure":   ( 0.0058, 0.0060, 0.391, "C1-A"),
 "A_triplediff_y_formalization": ( 0.0006, 0.0041, 0.216, "C1-A"),
 "B_market_dose_y_backstop":     (-0.0230, 0.0074, 0.051, "C1-B"),
 "B_market_dose_y_relational":   (-0.0139, 0.0115, 0.645, "C1-B"),
 "B_market_dose_y_rel_failure":  ( 0.0011, 0.0030, 0.439, "C1-B"),
 "B_market_dose_y_formalization":( 0.0008, 0.0046, 0.155, "C1-B"),
 "B_market_dose_logn":           ( 0.0223, 0.0492, 0.185, "C1-B"),
 "B_violence_dose_y_backstop":   (-0.069,  0.013,  0.001, "text/macros"),
 "E_theftplacebo_y_backstop":    (-0.0057, 0.0097, 0.358, "C3"),
 "E_theftplacebo_y_rel_failure": (-0.0053, 0.0033, 0.989, "C3"),
 "E_pre2020_y_backstop":         (-0.0272, 0.0092, 0.037, "C3"),
 "E_pre2020_y_rel_failure":      (-0.0020, 0.0037, 0.428, "C3"),
 "E_expcomp_direct_share_z":     (-0.0021, 0.0015, 0.619, "C3"),
 "E_expcomp_coercive_rate_z":    ( 0.0019, 0.0017, 0.355, "C3"),
}
n_stale = 0
for tag, (pe, pse, pwp, where) in PAPER.items():
    r = res1.loc[tag]
    flag = ""
    if abs(round(float(r["wild_p"]), 3) - pwp) >= 0.0005:
        flag = "  <-- STALE wild p in paper"; n_stale += 1
    log(f"  [{where:11s}] {tag:32s} csv: {r['est']:+.4f} ({r['se']:.4f}) "
        f"wild {r['wild_p']:.3f} | paper wild {pwp:.3f}{flag}")
    rows.append(dict(part="P0", tag=tag, est=float(r["est"]), se=float(r["se"]),
                     stat=float(r["wild_p"]), stat2=pwp, note=where + flag))
log(f"P0: {n_stale} stale wild-p cell(s) found in paper hardcoded tables")

# ============================================================================
# shared clean-window civil relational panel (mirrors step 26)
# ============================================================================
log("== building clean-window relational panel ==")
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp["month"] = cp["jmonth"].astype(str).str[:7]
cp = cp[(cp["month"] >= WINDOW[0]) & (cp["month"] <= WINDOW[1])].copy()
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
cp = cp.merge(sched, on="province", how="left").dropna(subset=["exposure_v2_z"])
cp = cp[cp["cause_family"] == "relational"].copy()
cp["pcode2"] = cp["prefecture_code"].astype(str).str[:2]
cp["treat"] = (cp["inspection_round"] == 1).astype(int)
cp["postc"] = (cp["month"] >= POST0).astype(int)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["pref_cause"] = cp["prefecture_code"].astype(str) + "_" + cp["cause"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp["H"] = cp["exposure_v2_z"]
cp = cp.reset_index(drop=True)

ADJ = {"11":"12,13","12":"11,13","13":"11,12,14,15,21,37,41","14":"13,15,41,61",
 "15":"13,14,21,22,23,61,62,64","21":"13,15,22","22":"15,21,23","23":"15,22",
 "31":"32,33","32":"31,33,34,37","33":"31,32,34,35,36","34":"32,33,36,37,41,42",
 "35":"33,36,44","36":"33,34,35,42,43,44","37":"13,32,34,41",
 "41":"13,14,34,37,42,61","42":"34,36,41,43,50,61","43":"36,42,44,45,50,52",
 "44":"35,36,43,45","45":"43,44,52,53","46":"","50":"42,43,51,52,61",
 "51":"50,52,53,54,61,62,63","52":"43,45,50,51,53","53":"45,51,52,54",
 "54":"51,53,63,65","61":"14,15,41,42,50,51,62,64","62":"15,51,61,63,64,65",
 "63":"51,54,62,65","64":"15,61,62","65":"54,62,63"}
ADJ = {k: set(v.split(",")) - {""} for k, v in ADJ.items()}


def haversine(la1, lo1, la2, lo2):
    r = np.pi / 180; R = 6371.0
    dla = (la2 - la1) * r; dlo = (lo2 - lo1) * r
    a = np.sin(dla/2)**2 + np.cos(la1*r)*np.cos(la2*r)*np.sin(dlo/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


cen = pd.read_csv(f"{DATA}/pref_centroids.csv", dtype={"prefecture_code": str})
pref = cp[["prefecture_code", "pcode2", "province"]].drop_duplicates(
    "prefecture_code").merge(cen, on="prefecture_code", how="left").dropna(
    subset=["lat"]).reset_index(drop=True)
P_ = len(pref)
La = np.asarray(pref["lat"], float); Lo = np.asarray(pref["lon"], float)
Pc = np.asarray(pref["pcode2"], dtype=object)
DM_all = haversine(La[:, None], Lo[:, None], La[None, :], Lo[None, :])
DM_all = np.where(Pc[:, None] != Pc[None, :], DM_all, np.inf)

# row-index caches over cp
pref_pos = {c: i for i, c in enumerate(pref["prefecture_code"])}
cp["pref_i"] = cp["prefecture_code"].map(pref_pos)          # NaN if no centroid
cp_pc = pd.factorize(cp["pref_cause"])[0]
cp_mc = pd.factorize(cp["month"])[0]
cp_rows_by_pref = {int(i): np.where(cp["pref_i"].values == i)[0]
                   for i in pref.index}
ASINH = cp["asinh_n"].values; HARR = cp["H"].values
POSTC = cp["postc"].values.astype(float)
prov_by_pref = pref["province"].values
prov_list = sorted(cp["province"].unique())
treat_by_prov_obs = dict(cp[["province", "treat"]].drop_duplicates().values)
n_treat = int(sum(treat_by_prov_obs[p] for p in prov_list))
log(f"panel: {len(cp):,} cells, {P_} prefectures with centroids, "
    f"{len(prov_list)} provinces, {n_treat} first-wave")


def build_pairs(tmask):
    """DLR pairs given prefecture-level treatment mask; returns list of (i,j)."""
    Ti = np.where(tmask)[0]; Ci = np.where(~tmask)[0]
    if len(Ti) == 0 or len(Ci) == 0: return []
    D = DM_all[np.ix_(Ti, Ci)]
    tn = D.min(axis=1); cn = D.min(axis=0)
    am1 = D.argmin(axis=1); am0 = D.argmin(axis=0)
    prs = set()
    for k in range(len(Ti)):
        if tn[k] <= D_KM: prs.add((int(Ti[k]), int(Ci[am1[k]])))
    for k in range(len(Ci)):
        if cn[k] <= D_KM: prs.add((int(Ti[am0[k]]), int(Ci[k])))
    return sorted(prs)


def fit_designs(treat_by_prov, want=("contig", "pborder", "dlr"), vc="iid"):
    """Fit the three geographic designs under a province->treat map."""
    tr_pref = np.array([treat_by_prov[p] for p in prov_by_pref], dtype=float)
    tr_row = cp["province"].map(treat_by_prov).values.astype(float)
    pt = POSTC * tr_row; pth = pt * HARR; ph = POSTC * HARR
    out = {}
    base = pd.DataFrame({"asinh_n": ASINH, "pth": pth, "ph": ph, "pt": pt,
                         "pc": cp_pc, "mc": cp_mc})
    if "contig" in want:
        cr = {}
        for c in np.unique(Pc):
            provs_c = pref.loc[pref["pcode2"] == c, "province"]
            cr[c] = "T" if treat_by_prov[provs_c.iloc[0]] == 1 else "C"
        bc = {c for c in cr
              if any(cr.get(n) == ("C" if cr[c] == "T" else "T")
                     for n in ADJ.get(c, set()))}
        mrows = cp["pcode2"].isin(bc).values
        m = pf.feols("asinh_n ~ pth + ph + pt | pc + mc",
                     data=base[mrows], vcov=vc)
        out["contig"] = float(m.coef()["pth"])
    tmask = tr_pref == 1
    prs = build_pairs(tmask)
    if "pborder" in want:
        bp = sorted({i for ij in prs for i in ij})
        keep = np.isin(cp["pref_i"].values, bp)
        m = pf.feols("asinh_n ~ pth + ph + pt | pc + mc",
                     data=base[keep], vcov=vc)
        out["pborder"] = float(m.coef()["pth"])
    if "dlr" in want:
        if not prs:
            out["dlr"] = np.nan
        else:
            idx, pid = [], []
            for k, (i, j) in enumerate(prs):
                ri = np.concatenate([cp_rows_by_pref[i], cp_rows_by_pref[j]])
                idx.append(ri); pid.append(np.full(len(ri), k))
            idx = np.concatenate(idx); pid = np.concatenate(pid)
            sub = base.iloc[idx].copy()
            sub["ppc"] = pid * 100000 + cp_pc[idx]
            sub["pm"] = pid * 1000 + cp_mc[idx]
            m = pf.feols("asinh_n ~ pth + ph + pt | ppc + pm",
                         data=sub, vcov=vc)
            out["dlr"] = float(m.coef()["pth"])
    out["_npairs"] = len(prs)
    return out


# ============================================================================
# P1 — wave-timing permutation for the geographic designs
# ============================================================================
log("== P1: geographic-design wave permutation (999 draws) ==")
t0 = time.time()
obs = fit_designs(treat_by_prov_obs)
log(f"observed: contig {obs['contig']:+.4f}  pborder {obs['pborder']:+.4f}  "
    f"dlr {obs['dlr']:+.4f}  pairs {obs['_npairs']} "
    f"(paper: +0.2172 / +0.1908 / +0.3171 / 82)")
rng = np.random.default_rng(42)
hits = {k: 0 for k in ("contig", "pborder", "dlr")}
valid = {k: 0 for k in ("contig", "pborder", "dlr")}
npairs_draws = []
for r in range(R_PERM):
    lab = np.zeros(len(prov_list), int)
    lab[rng.choice(len(prov_list), n_treat, replace=False)] = 1
    tmap = dict(zip(prov_list, lab))
    try:
        d = fit_designs(tmap)
    except Exception:
        continue
    npairs_draws.append(d["_npairs"])
    for k in hits:
        if np.isfinite(d.get(k, np.nan)):
            valid[k] += 1
            if abs(d[k]) >= abs(obs[k]): hits[k] += 1
    if (r + 1) % 100 == 0:
        log(f"   P1 {r+1}/{R_PERM} ({time.time()-t0:.0f}s)")
for k in ("contig", "pborder", "dlr"):
    pp = (1 + hits[k]) / (1 + valid[k])
    log(f"P1 {k:8s}: obs {obs[k]:+.4f}, perm p = {pp:.3f} "
        f"(hits {hits[k]}/{valid[k]})")
    rows.append(dict(part="P1", tag=f"perm_{k}", est=obs[k], se=np.nan,
                     stat=pp, stat2=valid[k], note="wave-timing RI, 999 draws"))
log(f"P1 pairs per draw: median {np.median(npairs_draws):.0f}, "
    f"IQR [{np.percentile(npairs_draws,25):.0f},{np.percentile(npairs_draws,75):.0f}]")

# ============================================================================
# P2 — Callaway-Sant'Anna: continuous dose slope + quartile contrast
# ============================================================================
log("== P2: CS continuous-dose upgrade ==")


def midx(s): return s.str[:4].astype(int) * 12 + s.str[5:7].astype(int)


kv = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
prefs2 = kv[["prefecture_code", "province", "exposure_v2_z", "insp_month"]
            ].drop_duplicates("prefecture_code").dropna(
    subset=["exposure_v2_z"]).reset_index(drop=True)
prefs2["insp"] = prefs2["insp_month"].astype(str).str[:7]
months = pd.period_range("2014-01", "2020-12", freq="M").astype(str)
grid = prefs2.merge(pd.DataFrame({"month": months}), how="cross")

cv = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cv["month"] = cv["jmonth"].astype(str).str[:7]
rel_n = cv[cv["cause_family"] == "relational"].groupby(
    ["prefecture_code", "month"])["n_cases"].sum().rename("n_rel")
plc_n = cv[cv["cause_family"] == "placebo"].groupby(
    ["prefecture_code", "month"])["n_cases"].sum().rename("n_plc")
grid = grid.merge(rel_n, on=["prefecture_code", "month"], how="left") \
           .merge(plc_n, on=["prefecture_code", "month"], how="left")
enf = kv[kv["family"] == "enforcementcrime"].copy()
enf["month"] = enf["jmonth"].astype(str).str[:7]
enf_n = enf.groupby(["prefecture_code", "month"])["n_cases"].sum().rename("n_enf")
grid = grid.merge(enf_n, on=["prefecture_code", "month"], how="left")
grid[["n_rel", "n_plc", "n_enf"]] = grid[["n_rel", "n_plc", "n_enf"]].fillna(0)
grid["y_civ"] = np.arcsinh(grid["n_rel"]) - np.arcsinh(grid["n_plc"])
grid["y_enf"] = np.arcsinh(grid["n_enf"])
grid["t"] = midx(grid["month"]); grid["g"] = midx(grid["insp"])

PRE, POST = 12, 8
EV = list(range(-PRE, -1)) + list(range(0, POST + 1))
POSTS = [e for e in EV if e >= 0]; PRES = [e for e in EV if e < -1]


def wide(ycol):
    w = grid.pivot_table(index="prefecture_code", columns="t", values=ycol)
    w = w.reindex(prefs2["prefecture_code"])
    return np.asarray(w), {t: i for i, t in enumerate(w.columns)}


G_ARR = midx(prefs2["insp"]).values
H_ARR = prefs2["exposure_v2_z"].values
PROV2 = prefs2["province"].values
TERC = pd.qcut(H_ARR, 3, labels=False).astype(int)
QUART = pd.qcut(H_ARR, 4, labels=False).astype(int)
prov2_list = sorted(pd.unique(PROV2))
prov2_rows = {p: np.where(PROV2 == p)[0] for p in prov2_list}


def cs_all(M, col_of, idx):
    """Return {estimator: {e: value}} for tercile/quartile contrasts and the
    continuous-dose slope, on prefecture rows idx (bootstrap multiset ok).
    Tercile/quartile accumulate num/den PER dose group across g, exactly as
    step 22 (high and low groups enter independently per (g,e))."""
    Mi = M[idx]; gi = G_ARR[idx]; Hi = H_ARR[idx]
    Ti3 = TERC[idx]; Ti4 = QUART[idx]
    out = {"terc": {}, "quart": {}, "slope": {}}
    for e in EV:
        grp = {"terc": {2: [0.0, 0.0], 0: [0.0, 0.0]},
               "quart": {3: [0.0, 0.0], 0: [0.0, 0.0]}}
        sl = [0.0, 0.0]
        for g in np.unique(gi):
            t = int(g) + e
            if t not in col_of or (int(g) - 1) not in col_of: continue
            dy = Mi[:, col_of[t]] - Mi[:, col_of[int(g) - 1]]
            fin = np.isfinite(dy)
            trm = (gi == g) & fin
            ctm = (gi > max(t, int(g))) & fin
            # tercile / quartile top-bottom contrasts (mirrors step 22 exactly)
            for key, lab in (("terc", Ti3), ("quart", Ti4)):
                for T in grp[key]:
                    tr = dy[trm & (lab == T)]; ct = dy[ctm & (lab == T)]
                    if len(tr) < 3 or len(ct) < 3: continue
                    grp[key][T][0] += len(tr) * (tr.mean() - ct.mean())
                    grp[key][T][1] += len(tr)
            # continuous-dose slope: treated slope minus NYT slope
            ntr = int(trm.sum())
            if ntr >= 5 and ctm.sum() >= 5:
                hT, yT = Hi[trm], dy[trm]; hC, yC = Hi[ctm], dy[ctm]
                vT = ((hT - hT.mean()) ** 2).sum()
                vC = ((hC - hC.mean()) ** 2).sum()
                if vT > 0 and vC > 0:
                    bT = ((hT - hT.mean()) * (yT - yT.mean())).sum() / vT
                    bC = ((hC - hC.mean()) * (yC - yC.mean())).sum() / vC
                    sl[0] += ntr * (bT - bC); sl[1] += ntr
        for key in ("terc", "quart"):
            hiT = 2 if key == "terc" else 3
            (nh, dh), (nl, dl) = grp[key][hiT], grp[key][0]
            if dh > 0 and dl > 0:
                out[key][e] = nh / dh - nl / dl
        if sl[1] > 0:
            out["slope"][e] = sl[0] / sl[1]
    return out


def cs_summary(label, ycol):
    M, col_of = wide(ycol)
    all_idx = np.arange(len(prefs2))
    point = cs_all(M, col_of, all_idx)
    tv = np.mean([point["terc"][e] for e in POSTS if e in point["terc"]])
    log(f"P2 {label}: tercile point overall {tv:+.4f} "
        f"(step-22 cached: civilgap +0.1087 / enforce -0.1331)")
    rng2 = np.random.default_rng(42)
    boot = {k: {e: [] for e in EV} for k in point}
    boot_o = {k: [] for k in point}
    t1 = time.time()
    for r in range(R_BOOT):
        draw = rng2.choice(prov2_list, size=len(prov2_list), replace=True)
        idx = np.concatenate([prov2_rows[p] for p in draw])
        b = cs_all(M, col_of, idx)
        for k in point:
            ok = [e for e in POSTS if e in b[k]]
            if ok: boot_o[k].append(np.mean([b[k][e] for e in ok]))
            for e in b[k]: boot[k][e].append(b[k][e])
        if (r + 1) % 250 == 0:
            log(f"   P2 {label} boot {r+1}/{R_BOOT} ({time.time()-t1:.0f}s)")
    for k in ("terc", "quart", "slope"):
        posts_k = [e for e in POSTS if e in point[k]]
        pres_k = [e for e in PRES if e in point[k]]
        overall = np.mean([point[k][e] for e in posts_k])
        se_o = np.std(boot_o[k], ddof=1)
        p_o = 2 * (1 - sps.norm.cdf(abs(overall / se_o)))
        pre_p = np.nan
        if pres_k:
            nb = min(len(boot[k][e]) for e in pres_k)
            B = np.array([[boot[k][e][j] for e in pres_k] for j in range(nb)])
            bvec = np.array([point[k][e] for e in pres_k])
            try:
                pre_p = float(1 - sps.chi2.cdf(
                    bvec @ np.linalg.solve(np.cov(B.T), bvec), len(pres_k)))
            except Exception:
                pass
        log(f"P2 {label} [{k:5s}]: overall(e0..{POST}) {overall:+.4f} "
            f"(boot SE {se_o:.4f}, p {p_o:.3f}); joint lead p {pre_p:.3f}")
        rows.append(dict(part="P2", tag=f"CS_{label}_{k}", est=overall,
                         se=se_o, stat=p_o, stat2=pre_p,
                         note=f"posts {len(posts_k)}, leads {len(pres_k)}"))
        if k == "slope":
            dyn = [dict(outcome=label, e=e, est=point[k][e],
                        se=np.std(boot[k][e], ddof=1)
                        if len(boot[k][e]) > 2 else np.nan)
                   for e in sorted(point[k])]
            pd.DataFrame(dyn).to_csv(
                f"{OUTD}/cs_dose_dynamics_{label}.csv", index=False)


cs_summary("civilgap", "y_civ")
cs_summary("enforce", "y_enf")

# ============================================================================
# P3 — MDEs for the audited criminal margins (Table 3)
# ============================================================================
log("== P3: MDEs, audited criminal margins ==")
MULT = float(sps.t.ppf(0.975, 30) + sps.t.ppf(0.80, 30))
log(f"MDE multiplier (t(30), 80% power, 5% two-sided): {MULT:.3f}")
kp3 = kv[kv["n_cases"] > 0].copy()
kp3["month"] = kp3["jmonth"].astype(str).str[:7]
pre3 = kp3[kp3["post"] == 0]


def wmean(d, col, w="n_cases"):
    d = d.dropna(subset=[col])
    return float(np.average(d[col], weights=d[w]))


base_means = {
 "K2_market_backstop": wmean(pre3[pre3["family"] == "market"], "y_backstop"),
 "K2_market_relfail": wmean(pre3[pre3["family"] == "market"], "y_rel_fail"),
 "K2_market_formalization": wmean(pre3[pre3["family"] == "market"],
                                  "y_formalization"),
 "K2_enforcement_detentiondebt": wmean(pre3[pre3["family"] == "enforcementcrime"],
                                       "y_detention_debt"),
 "K2_enforcement_asinhN": np.nan,  # asinh outcome: MDE already in log points
}
for tag, bm in base_means.items():
    se = float(res2.loc[tag, "se"]); est = float(res2.loc[tag, "est"])
    mde = MULT * se
    share = mde / bm if np.isfinite(bm) and bm > 0 else np.nan
    log(f"P3 {tag:30s} est {est:+.4f} se {se:.4f} MDE {mde:.4f} "
        f"pre-mean {bm if np.isfinite(bm) else float('nan'):.4f} "
        f"MDE/mean {share if np.isfinite(share) else float('nan'):.2f}")
    rows.append(dict(part="P3", tag=f"MDE_{tag}", est=mde, se=se,
                     stat=bm, stat2=share, note="MULT=%.3f" % MULT))

# v1 proportional benchmark
p1 = pd.read_parquet(f"{DATA}/panel_month.parquet")
p1 = p1[(p1["analysis_group"] == "market") & (p1["n_fact"] > 0)].copy()
pre1 = p1[p1["post_judgment"] == 0]
v1_mean = float(np.average(pre1["y_backstop"].dropna(),
                           weights=pre1.loc[pre1["y_backstop"].notna(), "n_fact"]))
v1_est = float(res1.loc["B_market_dose_y_backstop", "est"])
v1_prop = abs(v1_est) / v1_mean
aud_mde_prop = (MULT * float(res2.loc["K2_market_backstop", "se"])
                / base_means["K2_market_backstop"])
log(f"P3 v1 benchmark: v1 pre-mean {v1_mean:.4f}, v1 est {v1_est:+.4f} "
    f"= {100*v1_prop:.1f}% of baseline; audited MDE = "
    f"{100*aud_mde_prop:.1f}% of ITS baseline "
    f"-> audited design {'CANNOT' if aud_mde_prop > v1_prop else 'can'} "
    f"detect a v1-sized proportional decline")
rows.append(dict(part="P3", tag="v1_benchmark", est=v1_prop, se=np.nan,
                 stat=aud_mde_prop, stat2=v1_mean,
                 note="proportional decline: v1 vs audited MDE"))

# ============================================================================
# P4 — canonical v1 market-backstop wild p (9,999 draws)
# ============================================================================
log("== P4: canonical v1 market backstop wild p ==")
p1["prov_id"] = pd.factorize(p1["province"])[0]
p1["pref"] = p1["prefecture_code"]
p1["prov_month"] = p1["province"] + "_" + p1["month"].astype(str) \
    if "month" in p1.columns else p1["province"] + "_" + \
    p1["judgment_month"].astype(str)
p1["px"] = p1["post_judgment"] * p1["exposure_z"]
FML4 = "y_backstop ~ px + x_factshare + x_spanshare | pref + prov_month"
m4 = pf.feols(FML4, data=p1, vcov={"CRV1": "prov_id"}, weights="n_fact")
wp9999 = wild_score_p(FML4, p1, "px", weights="n_fact", reps=9_999)
log(f"P4 v1 baseline: {m4.coef()['px']:+.5f} ({m4.se()['px']:.5f}) "
    f"CRV1 p={m4.pvalue()['px']:.4f}  wild(9999)={wp9999:.4f}  "
    f"N={int(m4._N):,}")
rows.append(dict(part="P4", tag="v1_backstop_canonical", est=float(m4.coef()["px"]),
                 se=float(m4.se()["px"]), stat=wp9999, stat2=np.nan,
                 note=f"CRV1 p {float(m4.pvalue()['px']):.4f}, N {int(m4._N)}"))

# ============================================================================
# P5 — clean-window calendar event study: lead-equality test (62-D spec)
# ============================================================================
log("== P5: Panel D lead equality ==")
esd = pd.read_parquet(f"{DATA}/civil_panel.parquet")
esd = esd[esd["cause_family"] == "relational"].copy()
esd["month"] = esd["jmonth"].astype(str).str[:7]
esd = esd[(esd["month"] >= WINDOW[0]) & (esd["month"] <= WINDOW[1])]
esd = esd.merge(sched, on="province", how="left")
esd["treat"] = (esd["inspection_round"] == 1).astype(int)
esd["prov_id"] = pd.factorize(esd["province"])[0]
esd["pref_cause"] = esd["prefecture_code"] + "_" + esd["cause"]
esd["asinh_n"] = np.arcsinh(esd["n_cases"])
et = (pd.PeriodIndex(esd["month"], freq="M") - pd.Period(POST0, freq="M"))
esd["et"] = [x.n for x in et]
BINS5 = [(-20, -13), (-12, -7), (0, 6)]
terms5 = []
for lo, hi in BINS5:
    nmH = f"b{lo}_{hi}H".replace("-", "m")
    nmT = f"b{lo}_{hi}T".replace("-", "m")
    nmX = f"b{lo}_{hi}X".replace("-", "m")
    inb = ((esd["et"] >= lo) & (esd["et"] <= hi)).astype(float)
    esd[nmH] = inb * esd["treat"] * esd["exposure_v2_z"]
    esd[nmT] = inb * esd["treat"]
    esd[nmX] = inb * esd["exposure_v2_z"]
    terms5 += [nmH, nmT, nmX]
m5 = pf.feols(f"asinh_n ~ {' + '.join(terms5)} | pref_cause + month",
              data=esd, vcov={"CRV1": "prov_id"})
names5 = list(m5.coef().index)
b1n, b2n = "bm20_m13H", "bm12_m7H"
for lo, hi in BINS5:
    nm = f"b{lo}_{hi}H".replace("-", "m")
    log(f"P5 [{lo:+d},{hi:+d}] TreatxH: {m5.coef()[nm]:+.4f} ({m5.se()[nm]:.4f})")
i1, i2 = names5.index(b1n), names5.index(b2n)
bd = float(m5.coef()[b1n] - m5.coef()[b2n])
vd = float(m5._vcov[i1, i1] + m5._vcov[i2, i2] - 2 * m5._vcov[i1, i2])
tstat = bd / np.sqrt(vd); p_eq = float(2 * (1 - sps.t.cdf(abs(tstat), 30)))
bvec5 = np.array([m5.coef()[b1n], m5.coef()[b2n]])
V5 = m5._vcov[np.ix_([i1, i2], [i1, i2])]
p_joint = float(1 - sps.chi2.cdf(float(bvec5 @ np.linalg.solve(V5, bvec5)), 2))
log(f"P5 lead equality: diff {bd:+.4f} (se {np.sqrt(vd):.4f}), "
    f"t={tstat:.2f}, p={p_eq:.3f}; joint leads p={p_joint:.3f}")
rows.append(dict(part="P5", tag="lead_equality", est=bd, se=float(np.sqrt(vd)),
                 stat=p_eq, stat2=p_joint, note="b[-20,-13]H - b[-12,-7]H"))

pd.DataFrame(rows).to_csv(f"{OUTD}/referee_shoreup.csv", index=False)
log("step 70 complete: referee_shoreup.csv written; NO tex modified")
