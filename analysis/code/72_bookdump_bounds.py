# -*- coding: utf-8 -*-
"""6B step 72 — worst-case book-dump deletion bounds.

Builds case-level suspicion flags for clean-window lending cases and re-estimates
the headline margins after deleting each tier. Over-deletion is the point: every
tier deletes MORE than the true mechanical-transfer set, so surviving estimates
are lower bounds on the non-mechanical response.

Flags (lending cases, 2017-01..2019-03):
  LAWYER5 / LAWYER3  same lawyer-firm pair x same court files >=5 (>=3) lending
                     cases within any 6-month span -> flag the pair's whole book
  CITE               the judgment cites a criminal case number, OR its own case
                     number is cited inside a criminal judgment (step-71 scans)
  ORG                organizational first plaintiff (step-32 flag)
  BATCH              consecutive first-instance case-number runs >=5 in court-year
  HIRATE             recorded monthly rate > 1.5 percent

Tiers: BASE, L5, L3, CITE, L5+CITE, L5+CITE+ORG+BATCH, ALL(+HIRATE).
Re-estimated outcomes per tier:
  flow   pooled relational-family clean-window flow (civil_panel cells with
         deleted counts subtracted; replicates E1_civ_stacked_baseline = 0.156)
  acq / stranger / lend  38-style lending dose splits rebuilt from case level.

Output: output/bookdump_bounds.csv
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
import os, sys, io, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import pyfixest as pf
from _wild import wild_p

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
WIN = ("2017-01", "2019-03"); POST0 = "2018-09"
LEND = "民间借贷纠纷"
rows = []

# ---------------- case-level lending frame ------------------------------------
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["case_no", "cause", "cause_family", "prefecture_code",
                              "province", "jmonth", "rel_txn", "monthly_rate_pct"])
cc["month"] = cc["jmonth"].astype(str).str[:7]
cc = cc[(cc["month"] >= WIN[0]) & (cc["month"] <= WIN[1])]
rel = cc[cc["cause_family"] == "relational"].copy()
ld = rel[rel["cause"] == LEND].copy()
print(f"window relational cases {len(rel):,}; lending {len(ld):,}", flush=True)

scan = pd.read_parquet(f"{DATA}/mech_civil_scan.parquet").rename(
    columns={"案号": "case_no"})
rel = rel.merge(scan[["case_no", "lawyer_keys", "n_crimcite"]],
                on="case_no", how="left")
ld = ld.merge(scan[["case_no", "lawyer_keys", "n_crimcite"]],
              on="case_no", how="left")
print(f"scan matched: rel {rel['n_crimcite'].notna().mean():.3f} "
      f"lend {ld['n_crimcite'].notna().mean():.3f}", flush=True)

org = pd.read_parquet(f"{DATA}/civil_party_orgflag.parquet").rename(
    columns={"案号": "case_no"})
ld = ld.merge(org[["case_no", "first_org"]], on="case_no", how="left")
ld["ORG"] = ld["first_org"].fillna(False).astype(bool)
ld["HIRATE"] = ld["monthly_rate_pct"].notna() & (ld["monthly_rate_pct"] > 1.5)

# ---------------- lawyer-cluster flags -----------------------------------------


def norm_no(s):
    return str(s).replace(" ", "").replace("(", "（").replace(")", "）")


ld["cno"] = ld["case_no"].map(norm_no)
ld["court_code"] = ld["cno"].str.extract(r"（\d{4}）([^（）]*?)民初", expand=False)
ld["serial"] = pd.to_numeric(
    ld["cno"].str.extract(r"民初(?:字第)?(\d+)号", expand=False), errors="coerce")
ld["cyear"] = ld["cno"].str.extract(r"（(\d{4})）", expand=False)
ld["ymi"] = (ld["month"].str[:4].astype(int) * 12
             + ld["month"].str[5:7].astype(int))

lw = ld.loc[ld["lawyer_keys"].fillna("") != "",
            ["case_no", "lawyer_keys", "court_code", "ymi"]].copy()
lw = lw.assign(key=lw["lawyer_keys"].str.split(";")).explode("key")
lw = lw[lw["key"].str.len() > 4]
lw["grp"] = lw["key"] + "@" + lw["court_code"].fillna("?")


def cluster_flag(kmin, span=6):
    flagged_groups = []
    for g, s in lw.groupby("grp")["ymi"]:
        v = np.sort(s.values)
        if len(v) < kmin: continue
        j = 0
        for i in range(len(v)):
            while v[i] - v[j] > span - 1: j += 1
            if i - j + 1 >= kmin:
                flagged_groups.append(g); break
    fg = set(flagged_groups)
    cases = set(lw.loc[lw["grp"].isin(fg), "case_no"])
    return cases, len(fg)


c5, g5 = cluster_flag(5)
c3, g3 = cluster_flag(3)
ld["LAWYER5"] = ld["case_no"].isin(c5)
ld["LAWYER3"] = ld["case_no"].isin(c3)
print(f"lawyer clusters: K5/6mo {g5:,} pairs -> {len(c5):,} cases "
      f"({ld['LAWYER5'].mean():.3f} of lending); K3/6mo {g3:,} -> {len(c3):,} "
      f"({ld['LAWYER3'].mean():.3f})", flush=True)

# ---------------- citation flags ----------------------------------------------
cited = pd.read_parquet(f"{DATA}/mech_crim_civcite.parquet")
cited_set = set()
for s in cited["civ_cites"]:
    cited_set.update(s.split(";"))
print(f"criminal judgments cite {len(cited_set):,} distinct civil case numbers "
      f"({len(cited):,} citing criminal cases)", flush=True)
for d in (ld, rel):
    d["CITE"] = (d["n_crimcite"].fillna(0) > 0) | \
                d["case_no"].map(norm_no).isin(cited_set)
print(f"CITE: lending {ld['CITE'].mean():.4f}, relational {rel['CITE'].mean():.4f}"
      f" (cited-by-criminal among lending: "
      f"{ld['case_no'].map(norm_no).isin(cited_set).mean():.4f})", flush=True)

# ---------------- batch case-number runs ----------------------------------------
ld["BATCH"] = False
sub = ld.dropna(subset=["serial", "court_code", "cyear"])
for (cc_, yy), s in sub.groupby(["court_code", "cyear"])["serial"]:
    v = np.sort(s.unique().astype(int))
    if len(v) < 5: continue
    runs = np.split(v, np.where(np.diff(v) != 1)[0] + 1)
    hit = np.concatenate([r for r in runs if len(r) >= 5]) if \
        any(len(r) >= 5 for r in runs) else None
    if hit is not None:
        m = (ld["court_code"] == cc_) & (ld["cyear"] == yy) & \
            ld["serial"].isin(hit)
        ld.loc[m, "BATCH"] = True
print(f"BATCH runs: {ld['BATCH'].mean():.4f} of lending", flush=True)

# ---------------- estimation builders ------------------------------------------
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[
    ["prefecture_code", "exposure_v2_z"]]


def dose(df, tag, outcome):
    g = (df.groupby(["prefecture_code", "province", "month"]).size().rename("n")
         .reset_index().merge(sched, on="province").merge(ex, on="prefecture_code")
         .dropna(subset=["exposure_v2_z", "inspection_round"]))
    g["H"] = g["exposure_v2_z"]; g["pref"] = g["prefecture_code"]
    g["treat"] = (g["inspection_round"] == 1).astype(int)
    g["postc"] = (g["month"] >= POST0).astype(int)
    g["prov_id"] = pd.factorize(g["province"])[0]
    g["pt"] = g["postc"] * g["treat"]; g["pth"] = g["pt"] * g["H"]
    g["ph"] = g["postc"] * g["H"]; g["y"] = np.arcsinh(g["n"])
    m = pf.feols("y ~ pth + ph + pt | pref + month", data=g,
                 vcov={"CRV1": "prov_id"})
    wp = wild_p("y ~ pth + ph + pt | pref + month", g, "pth")
    rows.append(dict(tier=tag, outcome=outcome, est=m.coef()["pth"],
                     se=m.se()["pth"], p=m.pvalue()["pth"], wild_p=wp,
                     n_cases=len(df)))
    print(f"  {tag:14s} {outcome:9s} {m.coef()['pth']:+.4f} "
          f"({m.se()['pth']:.4f}) p={m.pvalue()['pth']:.3f} wild={wp:.3f} "
          f"cases={len(df):,}", flush=True)


cpan = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cpan = cpan[cpan["cause_family"] == "relational"].copy()
cpan["month"] = cpan["jmonth"].astype(str).str[:7]
cpan = cpan[(cpan["month"] >= WIN[0]) & (cpan["month"] <= WIN[1])]
cpan = cpan.merge(sched, on="province", how="left")
cpan["treat"] = (cpan["inspection_round"] == 1).astype(int)
cpan["postc"] = (cpan["month"] >= POST0).astype(int)
cpan["prov_id"] = pd.factorize(cpan["province"])[0]
cpan["pref_cause"] = cpan["prefecture_code"] + "_" + cpan["cause"]
cpan["pt"] = cpan["postc"] * cpan["treat"]


def flow(del_frame, tag):
    """Pooled relational flow with deleted counts subtracted from cells."""
    d = cpan.copy()
    if del_frame is not None and len(del_frame):
        dc = (del_frame.groupby(["prefecture_code", "cause", "month"]).size()
              .rename("n_del").reset_index())
        d = d.merge(dc, on=["prefecture_code", "cause", "month"], how="left")
        d["n_del"] = d["n_del"].fillna(0)
    else:
        d["n_del"] = 0
    d["n_adj"] = (d["n_cases"] - d["n_del"]).clip(lower=0)
    d = d.dropna(subset=["exposure_v2_z"]).copy()
    d["asinh_n"] = np.arcsinh(d["n_adj"])
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    fml = "asinh_n ~ pth + ph + pt | pref_cause + month"
    m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
    wp = wild_p(fml, d, "pth")
    ndel = int(d["n_del"].sum())
    rows.append(dict(tier=tag, outcome="flow", est=m.coef()["pth"],
                     se=m.se()["pth"], p=m.pvalue()["pth"], wild_p=wp,
                     n_cases=ndel))
    print(f"  {tag:14s} flow      {m.coef()['pth']:+.4f} ({m.se()['pth']:.4f}) "
          f"p={m.pvalue()['pth']:.3f} wild={wp:.3f} deleted={ndel:,}", flush=True)


# ---------------- tiers ---------------------------------------------------------
TIERS = [
    ("BASE", None),
    ("L5", ld["LAWYER5"]),
    ("L3", ld["LAWYER3"]),
    ("CITE", ld["CITE"]),
    ("L5+CITE", ld["LAWYER5"] | ld["CITE"]),
    ("L5+CITE+OB", ld["LAWYER5"] | ld["CITE"] | ld["ORG"] | ld["BATCH"]),
    ("ALL", ld["LAWYER5"] | ld["CITE"] | ld["ORG"] | ld["BATCH"] | ld["HIRATE"]),
]
for tag, mask in TIERS:
    print(f"== tier {tag} ==", flush=True)
    if mask is None:
        keep = ld; delf = None
    else:
        keep = ld[~mask]
        delf = ld[mask][["prefecture_code", "cause", "month"]]
        # relational non-lending cases: only the citation flag applies
        if "CITE" in tag or tag == "ALL":
            reldel = rel[(rel["cause"] != LEND) & rel["CITE"]][
                ["prefecture_code", "cause", "month"]]
            delf = pd.concat([delf, reldel], ignore_index=True)
        acq_share = ld.loc[mask, "rel_txn"].fillna(0).astype(int).mean()
        print(f"  deleting {mask.sum():,} lending cases "
              f"({mask.mean():.3f} of lending; acq share of deleted "
              f"{acq_share:.3f})", flush=True)
    flow(delf, tag)
    dose(keep, tag, "lend_all")
    dose(keep[keep["rel_txn"].fillna(0).astype(int) == 1], tag, "acq")
    dose(keep[keep["rel_txn"].fillna(0).astype(int) == 0], tag, "stranger")

pd.DataFrame(rows).to_csv(f"{OUTD}/bookdump_bounds.csv", index=False)
print("step 72 complete: bookdump_bounds.csv", flush=True)
