# -*- coding: utf-8 -*-
"""6B step 73 — the two-clock timing test.

Criminal de-militarization margins re-dated by the PROSECUTION month (向本院提起
公诉, step-71 extraction) instead of the judgment month; civil relational flow
dated by the FILING month (step-41 extraction). Both clocks sit months closer to
behavior than the judgment month, so the ordering "violence margins fall first,
relational litigation rises after" becomes testable within the corpus.

Parts:
  A  assemble criminal case level: extracts + prosecution dates, first-instance
     only, symmetric 270-day observation horizon (prosecution months <=2020-01)
  B  extraction-coverage neutrality (cell share ~ Post x H)
  C  prosecution-clock dose responses: enforcement caseload; market backstop
     share (audited d_backstop); judgment-clock comparators on the SAME sample
  D  event dynamics on both clocks + fig_twoclock.pdf
  E  audit sample export (200 extracted + 100 blank, with text snippets)

Outputs: output/twoclock_results.csv, output/twoclock_dynamics.csv,
         output/figures/fig_twoclock.pdf, output/prosdate_audit_sample.csv
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
import argparse, os, sys, io, re, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import pyfixest as pf
import pyarrow.parquet as pq
from _wild import wild_p

parser = argparse.ArgumentParser(description="Run the two-clock timing analysis.")
parser.add_argument(
    "--public-only",
    action="store_true",
    help="skip the restricted judgment-text audit sample after reproducing public results",
)
args = parser.parse_args()

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
EXT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
rows, dyn_rows = [], []

FAM = {}
for k, v in {("赌博", "开设赌场", "组织卖淫", "非法经营", "走私普通货物、物品"): "market",
             ("危险驾驶", "交通肇事", "过失致人死亡"): "placebo",
             ("盗窃",): "theft", ("故意伤害",): "violence",
             ("非法拘禁", "寻衅滋事", "聚众斗殴", "强迫交易", "敲诈勒索",
              "组织、领导、参加黑社会性质组织"): "enforcementcrime",
             ("诈骗",): "fraud"}.items():
    for c in k: FAM[c] = v

# ---------------- A: criminal case level with prosecution dates ---------------
print("== A: assemble ==", flush=True)
extract_files = sorted(glob.glob(f"{EXT}/crim_*.parquet"))
extract_columns = ["case_no", "court", "crime", "proceeding", "judgment_date",
                   "d_backstop", "doc_len"]
if extract_files and "docket_first_instance" in pq.read_schema(extract_files[0]).names:
    extract_columns.append("docket_first_instance")
ext = pd.concat([pd.read_parquet(f, columns=extract_columns)
                 for f in extract_files],
                ignore_index=True)
print(f"extract rows {len(ext):,}; proceeding: "
      f"{ext['proceeding'].value_counts(dropna=False).head(5).to_dict()}", flush=True)
pros = pd.read_parquet(f"{DATA}/mech_crim_prosdate.parquet").rename(
    columns={"案号": "case_no", "案由": "crime_raw"})
ext = ext.merge(pros[["case_no", "pros_ymd", "src"]], on="case_no", how="left")
docket_first = (
    ext["docket_first_instance"].fillna(False).astype(bool)
    if "docket_first_instance" in ext.columns
    else ext["case_no"].astype(str).str.contains("刑初")
)
first = ext["proceeding"].astype(str).str.contains("一审") | docket_first
ext = ext[first].copy()
ext["family"] = ext["crime"].map(FAM)
ext = ext.dropna(subset=["family"])
print(f"first-instance analysis cases {len(ext):,}; prosecution-date coverage "
      f"{ext['pros_ymd'].notna().mean():.3f}", flush=True)

xw = pd.read_parquet(f"{DATA}/court_xwalk.parquet")
ext = ext.merge(xw.rename(columns={"court_name": "court"})[
    ["court", "prefecture_code", "province"]], on="court", how="left")
ext = ext.dropna(subset=["prefecture_code"])
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[
    ["prefecture_code", "exposure_v2_z"]]
ext = ext.merge(ex, on="prefecture_code", how="left").dropna(
    subset=["exposure_v2_z"])
insp = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")[
    ["province", "insp_month"]].drop_duplicates()
insp["insp"] = insp["insp_month"].astype(str).str[:7]
ext = ext.merge(insp[["province", "insp"]], on="province", how="left")

ext["jd"] = pd.to_datetime(ext["judgment_date"], errors="coerce")
ext["pd_"] = pd.to_datetime(ext["pros_ymd"], errors="coerce")
ext["dur"] = (ext["jd"] - ext["pd_"]).dt.days
have = ext["pd_"].notna()
neg = (ext.loc[have, "dur"] < 0).mean()
print(f"QC: dur<0 share {neg:.4f}; dur quantiles "
      f"{ext.loc[have, 'dur'].quantile([.25, .5, .75, .9]).to_dict()}; "
      f"src pros {(ext.loc[have, 'src'] == 'pros').mean():.3f}", flush=True)
ok = have & ext["dur"].between(0, 270)
ext["pm"] = ext["pd_"].dt.strftime("%Y-%m")
ext["jm"] = ext["jd"].dt.strftime("%Y-%m")
samp = ext[ok & (ext["pm"] >= "2014-01") & (ext["pm"] <= "2020-01")].copy()
print(f"two-clock sample (0<=dur<=270, pm<=2020-01): {len(samp):,} "
      f"({ok.mean():.3f} of first-instance)", flush=True)

# ---------------- B: coverage neutrality ---------------------------------------
print("== B: coverage neutrality ==", flush=True)
cov = ext[ext["family"] == "enforcementcrime"].copy()
cov["okx"] = (cov["pd_"].notna() & cov["dur"].between(0, 270)).astype(float)
cov["jm7"] = cov["jm"].astype(str).str[:7]
cg = (cov.groupby(["prefecture_code", "province", "jm7"])
      .agg(share=("okx", "mean"), H=("exposure_v2_z", "first"),
           insp=("insp", "first")).reset_index())
cg["post"] = (cg["jm7"] >= cg["insp"]).astype(int)
cg["px"] = cg["post"] * cg["H"]
cg["prov_id"] = pd.factorize(cg["province"])[0]
cg["prov_month"] = cg["province"] + "_" + cg["jm7"]
mc = pf.feols("share ~ px | prefecture_code + prov_month", data=cg,
              vcov={"CRV1": "prov_id"})
print(f"coverage ~ PostxH: {mc.coef()['px']:+.4f} ({mc.se()['px']:.4f}) "
      f"p={mc.pvalue()['px']:.3f}", flush=True)
rows.append(dict(part="B", tag="coverage_neutrality", est=mc.coef()["px"],
                 se=mc.se()["px"], p=mc.pvalue()["px"], wild_p=np.nan, n=len(cg)))


# ---------------- C: dose responses on both clocks ------------------------------
def cellfit(d, mcol, fam, tag, share=False):
    dd = d[d["family"] == fam].copy()
    dd["m"] = dd[mcol]
    if share:
        g = (dd.groupby(["prefecture_code", "province", "m"])
             .agg(y=("d_backstop", "mean"), n=("case_no", "size"),
                  x_doclen=("doc_len", "mean"), H=("exposure_v2_z", "first"),
                  insp=("insp", "first")).reset_index())
        fml = "y ~ px + x_doclen | pref + prov_month"; w = "n"
    else:
        g = (dd.groupby(["prefecture_code", "province", "m"])
             .agg(n=("case_no", "size"), H=("exposure_v2_z", "first"),
                  insp=("insp", "first")).reset_index())
        g["y"] = np.arcsinh(g["n"]); fml = "y ~ px | pref + prov_month"; w = None
    g["post"] = (g["m"] >= g["insp"]).astype(int)
    g["px"] = g["post"] * g["H"]
    g["pref"] = g["prefecture_code"]
    g["prov_month"] = g["province"] + "_" + g["m"]
    g["prov_id"] = pd.factorize(g["province"])[0]
    m = pf.feols(fml, data=g, vcov={"CRV1": "prov_id"}, weights=w)
    try: wp = wild_p(fml, g, "px", weights=w)
    except Exception: wp = np.nan
    rows.append(dict(part="C", tag=tag, est=m.coef()["px"], se=m.se()["px"],
                     p=m.pvalue()["px"], wild_p=wp, n=int(m._N)))
    print(f"  {tag:34s} {m.coef()['px']:+.4f} ({m.se()['px']:.4f}) "
          f"p={m.pvalue()['px']:.3f} wild={wp:.3f} N={int(m._N):,}", flush=True)
    return g


print("== C: prosecution-clock vs judgment-clock ==", flush=True)
g_enf_p = cellfit(samp, "pm", "enforcementcrime", "enforceN_prosclock")
cellfit(samp, "jm", "enforcementcrime", "enforceN_judclock_samesample")
g_mkt_p = cellfit(samp, "pm", "market", "marketBackstop_prosclock", share=True)
cellfit(samp, "jm", "market", "marketBackstop_judclock_samesample", share=True)

# ---------------- D: event dynamics on both clocks ------------------------------
print("== D: dynamics ==", flush=True)
BINS = [(-24, -19), (-18, -13), (-12, -7), (-6, -1), (0, 5), (6, 11), (12, 17)]
REF = (-6, -1)


def dyn(g, tag, weights=None):
    d = g.copy()
    d["ei"] = ((d["m"].str[:4].astype(int) * 12 + d["m"].str[5:7].astype(int))
               - (d["insp"].str[:4].astype(int) * 12
                  + d["insp"].str[5:7].astype(int)))
    terms = []
    for lo, hi in BINS:
        if (lo, hi) == REF: continue
        nm = f"b_{lo}_{hi}".replace("-", "m")
        d[nm] = ((d["ei"] >= lo) & (d["ei"] <= hi)).astype(float) * d["H"]
        terms.append(nm)
    fml = f"y ~ {' + '.join(terms)}" + \
        (" + x_doclen" if "x_doclen" in d.columns else "") + " | pref + prov_month"
    m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"}, weights=weights)
    for t in terms:
        lo, hi = t[2:].replace("m", "-").rsplit("_", 1)
        dyn_rows.append(dict(series=tag, lo=int(lo), hi=int(hi),
                             est=m.coef()[t], se=m.se()[t]))
    print(f"  dynamics [{tag}] done", flush=True)


dyn(g_enf_p, "enforceN_pros")
dyn(g_mkt_p, "marketBackstop_pros", weights="n")

# civil filing clock (clean window, relational causes, dur<=270 as in step 62)
ccols = ["case_no", "cause", "cause_family", "prefecture_code", "province",
         "jmonth"]
cc = pd.read_parquet(f"{DATA}/civil_case.parquet", columns=ccols)
cc = cc[cc["cause_family"] == "relational"]
fil = pd.read_parquet(f"{DATA}/civil_filing.parquet").rename(
    columns={"案号": "case_no"})
cc = cc.merge(fil[["case_no", "filing_ymd"]], on="case_no", how="left")
cc["fd"] = pd.to_datetime(cc["filing_ymd"], errors="coerce")
cc["jd"] = pd.to_datetime(cc["jmonth"], errors="coerce")
cc["dur"] = (cc["jd"] - cc["fd"]).dt.days
cc = cc[cc["fd"].notna() & cc["dur"].between(0, 270)].copy()
cc["fm"] = cc["fd"].dt.strftime("%Y-%m")
cc = cc[(cc["fm"] >= "2017-01") & (cc["fm"] <= "2019-03")]
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
gf = (cc.groupby(["prefecture_code", "province", "fm"]).size().rename("n")
      .reset_index().merge(sched, on="province")
      .merge(ex, on="prefecture_code")
      .dropna(subset=["exposure_v2_z", "inspection_round"]))
gf["H"] = gf["exposure_v2_z"]; gf["treat"] = (gf["inspection_round"] == 1)
gf["y"] = np.arcsinh(gf["n"]); gf["prov_id"] = pd.factorize(gf["province"])[0]
CB = [(-20, -13), (-12, -7), (-6, -1), (0, 6)]
ei = (gf["fm"].str[:4].astype(int) * 12 + gf["fm"].str[5:7].astype(int)) \
    - (2018 * 12 + 9)
terms = []
for lo, hi in CB:
    if (lo, hi) == (-6, -1): continue
    for src, nm_sfx in ((gf["treat"].astype(float) * gf["H"], "H"),
                        (gf["treat"].astype(float), "T"),
                        (gf["H"], "X")):
        nm = f"c{lo}_{hi}{nm_sfx}".replace("-", "m")
        gf[nm] = ((ei >= lo) & (ei <= hi)).astype(float) * src
        if nm_sfx == "H": terms.append(nm)
        else: terms.append(nm)
mf = pf.feols(f"y ~ {' + '.join(terms)} | prefecture_code + fm", data=gf,
              vcov={"CRV1": "prov_id"})
for lo, hi in CB:
    if (lo, hi) == (-6, -1): continue
    nm = f"c{lo}_{hi}H".replace("-", "m")
    dyn_rows.append(dict(series="civilFlow_filing", lo=lo, hi=hi,
                         est=mf.coef()[nm], se=mf.se()[nm]))
print("  dynamics [civilFlow_filing] done", flush=True)

dd = pd.DataFrame(dyn_rows)
dd.to_csv(f"{OUTD}/twoclock_dynamics.csv", index=False)

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True)
panels = [("enforceN_pros", "Violent-enforcement caseload (prosecution clock)",
           axes[0]), ("civilFlow_filing",
                      "Relational civil flow (filing clock, clean window)",
                      axes[1])]
for ser, ttl, ax in panels:
    s = dd[dd["series"] == ser].sort_values("lo")
    x = (s["lo"] + s["hi"]) / 2
    ax.errorbar(x, s["est"], yerr=1.96 * s["se"], fmt="o", capsize=3,
                color="#1f4e79")
    ax.axhline(0, lw=.8, color="grey"); ax.axvline(-3.5, lw=.8, ls="--",
                                                   color="grey")
    ax.set_title(ttl, fontsize=10); ax.set_ylabel("per SD of exposure")
axes[1].set_xlabel("months since inspection arrival (behavior clocks)")
fig.tight_layout()
os.makedirs(f"{OUTD}/figures", exist_ok=True)
fig.savefig(f"{OUTD}/figures/fig_twoclock.pdf")
plt.close(fig)
print("saved fig_twoclock.pdf", flush=True)

pd.DataFrame(rows).to_csv(f"{OUTD}/twoclock_results.csv", index=False)
if args.public_only:
    print("restricted judgment-text audit sample skipped (--public-only)", flush=True)
    print("step 73 public analysis complete", flush=True)
    raise SystemExit(0)

# ---------------- E: audit sample ------------------------------------------------
print("== E: audit sample (2018-09 re-scan) ==", flush=True)
BASE = str(_REP_JUDGMENTS)
p = f"{BASE}/2018_Court_Judgments_CSV/2018_MacroData_Court_Judgments_CSV/2018年09月裁判文书数据.csv"
D = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
PROS = re.compile(rf"{D}[^。]{{0,40}}?向本院提起公诉")
OFF = set(FAM)
snips = []
for ch in pd.read_csv(p, chunksize=60000, encoding="utf-8",
                      usecols=["案号", "案件类型", "案由", "全文"], dtype=str,
                      on_bad_lines="skip"):
    a = ch[(ch["案件类型"] == "刑事案件") & ch["案由"].isin(OFF)
           & ch["案号"].str.contains("刑初", na=False)]
    for _, r in a.iterrows():
        t = r["全文"] if isinstance(r["全文"], str) else ""
        mm = PROS.search(t[:3500])
        if mm:
            lo = max(0, mm.start() - 60)
            snips.append(dict(case_no=r["案号"], 案由=r["案由"], hit=1,
                              date=f"{mm.group(1)}-{mm.group(2)}-{mm.group(3)}",
                              snippet=t[lo:mm.end() + 40]))
        else:
            snips.append(dict(case_no=r["案号"], 案由=r["案由"], hit=0, date="",
                              snippet=t[:260]))
sn = pd.DataFrame(snips)
rng = np.random.default_rng(42)
aud = pd.concat([sn[sn["hit"] == 1].sample(min(200, (sn["hit"] == 1).sum()),
                                           random_state=42),
                 sn[sn["hit"] == 0].sample(min(100, (sn["hit"] == 0).sum()),
                                           random_state=42)])
aud.to_csv(f"{OUTD}/prosdate_audit_sample.csv", index=False, encoding="utf-8-sig")
print(f"audit sample: {len(aud)} rows (hit rate in month "
      f"{sn['hit'].mean():.3f})", flush=True)

print("step 73 complete", flush=True)
