# -*- coding: utf-8 -*-
"""6B step 75 — parse PBoC provincial small-loan PDFs into a quarterly panel.

Line-based parser on the PDF text layer (validated on the 2025Q3 report; the
table-detector approach mangles these PDFs). Filename encodes the period:
一季度=Q1, 上半年=Q2, 前三季度=Q3, 年(only)=Q4.

Usage: python 75_pboc_panel.py [estimate]
Output: data/pboc_prov/pboc_prov_panel.csv (+ per-file QC prints)
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
import os, re, sys, io, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np

DATA = str(_REP_PROJECT / "data")
PDIR = f"{DATA}/pboc_prov"

PROV = ("北京|天津|河北|山西|内蒙古|辽宁|吉林|黑龙江|上海|江苏|浙江|安徽|福建|"
        "江西|山东|河南|湖北|湖南|广东|广西|海南|重庆|四川|贵州|云南|西藏|陕西|"
        "甘肃|青海|宁夏|新疆")
LINE = re.compile(rf"^((?:{PROV})[一-龥]*?|全国)\s+(\d+)\s+(\d+)\s+"
                  r"([\d,]+\.?\d*)\s+([\d,]+\.?\d*)\s*$")


def period_of(fn):
    y = re.search(r"(20\d\d)年", fn)
    if not y: return None
    y = y.group(1)
    if "一季度" in fn: q = 1
    elif "上半年" in fn: q = 2
    elif "前三季度" in fn: q = 3
    else: q = 4
    return f"{y}Q{q}"


def parse_pdf(fp):
    import pdfplumber
    rows = []
    with pdfplumber.open(fp) as pdf:
        for pg in pdf.pages:
            t = pg.extract_text() or ""
            for line in t.split("\n"):
                m = LINE.match(line.strip())
                if m:
                    rows.append(dict(
                        region=m.group(1), n_comp=int(m.group(2)),
                        staff=int(m.group(3)),
                        capital=float(m.group(4).replace(",", "")),
                        balance=float(m.group(5).replace(",", ""))))
    return rows


def main():
    out = []
    for fp in sorted(glob.glob(f"{PDIR}/**/*.pdf", recursive=True)):
        fn = os.path.basename(fp)
        per = period_of(fn)
        if per is None:
            print(f"[skip] {fn}: no period"); continue
        rows = parse_pdf(fp)
        provs = [r for r in rows if r["region"] != "全国"]
        nat = [r for r in rows if r["region"] == "全国"]
        chk = ""
        if nat and provs:
            sb = sum(r["balance"] for r in provs)
            gap = abs(sb - nat[0]["balance"]) / max(nat[0]["balance"], 1)
            chk = (f" nat_bal={nat[0]['balance']:.0f} sum={sb:.0f} "
                   f"gap={gap:.4f}" + ("  <-- CHECK" if gap > 0.005 else " OK"))
        print(f"[{per}] {fn}: {len(provs)} provinces{chk}")
        for r in provs:
            r["period"] = per
            out.append(r)
    # manual JPG transcriptions merged if present
    man_hits = glob.glob(f"{PDIR}/**/manual_jpg_periods.csv", recursive=True) \
        + glob.glob(f"{PDIR}/manual_jpg_periods.csv")
    man = man_hits[0] if man_hits else f"{PDIR}/manual_jpg_periods.csv"
    if os.path.exists(man):
        mm = pd.read_csv(man)
        print(f"[manual] {man}: {len(mm)} rows, periods "
              f"{sorted(mm['period'].unique())}")
        out += mm.to_dict("records")
    for r in out:
        r.setdefault("src", "pdf" if "staff" in r and r.get("staff") else "manual")
    # user-collected republication periods (short province names, no staff/capital)
    rep = str(_REP_PACKAGE / '转载来源小额贷款分省数据（4期）.csv').replace('\\', '/')
    if os.path.exists(rep):
        rr = pd.read_csv(rep)
        rr["period"] = rr["report"].map(period_of)
        rr = rr.rename(columns={"province": "region", "n_companies": "n_comp",
                                "loan_balance": "balance"})
        rr["src"] = "repub"
        print(f"[repub] {len(rr)} rows, periods {sorted(rr['period'].unique())}")
        for per in sorted(rr["period"].unique()):
            s = rr[rr["period"] == per]
            print(f"  [repub QC] {per}: {len(s)} provinces, "
                  f"sum_bal={s['balance'].sum():.0f}, sum_n={int(s['n_comp'].sum())}")
        out += rr[["region", "n_comp", "balance", "period", "src"]].to_dict("records")
    df = pd.DataFrame(out)
    df["key"] = df["region"].astype(str).str[:2]
    PRIO = {"pdf": 0, "manual": 1, "repub": 2}
    df["prio"] = df["src"].map(PRIO).fillna(1)
    df = (df.sort_values("prio").drop_duplicates(["period", "key"])
          .drop(columns=["prio"]))
    df.to_csv(f"{PDIR}/pboc_prov_panel.csv", index=False, encoding="utf-8-sig")
    per_list = sorted(df["period"].unique())
    print(f"\n[panel] {len(df)} rows, {len(per_list)} periods: {per_list}")
    ALL = [f"{y}Q{q}" for y in range(2014, 2021) for q in range(1, 5)]
    missing = [p for p in ALL if p not in per_list]
    print(f"[missing vs 2014Q1-2020Q4] {missing}")
    return df


def estimate(df):
    import pyfixest as pf
    from _wild import wild_p
    ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")
    xw = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")[
        ["prefecture_code", "province", "insp_month"]].drop_duplicates(
        "prefecture_code")
    hp = (xw.merge(ex[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
          .groupby("province")["exposure_v2_z"].mean().rename("H").reset_index())
    hp["key"] = hp["province"].str[:2]
    insp = xw[["province", "insp_month"]].drop_duplicates()
    insp["iy"] = insp["insp_month"].astype(str).str[:4].astype(int)
    insp["iq"] = ((insp["insp_month"].astype(str).str[5:7].astype(int) - 1) // 3
                  + 1)
    hp = hp.merge(insp, on="province")
    d = df.copy()
    if "key" not in d.columns:
        d["key"] = d["region"].astype(str).str[:2]
    d = d.merge(hp[["key", "province", "H", "iy", "iq"]], on="key", how="inner")
    d["y4"] = d["period"].str[:4].astype(int)
    d["q4"] = d["period"].str[5].astype(int)
    d["post"] = ((d["y4"] > d["iy"]) |
                 ((d["y4"] == d["iy"]) & (d["q4"] >= d["iq"]))).astype(int)
    d["px"] = d["post"] * d["H"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    print(f"\n[est] {d['province'].nunique()} provinces x "
          f"{d['period'].nunique()} periods, N={len(d)}")
    for y, lab in (("balance", "log loan balance"),
                   ("n_comp", "log #companies")):
        d["ly"] = np.log(d[y].clip(lower=0.1))
        m = pf.feols("ly ~ px | province + period", data=d,
                     vcov={"CRV1": "prov_id"})
        try: wp = wild_p("ly ~ px | province + period", d, "px")
        except Exception: wp = np.nan
        print(f"[est] {lab:18s} PostxH: {m.coef()['px']:+.4f} "
              f"({m.se()['px']:.4f}) CRV1 p={m.pvalue()['px']:.3f} "
              f"wild={wp:.3f} N={int(m._N)}")
        # implied 95% CI in percent per SD
        lo = m.coef()["px"] - 1.96 * m.se()["px"]
        hi = m.coef()["px"] + 1.96 * m.se()["px"]
        print(f"      95% CI per SD of H: [{100*lo:+.1f}%, {100*hi:+.1f}%]")


if __name__ == "__main__":
    df = main()
    if len(sys.argv) > 1 and sys.argv[1] == "estimate":
        estimate(df)
