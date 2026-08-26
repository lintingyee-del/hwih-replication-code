# -*- coding: utf-8 -*-
"""6B step 10 — v2 tables: civil judicialization + criminal v2, numbers macros."""

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
import pandas as pd, numpy as np, os

OUTD = str(_REP_PROJECT / "output")
TABD = f"{OUTD}/tables"
r = pd.read_csv(f"{OUTD}/results_v2.csv").set_index("tag")
es = pd.read_csv(f"{OUTD}/eventstudy_v2.csv")
cflow = pd.read_csv(str(_REP_PROJECT / 'data' / 'civil_flow.csv').replace('\\', '/'))

# Benjamini-Hochberg q-values within the civil outcome family.
FAMILIES = {"civil": ["C1_flow_asinh","C2_orig_iou","C2_orig_transfer","C3_rate",
                      "C5_rel_share","C6_mediated","C6_backstop_collect"]}
r["bh_q"] = np.nan
for fam, tags in FAMILIES.items():
    tags = [t for t in tags if t in r.index]
    pv = r.loc[tags, "wild_p"].values
    order = np.argsort(pv); m = len(pv)
    q = pv[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(q, 0, 1)
    r.loc[tags, "bh_q"] = out

# The five criminal headline outcomes use the balanced any-case extensive margin,
# not the positive-cell count intensity stored in results_v2.csv.
headline = pd.read_csv(f"{OUTD}/criminal_balanced_headline.csv").iloc[0]
criminal_tags = ["K2_market_backstop", "K2_market_relfail",
                 "K2_market_formalization", "K2_enforcement_detentiondebt"]
criminal_p = np.r_[r.loc[criminal_tags, "wild_p"].to_numpy(float),
                   float(headline["p_wild"])]
order = np.argsort(criminal_p); m = len(criminal_p)
criminal_q = criminal_p[order] * m / (np.arange(m) + 1)
criminal_q = np.minimum.accumulate(criminal_q[::-1])[::-1]
out = np.empty(m); out[order] = np.clip(criminal_q, 0, 1)
r.loc[criminal_tags, "bh_q"] = out[:-1]
headline_q = float(out[-1])

def cell(tag):
    row = r.loc[tag]
    wp = f"{row['wild_p']:.3f}" if not np.isnan(row["wild_p"]) else "--"
    q = f"{row['bh_q']:.3f}" if not np.isnan(row["bh_q"]) else "--"
    return f"{row['est']:.4f} & ({row['se']:.4f}) & {wp} & {q} & {int(row['n']):,}"

mac = []
def num(name, val): mac.append(f"\\newcommand{{\\{name}}}{{{val}}}")
for tag, m in [("C1_flow_asinh", "CivFlow"), ("C2_orig_iou", "CivOrigIou"),
               ("C2_orig_transfer", "CivOrigTransfer"), ("C3_rate", "CivRate"),
               ("C5_rel_share", "CivRelShare"), ("C6_mediated", "CivMediated"),
               ("C6_backstop_collect", "CivBackstopCollect"),
               ("K2_market_backstop", "KMarketBackstop"),
               ("K2_market_relfail", "KMarketRelFail"),
               ("K2_market_formalization", "KMarketFormalization"),
               ("K2_enforcement_detentiondebt", "KDetentionDebt"),
               ("K2_enforcement_asinhN", "KEnforceN"),
               ("K2_violence_backstop", "KViolenceBackstopB"),
               ("K2_theft_backstop", "KTheftBackstopB"),
               ("S1_civil_flow_dose", "StackedCivFlow"),
               ("S1_civil_flow_binary", "StackedCivFlowBin"),
               ("S1_crim_backstop_dose", "StackedCrimBackstop"),
               ("C2v2_orig_iou", "CivOrigIouV"),
               ("C2v2_orig_transfer", "CivOrigTransferV"),
               ("C3v2_rate", "CivRateV"),
               ("C3v2_rate_relational", "CivRateRelV"),
               ("C6v2_orig_volume", "CivOrigVolV")]:
    if tag in r.index:
        row = r.loc[tag]
        num(m, f"{row['est']:.3f}"); num(m+"SE", f"{row['se']:.3f}")
        num(m+"P", f"{row['p']:.3f}")
        num(m+"WildP", f"{row['wild_p']:.3f}" if not np.isnan(row['wild_p']) else "--")
        num(m+"N", f"{int(row['n']):,}")
num("CivPretrend", f"{es['pretrend_p'].iloc[0]:.3f}")
for _, row in cflow.iterrows():
    key = row["step"].split()[0].replace("0","Zero").replace("1","One").replace("2","Two") \
        .replace("3","Three").replace("4","Four").replace("a","A").replace("b","B")
    num("Flow"+key, f"{int(row['n']):,}")
with open(f"{TABD}/numbers_v2.tex", "w", encoding="utf-8") as f:
    f.write("% auto-generated by 10_tables_v2.py\n" + "\n".join(mac) + "\n")

# ---- civil main table ----
L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r"Outcome & Coefficient & (CRV1 SE) & Wild $p$ & BH $q$ & $N$ \\ \midrule",
     r"\multicolumn{6}{l}{\emph{Panel A. Litigation flow and margins (triple difference vs.\ traffic placebo)}} \\[2pt]",
     f"C1: asinh(cases), relational causes & {cell('C1_flow_asinh')} \\\\",
     f"C6: mediated share & {cell('C6_mediated')} \\\\",
     f"Backstop-collection residue & {cell('C6_backstop_collect')} \\\\",
     r"\midrule",
     r"\multicolumn{6}{l}{\emph{Panel B. Lending cases, Post$\times$Exposure (dose)}} \\[2pt]",
     f"C5: relational share of litigated disputes & {cell('C5_rel_share')} \\\\",
     f"C3: monthly interest rate (\\%, $\\le$2020-07) & {cell('C3_rate')} \\\\",
     r"\midrule",
     r"\multicolumn{6}{l}{\emph{Panel C. Origination cohorts (contract-date timing)}} \\[2pt]",
     f"C2: IOU documented, new loans & {cell('C2_orig_iou')} \\\\",
     f"C2: transfer records, new loans & {cell('C2_orig_transfer')} \\\\",
     r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_civil.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---- criminal v2 table ----
L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r"Outcome & Coefficient & (CRV1 SE) & Wild $p$ & BH $q$ & $N$ \\ \midrule",
     f"Market crimes: hard backstop & {cell('K2_market_backstop')} \\\\",
     f"Market crimes: relational failure & {cell('K2_market_relfail')} \\\\",
     f"Market crimes: formalization language & {cell('K2_market_formalization')} \\\\",
     f"Enforcement crimes: detention$\\times$debt & {cell('K2_enforcement_detentiondebt')} \\\\",
     ("Enforcement-related: any recorded case & "
      f"{headline['coefficient']:.4f} & ({headline['std_error_crv1']:.4f}) & "
      f"{headline['p_wild']:.3f} & {headline_q:.3f} & {int(headline['n_fit']):,} \\\\"),
     r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_crim_v2.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---- civil event study table ----
L = [r"\begin{tabular}{lcc}", r"\toprule",
     r"Event-time bin & Estimate & (SE) \\ \midrule"]
for _, row in es.iterrows():
    ref = " (ref.)" if (row["bin_lo"], row["bin_hi"]) == (-6, -1) else ""
    L.append(f"{{[{int(row['bin_lo'])},{int(row['bin_hi'])}]}}{ref} & "
             f"{row['est']:.4f} & ({row['se']:.4f}) \\\\")
L += [r"\midrule", f"Pretrend joint $p$ & \\multicolumn{{2}}{{c}}{{{es['pretrend_p'].iloc[0]:.3f}}} \\\\",
      r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_civil_es.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---- civil sample flow ----
L = [r"\begin{tabular}{lr}", r"\toprule", r"Step & Documents \\ \midrule"]
for _, row in cflow.iterrows():
    L.append(f"{row['step']} & {int(row['n']):,} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_civil_flow.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("v2 tables written")
