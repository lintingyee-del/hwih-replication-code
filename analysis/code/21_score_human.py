# -*- coding: utf-8 -*-
"""Score the human double-coding sheet vs the LLM gold labels.

Accepts the friend-facing sheet 双盲编码表_请填写.xlsx (columns 案号 / 人工标签 /
_task_内部勿删) OR the original double_coding_sheet.xlsx (case_no / human_label /
task). Reports per-indicator and pooled agreement + Cohen's kappa, and writes
tab_humanval.tex for Appendix B.
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
import pandas as pd, numpy as np, json, glob, os

PKT = str(_REP_PROJECT / 'docs' / 'human_coding_packet').replace('\\', '/')
VAL = str(_REP_PROJECT / "output" / "validation")
TABD = str(_REP_PROJECT / "output" / "tables")
GOLD = {"civ_rel_txn": "gold_rel_txn", "civ_rel_fail": "gold_rel_fail",
        "civ_evidence": "gold_iou", "crim_backstop": "gold_backstop",
        "crim_fraudsplit": "gold_telecom"}
LABEL = {"civ_rel_txn": "Relational transaction", "civ_rel_fail": "Relational failure",
         "civ_evidence": "IOU documented", "crim_backstop": "Hard backstop",
         "crim_fraudsplit": "Telecom fraud"}

def load_sheet():
    for name in ["双盲编码表_请填写.xlsx", "双盲编码表_请填写.csv", "double_coding_sheet.xlsx"]:
        p = os.path.join(PKT, name)
        if not os.path.exists(p): continue
        df = pd.read_csv(p, encoding="utf-8-sig") if p.endswith(".csv") else pd.read_excel(p)
        df = df.rename(columns={"案号": "case_no", "人工标签": "human_label",
                                "_task_内部勿删": "task"})
        if "human_label" in df.columns and df["human_label"].notna().any():
            return df, name
    return None, None

def kappa(h, g):
    m = pd.DataFrame({"h": h, "g": g}).dropna()
    m = m[m["h"].isin([0, 1]) & m["g"].isin([0, 1])]
    if len(m) == 0: return np.nan, np.nan, 0
    po = (m["h"] == m["g"]).mean()
    ph, pg = (m["h"] == 1).mean(), (m["g"] == 1).mean()
    pe = ph * pg + (1 - ph) * (1 - pg)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return po, k, len(m)

df, src = load_sheet()
if df is None:
    print("人工标签列还是空的——请先让编码者填写 双盲编码表_请填写.xlsx 的『人工标签』列。")
    raise SystemExit(0)
print(f"读取：{src}")

def norm(v):
    s = str(v).strip().lower()
    return 1 if s in ("1", "1.0") else 0 if s in ("0", "0.0") else np.nan
df["h"] = df["human_label"].map(norm)

rows, pooled_h, pooled_g = [], [], []
for task, g in GOLD.items():
    labs = []
    for f in glob.glob(f"{VAL}/labels_{task}_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                try: labs.append(json.loads(line))
                except json.JSONDecodeError: pass
    lb = pd.DataFrame(labs).drop_duplicates("case_no")[["case_no", g]]
    sub = df[df["task"] == task].merge(lb, on="case_no")
    po, k, n = kappa(sub["h"], sub[g])
    rows.append((LABEL[task], po, k, n))
    v = sub.dropna(subset=["h"]); v = v[v["h"].isin([0, 1]) & v[g].isin([0, 1])]
    pooled_h += list(v["h"]); pooled_g += list(v[g])
    print(f"{LABEL[task]:22s} n={n:3d}  agreement={po:.3f}  kappa={k:.3f}"
          if n else f"{LABEL[task]:22s} (未填)")

po, k, n = kappa(pd.Series(pooled_h), pd.Series(pooled_g))
rows.append(("Pooled", po, k, n))
print(f"{'Pooled':22s} n={n:3d}  agreement={po:.3f}  kappa={k:.3f}")

L = [r"\begin{tabular}{lccc}", r"\toprule",
     r"Indicator & Agreement & Cohen's $\kappa$ & $n$ \\ \midrule"]
for lab, po, k, n in rows:
    if lab == "Pooled": L.append(r"\midrule")
    f = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    L.append(f"{lab} & {f(po)} & {f(k)} & {n} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_humanval.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("→ tab_humanval.tex 已生成，可接入附录 B。")
