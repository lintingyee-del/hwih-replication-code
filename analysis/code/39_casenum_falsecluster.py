# -*- coding: utf-8 -*-
"""6B step 39 — VALIDATE the case-number quasi-signature BEFORE using it. Batch-filed
professional lending -> a run of consecutive first-instance civil case numbers
(民初 seq) that are all 民间借贷, within one court-year. Question: how often does such a
run arise by CHANCE (false cluster) rather than a real batch?

Method: within (court, year), let the 民间借贷 first-instance cases have sequence
numbers with marginal density p = M / (max_seq - min_seq + 1) (p = 民间借贷 share of the
first-instance docket, inferred from the seq range so we don't need the full docket).
Under random placement, expected number of runs of length >= k in a length-T sequence is
~ T * p^k * (1-p). Compare to OBSERVED runs (maximal gap=1 chains of 民间借贷 seq nums).
false-cluster rate(>=k) = null_expected(>=k) / observed(>=k). Report by k.
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
import re, numpy as np, pandas as pd
DATA = str(_REP_PROJECT / "data")
cc = pd.read_parquet(f"{DATA}/civil_case.parquet", columns=["case_no","cause","jmonth"])

PAT = re.compile(r"）(.+?)民[一二三四]?(初|终|再)[^\d]*(\d+)号")
def parse(cn):
    if not isinstance(cn,str): return None
    y = re.search(r"(\d{4})", cn); m = PAT.search(cn)
    if not (y and m): return None
    return (m.group(1), int(y.group(1)), m.group(2), int(m.group(3)))  # court, year, div, seq

p = cc["case_no"].map(parse)
ok = p.notna()
print(f"[parse] case_no parsed: {ok.mean():.3f} ({ok.sum():,}/{len(cc):,})", flush=True)
d = cc[ok].copy()
d[["court","year","div","seq"]] = pd.DataFrame(p[ok].tolist(), index=d[ok].index)
d = d[d["div"]=="初"].copy()            # first instance only
d["lend"] = (d["cause"]=="民间借贷纠纷")
print(f"[first-instance] {len(d):,}; 民间借贷 among them: {d['lend'].mean():.3f}", flush=True)

obs = {k:0 for k in (2,3,4,5,10)}; nul = {k:0 for k in (2,3,4,5,10)}
ncy = 0
for (ct,yr), g in d.groupby(["court","year"]):
    seqs = np.sort(g.loc[g["lend"],"seq"].unique())
    if len(seqs) < 5: continue
    lo, hi = seqs.min(), seqs.max(); T = hi-lo+1
    if T < 20: continue
    p_dens = len(seqs)/T
    if p_dens <= 0 or p_dens >= 1: continue
    ncy += 1
    # observed maximal consecutive runs
    runs=[]; cur=1
    for i in range(1,len(seqs)):
        if seqs[i]==seqs[i-1]+1: cur+=1
        else: runs.append(cur); cur=1
    runs.append(cur); runs=np.array(runs)
    for k in obs:
        obs[k]+=int((runs>=k).sum())
        nul[k]+=T*(p_dens**k)*(1-p_dens)      # expected runs >=k under random
print(f"\n[court-years used] {ncy}", flush=True)
print(f"{'k':>3} {'observed':>10} {'null(chance)':>13} {'false-cluster rate':>20}", flush=True)
for k in (2,3,4,5,10):
    fr = nul[k]/obs[k] if obs[k]>0 else float('nan')
    print(f"{k:>3} {obs[k]:>10,} {nul[k]:>13.1f} {fr:>19.3f}", flush=True)
print("\n[read] false-cluster rate = fraction of runs>=k expected by chance. Low rate at "
      "some k => runs>=k are reliable batch markers; high everywhere => signal too noisy.", flush=True)
