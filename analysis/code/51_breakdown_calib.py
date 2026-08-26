# -*- coding: utf-8 -*-
"""6B step 51 — breakdown calibration for the content-selection attack on M1,
plus TOST-style equivalence readout of the composition gates.

Logic: measured coercive share m = sV/(sV+(1-s)) if a fraction delta of
violent-CONTENT cases is differentially unpublished per SD of H (V=1-delta).
dm/ddelta at 0 = -s(1-s). The delta* needed to fully generate the observed
long-horizon coefficient is |beta_long| / (s(1-s)). That much violent-only
suppression would also shift the released docket's violent-offense MIX by
v(1-v)*delta*, which the composition gate bounds directly. Verdict: does the
gate's 95% CI exclude the mix shift that worst-case suppression implies?
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
import duckdb, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = str(_REP_PROJECT / "output" / "ext2124")
con = duckdb.connect()

pan = con.sql(f"SELECT * FROM '{OUT}/persist_panel.parquet'").df()
post = pan[pan.yr.isin([2021, 2022, 2023, 2024])]
w = post["n_target_fact"]
s = float(np.average(post["sh_coercive"].fillna(0), weights=w))          # coercive share
v = float((post["n_violenf"].sum()) / max(post["n_target"].sum(), 1))    # violent-offense mix

es = pd.read_csv(f"{OUT}/persist_es.csv")
main = es[(es.spec == "coercive_share") & (es.bin_lo >= 42)]
beta_long = float(np.average(main["est"], weights=1 / main["se"] ** 2))

delta_star = abs(beta_long) / (s * (1 - s))
mix_shift_implied = v * (1 - v) * delta_star

g = pd.read_csv(f"{OUT}/gates_composition.csv")
gate = g[(g.horizon == "2021-24") & (g.attribute == "violent_offense_mix")].iloc[0]
ci_lo, ci_hi = gate.beta_H - 1.96 * gate.se, gate.beta_H + 1.96 * gate.se
excluded = mix_shift_implied > max(abs(ci_lo), abs(ci_hi))

print(f"post-2020 coercive share s = {s:.4f}; violent-offense mix v = {v:.4f}")
print(f"observed long-horizon (+42..+66) coefficient = {beta_long:.5f} per SD H")
print(f"breakdown delta* (violent-only unpublication per SD H) = {delta_star:.4f}")
print(f"implied violent-mix shift under worst case = {mix_shift_implied:.4f}")
print(f"composition gate (2021-24): beta = {gate.beta_H:.4f}, "
      f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"WORST-CASE SUPPRESSION {'EXCLUDED' if excluded else 'NOT excluded'} by the gate")

# TOST-style equivalence: for each gate attribute, the smallest symmetric bound
# under which both one-sided tests reject at 5% (90% CI width), i.e. the
# tightest equivalence margin the data certify.
rows = []
for _, r in g[g.horizon.isin(["2022-23", "2021-24"])].iterrows():
    bound90 = abs(r.beta_H) + 1.645 * r.se
    rows.append(dict(horizon=r.horizon, attribute=r.attribute, beta=r.beta_H,
                     se=r.se, tost_margin_5pct=bound90))
t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/gates_tost.csv", index=False)
print()
print("TOST margins (smallest |bound| certified at 5%):")
print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
