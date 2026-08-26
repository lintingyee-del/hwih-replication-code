# Replication code for *From Private Enforcement to Public Courts*

This repository is the public code-only companion to *From Private Enforcement
to Public Courts: The Judicialization of Informal Markets* by Ruoran Lai,
Tingyi Lin, and Wanlin Lin. It contains the Python programs used to construct
the analysis files, estimate the paper's specifications, conduct inference and
robustness checks, and generate exhibits.

> **Data are not included.** This repository cannot reproduce the reported
> results on its own. A complete replication package will be deposited in a
> trusted data and code repository in accordance with the journal's data policy.
> The archived deposit, rather than this code-only repository, will be the
> version of record.

## Repository structure

- `analysis/code/` contains data construction, estimation, inference,
  robustness, and figure programs.
- `manuscript/figures/` contains the public figure-generation program retained
  with this snapshot.
- `run_all.py` is the master pipeline runner used by the complete package.
- `verify_package.py`, `verify_results.py`, and `verify_exhibits.py` implement
  package-integrity, numerical-result, and rendered-exhibit checks.
- `hz_figstyle.py` provides the common figure style.

## Running the code

The scripts expect the directory structure, pipeline configuration, and
analysis data supplied with the complete replication package. Those materials
are not part of this GitHub snapshot, so `run_all.py` is not executable from a
standalone checkout of this repository. The reference environment uses Python
3.11.5; exact dependencies and step-by-step commands will accompany the
archived package.

## Data availability

The study uses court judgments, central inspection timing, mortality
tabulations, business-registry aggregates, Baidu Index series, and aggregate
interview coding. Source judgment text, direct identifiers, interview
transcripts, platform credentials, and named firm records are not distributed
through GitHub. The archived replication package will include all
redistributable derived data and analysis code, together with provenance and
access documentation for source materials that cannot be redistributed.

## Status and citation

This is a submission-stage code snapshot and may be updated as the manuscript
develops. Final citation metadata, repository license, and a persistent DOI
will be supplied with the archived replication package.
