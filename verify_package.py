#!/usr/bin/env python3
"""Check integrity, portability, disclosure controls, and exhibit coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "documentation" / "file_manifest.csv"
CROSSWALK = ROOT / "documentation" / "exhibit_crosswalk.csv"
PIPELINE = ROOT / "config" / "pipeline.csv"
PROHIBITED_NAMES = {
    "baidu_api_token.txt",
    "baidu_cookie.txt",
    "quote_bank.json",
    "interview_quote_bank_full.md",
    "prosdate_audit_sample.csv",
}
INTERNAL_ANALYSIS = "".join(("xian", "zhu"))
INTERNAL_JOURNAL = "".join(("j", "le"))
INTERNAL_PROJECT = "6b_" + INTERNAL_JOURNAL
INTERNAL_INSTITUTION = "".join(("boo", "th"))
INTERNAL_SUBMISSION = INTERNAL_INSTITUTION + "_lipics_submission_anonymous"
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".csv", ".json", ".tex", ".toml", ".yml", ".yaml",
    ".aux", ".bbl", ".blg", ".log", ".out",
}
PROHIBITED_TEXT = {
    "personal C-drive path": re.compile(
        r"C:(?:[/\\]+)Users(?:[/\\]+)86131", re.I),
    "private source drive path": re.compile(
        r"E:(?:[/\\]+)(?:6b_|Court_Judgments|as1593)", re.I),
    "internal analysis label": re.compile(
        rf"\b{re.escape(INTERNAL_ANALYSIS)}\b", re.I),
    "internal project label": re.compile(
        re.escape(INTERNAL_PROJECT).replace("_", r"(?:_|\\_)")
        + rf"|\b6b\s+{re.escape(INTERNAL_JOURNAL)}\b"
        + rf"|\b{re.escape(INTERNAL_JOURNAL)}\b",
        re.I,
    ),
    "internal submission label": re.compile(
        rf"(?:{re.escape(INTERNAL_SUBMISSION)}|"
        rf"\b{re.escape(INTERNAL_INSTITUTION)}\b)",
        re.I,
    ),
    "credential marker": re.compile(
        r"(?:baidu_cookie|baidu_api_token)\s*=", re.I),
}
PROFILE_NAMES = {"public-core", "public-full", "restricted-full"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(errors: list[str], strict: bool) -> None:
    if not MANIFEST.exists():
        errors.append("missing documentation/file_manifest.csv")
        return
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    listed = set()
    for row in rows:
        relative = row["path"]
        listed.add(relative)
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"manifest entry missing: {relative}")
            continue
        if int(row["bytes"]) != path.stat().st_size:
            errors.append(f"size mismatch: {relative}")
        if row["sha256"] != sha256(path):
            errors.append(f"hash mismatch: {relative}")
    if strict:
        ignored_prefixes = (
            "verification/", "__pycache__/", "analysis/code/__pycache__/")
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file()
            and path != MANIFEST
            and path.suffix != ".pyc"
            and not path.relative_to(ROOT).as_posix().startswith(ignored_prefixes)
        }
        for relative in sorted(actual - listed):
            errors.append(f"unmanifested file: {relative}")


def check_names_and_text(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.name.lower() in PROHIBITED_NAMES:
            errors.append(f"prohibited file: {relative}")
        for label, pattern in PROHIBITED_TEXT.items():
            if pattern.search(relative):
                errors.append(f"{label} in path: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20_000_000:
            continue
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for label, pattern in PROHIBITED_TEXT.items():
            if pattern.search(content):
                errors.append(f"{label}: {relative}")


def check_python_syntax(errors: list[str]) -> None:
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, relative, "exec")
        except (OSError, SyntaxError, UnicodeError) as error:
            errors.append(f"Python syntax/read failure: {relative}: {error}")


def valid_docket(value: object) -> bool:
    text = str(value)
    return bool(
        text.startswith("CASE_")
        or re.fullmatch(
            r"（20\d{2}）C[0-9A-F]{10}(?:民(?:[一二三四])?初|刑初)\d+号",
            text,
        )
    )


def check_pseudonymized_data(errors: list[str]) -> None:
    roots = (
        ROOT / "analysis" / "data",
        ROOT / "analysis" / "output" / "validation",
    )
    files = sorted({
        path for base in roots if base.exists() for path in base.rglob("*.parquet")
    })
    core = {
        ROOT / "analysis/data/case_clean.parquet",
        ROOT / "analysis/data/civil_case.parquet",
    }
    for path in sorted(core):
        if not path.exists():
            errors.append(f"missing core dataset: {path.relative_to(ROOT)}")
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if "filing_raw" in names:
            errors.append(f"raw filing text column retained: {relative}")
        columns = [
            name for name in (
                "案号", "case_no", "doc_id", "case_family_id", "raw_text_path",
                "lawyer_keys", "crim_cites", "civ_cites", "parties",
            ) if name in names
        ]
        if not columns:
            continue
        for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
            table = batch.to_pydict()
            for column in ("案号", "case_no"):
                bad = [
                    value for value in table.get(column, [])
                    if value is not None and not valid_docket(value)
                ]
                if bad:
                    errors.append(f"unmasked docket value in {relative}")
                    break
            for column, prefix in (
                ("doc_id", "DOC_ID_"),
                ("case_family_id", "CASE_FAMILY_ID_"),
            ):
                if any(
                    value is not None and not str(value).startswith(prefix)
                    for value in table.get(column, [])
                ):
                    errors.append(f"unmasked {column} in {relative}")
            if any(value for value in table.get("raw_text_path", [])):
                errors.append(f"source path retained in {relative}")
            if any(
                token and not token.startswith("LAW_")
                for value in table.get("lawyer_keys", []) if value is not None
                for token in str(value).split(";")
            ):
                errors.append(f"unmasked lawyer key in {relative}")
            for column in ("crim_cites", "civ_cites"):
                if any(
                    token and not valid_docket(token)
                    for value in table.get(column, []) if value is not None
                    for token in str(value).split(";")
                ):
                    errors.append(f"unmasked cited docket in {relative}")
            if any(
                value is not None
                and not str(value).startswith(("某_", "公司_", "个人_"))
                for value in table.get("parties", [])
            ):
                errors.append(f"named party value retained in {relative}")
            if any(message.endswith(relative) for message in errors[-8:]):
                break


def check_registry_release(errors: list[str]) -> None:
    path = ROOT / "analysis" / "data" / "derived" / "registry_hits_deidentified.csv"
    if not path.exists():
        errors.append("missing deidentified registry hit file")
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"企业名称", "snippet", "bank_context", "name_hit"}
        if not required <= set(reader.fieldnames or []):
            errors.append("deidentified registry hit file lacks release-safe columns")
            return
        for row_number, row in enumerate(reader, start=2):
            name = row.get("企业名称", "")
            if name and not name.startswith("FIRM_"):
                errors.append(f"unmasked firm name at registry row {row_number}")
                return
            if row.get("snippet", "") or row.get("matched_field", ""):
                errors.append(f"registry text retained at row {row_number}")
                return
            if row.get("bank_context", "").lower() not in {"true", "false"}:
                errors.append(f"invalid released bank_context at row {row_number}")
                return
            if row.get("name_hit", "").lower() not in {"true", "false"}:
                errors.append(f"invalid released name_hit at row {row_number}")
                return


def resolve_program(value: str) -> Path:
    relative = Path(value)
    if len(relative.parts) > 1:
        return ROOT / relative
    return ROOT / "analysis" / "code" / relative


def check_pipeline(errors: list[str]) -> None:
    if not PIPELINE.exists():
        errors.append("missing config/pipeline.csv")
        return
    with PIPELINE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    orders: set[int] = set()
    for row in rows:
        try:
            order = int(row["order"])
        except (KeyError, ValueError):
            errors.append(f"invalid pipeline order: {row}")
            continue
        if order in orders:
            errors.append(f"duplicate pipeline order: {order}")
        orders.add(order)
        profiles = set(row.get("profiles", "").split("|"))
        if not profiles or not profiles <= PROFILE_NAMES:
            errors.append(f"invalid pipeline profiles at order {order}")
        script = resolve_program(row.get("script", ""))
        if not script.is_file():
            errors.append(
                f"pipeline program missing: {row.get('script', '')}")
        if profiles & {"public-core", "public-full"} and row.get("access") != "public":
            errors.append(f"restricted step exposed in public pipeline: {order}")


def check_crosswalk(errors: list[str]) -> None:
    if not CROSSWALK.exists():
        errors.append("missing documentation/exhibit_crosswalk.csv")
        return
    paper_dir = ROOT / "manuscript"
    manuscripts = [
        paper_dir / "hwih_paper_aejep_submission.tex",
        paper_dir / "criminal_specification_appendix.tex",
    ]
    labels: set[str] = set()
    for path in manuscripts:
        if not path.exists():
            errors.append(f"missing manuscript source: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        labels.update(re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", text))
    with CROSSWALK.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    covered = " ".join(row.get("paper_exhibit", "") for row in rows)
    for label in sorted(labels):
        if not re.search(rf"(?<![A-Za-z0-9_:]){re.escape(label)}(?![A-Za-z0-9_:])", covered):
            errors.append(f"paper exhibit absent from crosswalk: {label}")
    for row in rows:
        for program in filter(None, map(str.strip, row.get("principal_programs", "").split(";"))):
            if not resolve_program(program).is_file():
                errors.append(f"crosswalk program missing: {program}")
        for output in filter(None, map(str.strip, row.get("principal_outputs", "").split(";"))):
            if not (ROOT / output).exists():
                errors.append(f"crosswalk output missing: {output}")


def check_build_info(errors: list[str]) -> None:
    path = ROOT / "documentation" / "build_info.json"
    if not path.exists():
        errors.append("missing documentation/build_info.json")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid build_info.json: {error}")
        return
    if payload.get("package_schema_version") != 3:
        errors.append("unexpected package schema version")
    if payload.get("sanitization_version") != "2026-08-22-v2":
        errors.append("unexpected or missing sanitization version")
    if not re.fullmatch(r"[0-9a-f]{40}", payload.get("source_git_commit", "")):
        errors.append("missing or invalid source git commit in build info")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--skip-manifest", action="store_true",
        help="Run portability and disclosure checks after analysis outputs changed")
    args = parser.parse_args()
    errors: list[str] = []
    if not args.skip_manifest:
        check_manifest(errors, args.strict)
    check_names_and_text(errors)
    check_python_syntax(errors)
    check_pseudonymized_data(errors)
    check_registry_release(errors)
    check_pipeline(errors)
    check_crosswalk(errors)
    check_build_info(errors)
    if errors:
        print("PACKAGE CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PACKAGE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
