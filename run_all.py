#!/usr/bin/env python3
"""Master runner for the Hit Where It Hurts replication package."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "analysis" / "code"
PIPELINE = ROOT / "config" / "pipeline.csv"
LOG_DIR = ROOT / "verification"


def portable_log_text(text: str) -> str:
    """Remove machine- and user-specific paths from retained run logs."""
    text = text.replace(str(ROOT), "<package-root>")
    text = text.replace(sys.executable, "python")
    text = re.sub(
        r"[A-Za-z]:(?:[/\\]+)Users(?:[/\\]+)[^\s,;]+",
        "<local-path>",
        text,
        flags=re.IGNORECASE,
    )
    return text


def resolve_script(value: str) -> Path:
    relative = Path(value)
    candidate = ROOT / relative if len(relative.parts) > 1 else CODE / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"pipeline script escapes package root: {value}") from error
    return resolved


def configure_restricted_paths(environment: dict[str, str]) -> None:
    config = ROOT / "config" / "restricted_paths.toml"
    if not config.exists():
        raise SystemExit(
            "restricted-full requires config/restricted_paths.toml and "
            "authorized source materials"
        )
    with config.open("rb") as handle:
        values = tomllib.load(handle)
    mappings = (
        ("judgments", "archive_root", "HWIH_JUDGMENT_ARCHIVE"),
        ("judgments", "case_level_archive", "HWIH_CASE_LEVEL_ARCHIVE"),
        ("mortality", "annual_volumes", "HWIH_CDC_SOURCE_ROOT"),
        ("registry", "named_bulk_extract", "HWIH_REGISTRY_ROOT"),
        ("baidu", "download_directory", "HWIH_BAIDU_ROOT"),
        ("interviews", "approved_coded_source", "HWIH_INTERVIEWS_ROOT"),
    )
    for section, key, variable in mappings:
        raw = values.get(section, {}).get(key)
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        environment[variable] = str(path.resolve())


def load_pipeline(profile: str) -> list[dict[str, str]]:
    with PIPELINE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows if profile in row["profiles"].split("|")
    ]
    return sorted(selected, key=lambda row: int(row["order"]))


def compile_paper(log) -> None:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex was requested but is not on PATH")
    paper = ROOT / "manuscript"
    source = "hwih_paper_aejep_submission.tex"
    for pass_number in (1, 2):
        command = [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            source,
        ]
        log.write(f"paper pass {pass_number}: {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=paper,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            raise RuntimeError(f"paper compilation failed on pass {pass_number}")


def sync_paper_figures(log) -> None:
    paper = ROOT / "manuscript"
    manuscript = (paper / "hwih_paper_aejep_submission.tex").read_text(
        encoding="utf-8-sig")
    names = {
        Path(value).name
        for value in re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", manuscript)
    }
    output_root = ROOT / "analysis" / "output"
    for name in sorted(names):
        candidates = sorted(
            output_root.rglob(name),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if not candidates:
            log.write(f"figure snapshot retained (no regenerated file): {name}\n")
            continue
        destination = paper / "figures" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], destination)
        log.write(f"figure synced: {candidates[0]} -> {destination}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("public-core", "public-full", "restricted-full"),
        default="public-core",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--compile-paper", action="store_true")
    args = parser.parse_args()

    rows = load_pipeline(args.profile)
    if args.list or args.dry_run:
        for row in rows:
            print(
                f"{int(row['order']):03d}  {row['script']:<42} "
                f"[{row['access']}] {row['description']}"
            )
        return 0

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{args.profile}_{stamp}.log"
    summary_path = log_path.with_suffix(".json")
    failures: list[str] = []
    records: list[dict[str, object]] = []
    run_started = datetime.now().astimezone()
    total_started = time.perf_counter()
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONHASHSEED"] = "0"
    environment["HWIH_REPLICATION"] = "1"
    if args.profile == "restricted-full":
        configure_restricted_paths(environment)

    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"profile={args.profile}\npython={sys.executable}\nroot={ROOT}\n")
        for row in rows:
            script = resolve_script(row["script"])
            if not script.exists():
                message = f"missing script: {script}"
                log.write(message + "\n")
                failures.append(row["script"])
                records.append({
                    "order": int(row["order"]),
                    "script": row["script"],
                    "returncode": None,
                    "seconds": 0.0,
                    "status": "missing",
                })
                if not args.continue_on_error:
                    break
                continue
            command = [
                sys.executable,
                str(script),
                *shlex.split(row.get("arguments", ""), posix=os.name != "nt"),
            ]
            print(f"[{row['order']}] {row['script']}", flush=True)
            log.write(f"\nRUN {' '.join(command)}\n")
            log.flush()
            step_started = time.perf_counter()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.perf_counter() - step_started
            records.append({
                "order": int(row["order"]),
                "script": row["script"],
                "returncode": result.returncode,
                "seconds": round(elapsed, 3),
                "status": "passed" if result.returncode == 0 else "failed",
            })
            log.write(f"ELAPSED {elapsed:.3f} seconds\n")
            if result.returncode:
                failures.append(row["script"])
                print(f"failed: {row['script']} (see {log_path})", file=sys.stderr)
                if not args.continue_on_error:
                    break
        if not failures and args.compile_paper:
            compile_started = time.perf_counter()
            sync_paper_figures(log)
            compile_paper(log)
            records.append({
                "order": 999,
                "script": "manuscript (two pdflatex passes)",
                "returncode": 0,
                "seconds": round(time.perf_counter() - compile_started, 3),
                "status": "passed",
            })

    log_path.write_text(
        portable_log_text(log_path.read_text(encoding="utf-8")),
        encoding="utf-8",
        newline="\n",
    )

    summary = {
        "profile": args.profile,
        "python": f"{Path(sys.executable).name} {platform.python_version()}",
        "root": ".",
        "started_at": run_started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "seconds": round(time.perf_counter() - total_started, 3),
        "status": "failed" if failures else "passed",
        "failures": failures,
        "steps": records,
        "log": str(log_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        print("failed scripts: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"completed {args.profile}; log: {log_path}; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
