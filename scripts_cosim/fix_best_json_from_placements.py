#!/usr/bin/env python3
"""
Fix best.json using placements/placements.jsonl as ground truth.

Scans a base directory (default: run_queue_big) for dataset directories matching:
  **/ds_*/placements/placements.jsonl

For each dataset:
  - Compute min RTT from placements.jsonl
  - If best.json is missing, has different RTT, or points to a wrong/missing result file, rewrite best.json
  - Optionally backup existing best.json -> best.json.bak

Notes:
  - This script does not try to regenerate missing simulation_*.json files.
  - The 'file' field is set to a real file only when we can confidently match RTT.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _approx_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def _min_rtt_from_placements(jsonl_path: Path) -> Tuple[Optional[float], Optional[Dict[str, Any]], int]:
    best_rtt: Optional[float] = None
    best_row: Optional[Dict[str, Any]] = None
    n_lines = 0

    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            rtt = row.get("rtt", None)
            if rtt is None:
                continue
            try:
                rtt_f = float(rtt)
            except Exception:
                continue
            if not math.isfinite(rtt_f):
                continue
            if best_rtt is None or rtt_f < best_rtt:
                best_rtt = rtt_f
                best_row = row

    return best_rtt, best_row, n_lines


def _rtt_from_result_json(result_path: Path) -> Optional[float]:
    try:
        data = json.loads(result_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    stats = data.get("stats", {})
    if not isinstance(stats, dict):
        return None
    task_results = stats.get("taskResults", [])
    if not isinstance(task_results, list):
        return None

    total = 0.0
    counted = 0
    for tr in task_results:
        if not isinstance(tr, dict):
            continue
        task_id = tr.get("taskId", None)
        if isinstance(task_id, int) and task_id >= 0:
            try:
                total += float(tr.get("elapsedTime", 0.0) or 0.0)
            except Exception:
                total += 0.0
            counted += 1

    return total if counted > 0 else None


def _choose_best_file(ds_dir: Path, min_rtt: float) -> Optional[str]:
    """
    Try to set best.json["file"] to something that exists and matches min_rtt.
    Priority:
      1) optimal_result.json
      2) existing best.json's referenced file (if exists and matches)
    Otherwise returns None.
    """
    opt = ds_dir / "optimal_result.json"
    if opt.exists():
        r = _rtt_from_result_json(opt)
        if r is not None and _approx_equal(r, min_rtt):
            return "optimal_result.json"

    bj = ds_dir / "best.json"
    if bj.exists():
        try:
            cur = json.loads(bj.read_text())
        except Exception:
            cur = {}
        if isinstance(cur, dict):
            f = cur.get("file")
            if isinstance(f, str) and f:
                cand = ds_dir / f
                if cand.exists():
                    r = _rtt_from_result_json(cand)
                    if r is not None and _approx_equal(r, min_rtt):
                        return f

    return None


def _current_best_file(existing: Dict[str, Any]) -> Optional[str]:
    f = existing.get("file")
    if isinstance(f, str) and f.strip():
        return f.strip()
    return None


def _best_file_matches(ds_dir: Path, file_name: str, min_rtt: float) -> bool:
    cand = ds_dir / file_name
    if not cand.exists():
        return False
    r = _rtt_from_result_json(cand)
    return r is not None and _approx_equal(r, min_rtt)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix best.json from placements.jsonl")
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("/root/projects/my-herosim/simulation_data/artifacts/run_queue_big"),
        help="Directory to scan (default: run_queue_big)",
    )
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    ap.add_argument("--backup", action="store_true", help="Backup best.json -> best.json.bak before overwriting")
    ap.add_argument("--eps", type=float, default=1e-9, help="Equality tolerance for existing RTT check")
    args = ap.parse_args()

    base: Path = args.base
    if not base.exists():
        raise SystemExit(f"Base directory not found: {base}")

    placements_files = sorted(base.rglob("ds_*/placements/placements.jsonl"))

    scanned = 0
    changed = 0
    skipped_empty = 0
    skipped_no_best_change = 0

    for pjsonl in placements_files:
        ds_dir = pjsonl.parent.parent
        best_json = ds_dir / "best.json"

        min_rtt, best_row, n_lines = _min_rtt_from_placements(pjsonl)
        if min_rtt is None:
            skipped_empty += 1
            continue

        scanned += 1

        existing: Dict[str, Any] = {}
        existing_rtt: Optional[float] = None
        if best_json.exists():
            try:
                loaded = json.loads(best_json.read_text())
                if isinstance(loaded, dict):
                    existing = loaded
                    if "rtt" in existing:
                        existing_rtt = float(existing.get("rtt"))
            except Exception:
                existing = {}
                existing_rtt = None

        best_file = _choose_best_file(ds_dir, min_rtt)
        existing_file = _current_best_file(existing) if existing else None

        rtt_ok = existing_rtt is not None and abs(existing_rtt - min_rtt) <= float(args.eps)
        # file is "ok" if it is exactly what we'd choose, OR both are None/empty.
        if best_file is None:
            file_ok = existing_file is None or existing_file == ""
        else:
            file_ok = (existing_file == best_file) and _best_file_matches(ds_dir, best_file, min_rtt)

        if rtt_ok and file_ok:
            skipped_no_best_change += 1
            continue

        out: Dict[str, Any] = {
            "rtt": float(min_rtt),
            "file": best_file,
            "source": "placements/placements.jsonl",
            "updated_by": "scripts_cosim/fix_best_json_from_placements.py",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "placements_lines": int(n_lines),
        }
        if isinstance(best_row, dict) and "placement_plan" in best_row:
            out["placement_plan"] = best_row.get("placement_plan")

        changed += 1
        rel = None
        try:
            rel = ds_dir.relative_to(base)
        except Exception:
            rel = ds_dir
        print(f"[fix] {rel}: best.rtt {existing_rtt} -> {min_rtt} (file={best_file})")

        if args.apply:
            if args.backup and best_json.exists():
                shutil.copy2(best_json, ds_dir / "best.json.bak")
            tmp = ds_dir / "best.json.tmp"
            tmp.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n")
            tmp.replace(best_json)

    print(
        f"datasets_with_placements={scanned} "
        f"updates_needed={changed} "
        f"skipped_empty={skipped_empty} "
        f"skipped_already_ok={skipped_no_best_change} "
        f"apply={bool(args.apply)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

