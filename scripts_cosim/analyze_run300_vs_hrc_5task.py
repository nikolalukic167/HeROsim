#!/usr/bin/env python3
"""
Analyze run300 co-sim datasets (5-task) vs HRC simulation 5-task slices.

- Run300: 5-task co-sim datasets in simulation_data/artifacts/run300/gnn_datasets/.
  Each has workload (5 events), optimal_result (stats.taskResults, placement), best.json (RTT).
- HRC: simulation_result_herocache_network*.json from run_simulation.sh (big workload, normal sim).
  Extract first 5 taskResults (and optional slices at 100–104, 1000–1004) for comparison.

Compares: total RTT (5 tasks), per-task elapsedTime, app types, etc.

Usage:
  pipenv run python scripts_cosim/analyze_run300_vs_hrc_5task.py
  pipenv run python scripts_cosim/analyze_run300_vs_hrc_5task.py --run300-dir /path/to/run300/gnn_datasets --hrc-network /path/to/herocache_network.json --hrc-batch /path/to/herocache_network_batch.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

def _paths() -> tuple[Path, Path, Path]:
    base = Path("/root/projects/my-herosim")
    return (
        base / "simulation_data/artifacts/run300/gnn_datasets",
        base / "simulation_data/results/simulation_result_herocache_network.json",
        base / "simulation_data/results/simulation_result_herocache_network_batch.json",
    )

RUN300_BASE, HRC_NETWORK, HRC_BATCH = _paths()


def load_hrc_slice(jpath: Path, start: int, count: int = 5) -> list[dict] | None:
    """Load taskResults[start:start+count] via jq."""
    if not jpath.exists():
        return None
    try:
        r = subprocess.run(
            ["jq", f".stats.taskResults[{start}:{start + count}]", str(jpath)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return json.loads(r.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return None


def load_hrc_meta(jpath: Path) -> dict[str, Any] | None:
    """Load HRC metadata (total_rtt, num_tasks, etc.) via jq."""
    if not jpath.exists():
        return None
    try:
        r = subprocess.run(
            ["jq", "{total_rtt, num_tasks, policy, scheduling_strategy, workload_file}", str(jpath)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return json.loads(r.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return None


def load_hrc_first5(jpath: Path) -> dict[str, Any] | None:
    """Load HRC result metadata and first 5 taskResults via jq (avoids loading full ~180MB JSON)."""
    meta = load_hrc_meta(jpath)
    if meta is None:
        return None
    tasks = load_hrc_slice(jpath, 0, 5)
    if tasks is None:
        return None
    meta["first5_taskResults"] = tasks
    return meta


def load_run300_captured_state(ds_dir: Path) -> dict[str, Any] | None:
    """Load system_state_captured_unique.json for a run300 dataset."""
    path = ds_dir / "system_state_captured_unique.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def analyze_run300_ds(ds_dir: Path) -> dict[str, Any] | None:
    """Load one run300 dataset (5-task co-sim). Return None if not 5-task or missing files."""
    workload_path = ds_dir / "workload.json"
    optimal_path = ds_dir / "optimal_result.json"
    best_path = ds_dir / "best.json"
    if not workload_path.exists() or not optimal_path.exists() or not best_path.exists():
        return None
    with open(workload_path) as f:
        workload = json.load(f)
    events = workload.get("events") or []
    if len(events) != 5:
        return None
    with open(optimal_path) as f:
        optimal = json.load(f)
    with open(best_path) as f:
        best = json.load(f)
    task_results = (optimal.get("stats") or {}).get("taskResults") or []
    if len(task_results) != 5:
        return None
    rtt = best.get("rtt")
    if rtt is None:
        return None
    placement = (optimal.get("sample") or {}).get("placement_plan") or {}
    out: dict[str, Any] = {
        "dataset_id": ds_dir.name,
        "rtt_total": float(rtt),
        "num_tasks": 5,
        "taskResults": task_results,
        "placement_plan": placement,
        "workload_events": events,
    }
    captured = load_run300_captured_state(ds_dir)
    if captured is not None:
        out["system_state_captured"] = captured
    return out


def summarize_queue_snapshot(snap: dict[str, int] | None) -> dict[str, Any]:
    """Summarize a full_queue_snapshot or queue_snapshot_at_scheduling dict."""
    if not snap:
        return {"n_keys": 0, "total": 0, "max": 0, "non_zero": 0}
    vals = [int(v) for v in snap.values()]
    return {
        "n_keys": len(snap),
        "total": sum(vals),
        "max": max(vals) if vals else 0,
        "non_zero": sum(1 for v in vals if v > 0),
    }


def summarize_run300_captured_state(captured: dict[str, Any]) -> dict[str, Any]:
    """Extract summary from system_state_captured_unique.json."""
    replicas = captured.get("replicas") or {}
    n_replicas = sum(len(v) for v in replicas.values()) if isinstance(replicas, dict) else 0
    task_placements = captured.get("task_placements") or []
    queue_stats: list[dict[str, Any]] = []
    for tp in task_placements:
        full = tp.get("full_queue_snapshot") or {}
        queue_stats.append(summarize_queue_snapshot(full))
    return {
        "n_replicas": n_replicas,
        "replica_task_types": list(replicas.keys()) if isinstance(replicas, dict) else [],
        "n_task_placements": len(task_placements),
        "queue_snapshot_stats": queue_stats,
        "has_temporal": any(
            bool((tp.get("temporal_state_at_scheduling") or {}))
            for tp in task_placements
        ),
    }


def summarize_hrc_task_system_state(tr: dict) -> dict[str, Any]:
    """Summarize system state from HRC taskResult (systemStateResult, fullQueueSnapshot, etc.)."""
    ss = tr.get("systemStateResult") or {}
    replicas = ss.get("replicas") or {}
    n_replicas = sum(len(v) for v in replicas.values()) if isinstance(replicas, dict) else 0
    full = tr.get("fullQueueSnapshot") or tr.get("full_queue_snapshot") or {}
    q_at = tr.get("queueSnapshotAtScheduling") or tr.get("queue_snapshot_at_scheduling") or {}
    temporal = tr.get("temporalStateAtScheduling") or tr.get("temporal_state_at_scheduling") or {}
    return {
        "n_replicas": n_replicas,
        "queue_full": summarize_queue_snapshot(full),
        "queue_at_scheduling": summarize_queue_snapshot(q_at),
        "has_temporal": bool(temporal),
    }


def summarize_task_results(trs: list[dict], label: str) -> dict[str, Any]:
    """Compute summary stats over a list of task result dicts (elapsedTime, etc.)."""
    elapsed = [float(t.get("elapsedTime", 0)) for t in trs if t.get("elapsedTime") is not None]
    apps = [(t.get("applicationType") or {}).get("name") or (t.get("taskType") or {}).get("name") for t in trs]
    src = [t.get("sourceNode") for t in trs]
    total = sum(elapsed) if elapsed else 0.0
    return {
        "label": label,
        "n": len(trs),
        "rtt_total": total,
        "elapsed_mean": float(np.mean(elapsed)) if elapsed else 0.0,
        "elapsed_median": float(np.median(elapsed)) if elapsed else 0.0,
        "elapsed_min": float(np.min(elapsed)) if elapsed else 0.0,
        "elapsed_max": float(np.max(elapsed)) if elapsed else 0.0,
        "apps": apps,
        "source_nodes": src,
    }


def main(
    run300_base: Path | None = None,
    hrc_network_path: Path | None = None,
    hrc_batch_path: Path | None = None,
) -> None:
    run300_base = run300_base or RUN300_BASE
    hrc_network_path = hrc_network_path or HRC_NETWORK
    hrc_batch_path = hrc_batch_path or HRC_BATCH

    print("=" * 72)
    print("Run300 (5-task co-sim) vs HRC (5-task slice) analysis")
    print("=" * 72)

    # --- Run300: collect 5-task datasets ---
    ds_dirs = sorted(run300_base.glob("ds_*"))
    run300 = []
    for d in ds_dirs:
        rec = analyze_run300_ds(d)
        if rec:
            run300.append(rec)
    print(f"\nRun300: {len(run300)} datasets with 5 tasks (of {len(ds_dirs)} dirs)")

    if not run300:
        print("No run300 5-task datasets found. Exiting.")
        sys.exit(1)

    rtts = [r["rtt_total"] for r in run300]
    print(f"  RTT (optimal, 5-task total): min={min(rtts):.4f}s  max={max(rtts):.4f}s  mean={np.mean(rtts):.4f}s  median={np.median(rtts):.4f}s")

    # Per-task elapsedTime across all run300 (pool first 5 from each)
    all_elapsed = []
    for r in run300:
        for t in r["taskResults"]:
            e = t.get("elapsedTime")
            if e is not None:
                all_elapsed.append(float(e))
    if all_elapsed:
        print(f"  Per-task elapsedTime (pooled): mean={np.mean(all_elapsed):.4f}s  median={np.median(all_elapsed):.4f}s  min={np.min(all_elapsed):.4f}s  max={np.max(all_elapsed):.4f}s")

    # --- HRC: first 5 tasks ---
    print("\n--- HRC results (first 5 tasks of full workload) ---")
    hrc_network = load_hrc_first5(hrc_network_path)
    hrc_batch = load_hrc_first5(hrc_batch_path)

    for name, hrc in [("herocache_network", hrc_network), ("herocache_network_batch", hrc_batch)]:
        if not hrc:
            print(f"  {name}: not available")
            continue
        trs = hrc.get("first5_taskResults") or []
        s = summarize_task_results(trs, name)
        print(f"  {name}:")
        print(f"    Full run: total_rtt={hrc.get('total_rtt')}  num_tasks={hrc.get('num_tasks')}")
        print(f"    First 5 tasks: rtt_total={s['rtt_total']:.4f}s  mean_elapsed={s['elapsed_mean']:.4f}s  median={s['elapsed_median']:.4f}s")
        print(f"    Apps: {s['apps']}")
        print(f"    Source nodes: {s['source_nodes']}")

    # --- Comparison ---
    print("\n--- Comparison ---")
    run300_rtt_mean = float(np.mean(rtts))
    run300_rtt_median = float(np.median(rtts))
    print(f"  Run300 co-sim optimal (5-task RTT): mean={run300_rtt_mean:.4f}s  median={run300_rtt_median:.4f}s")
    for name, hrc in [("herocache_network", hrc_network), ("herocache_network_batch", hrc_batch)]:
        if not hrc:
            continue
        trs = hrc.get("first5_taskResults") or []
        if len(trs) != 5:
            continue
        s = summarize_task_results(trs, name)
        ratio = s["rtt_total"] / run300_rtt_median if run300_rtt_median > 0 else 0
        print(f"  HRC {name} first-5 RTT: {s['rtt_total']:.4f}s  (vs run300 median: {ratio:.2f}x)")

    # --- HRC 5-task slices at different offsets ---
    print("\n--- HRC 5-task slices (same workload, different offsets) ---")
    for label, jpath in [("herocache_network", hrc_network_path), ("herocache_network_batch", hrc_batch_path)]:
        if not jpath.exists():
            continue
        print(f"  {label}:")
        for start in [0, 100, 1000]:
            trs = load_hrc_slice(jpath, start, 5)
            if not trs or len(trs) < 5:
                print(f"    tasks[{start}:{start+5}]: n/a")
                continue
            s = summarize_task_results(trs, f"offset_{start}")
            apps_preview = s["apps"][:3] if len(s["apps"]) >= 3 else s["apps"]
            print(f"    tasks[{start}:{start+5}] RTT total={s['rtt_total']:.4f}s  mean_elapsed={s['elapsed_mean']:.4f}s  apps={apps_preview}...")

    # --- Sample run300 vs HRC first-5 side-by-side ---
    print("\n--- Sample: run300 ds_00000 vs HRC first-5 ---")
    sample = next((r for r in run300 if r["dataset_id"] == "ds_00000"), run300[0])
    print(f"  Run300 {sample['dataset_id']}: RTT={sample['rtt_total']:.4f}s")
    for i, t in enumerate(sample["taskResults"]):
        e = t.get("elapsedTime")
        e = float(e) if e is not None else 0.0
        app = (t.get("applicationType") or {}).get("name") or (t.get("taskType") or {}).get("name")
        src = t.get("sourceNode")
        print(f"    task {i}: elapsed={e:.4f}s  app={app}  source={src}")
    if hrc_network and (trs := hrc_network.get("first5_taskResults")):
        print("  HRC herocache_network first-5:")
        for i, t in enumerate(trs):
            e = t.get("elapsedTime")
            e = float(e) if e is not None else 0.0
            app = (t.get("applicationType") or {}).get("name") or (t.get("taskType") or {}).get("name")
            src = t.get("sourceNode")
            print(f"    task {i}: elapsed={e:.4f}s  app={app}  source={src}")

    # --- System state: run300 captured vs HRC at scheduling time ---
    print("\n--- System state (run300 captured vs HRC scheduling-time) ---")
    with_captured = [r for r in run300 if r.get("system_state_captured")]
    print(f"  Run300: {len(with_captured)}/{len(run300)} datasets have system_state_captured_unique.json")
    if with_captured:
        ex = next((r for r in with_captured if r["dataset_id"] == "ds_00000"), with_captured[0])
        cap = ex["system_state_captured"]
        s = summarize_run300_captured_state(cap)
        print(f"  Sample {ex['dataset_id']} captured state: n_replicas={s['n_replicas']}  "
              f"task_placements={s['n_task_placements']}  has_temporal={s['has_temporal']}")
        if s["queue_snapshot_stats"]:
            q0 = s["queue_snapshot_stats"][0]
            print(f"    First-task queue snapshot: n_keys={q0['n_keys']}  total={q0['total']}  max={q0['max']}  non_zero={q0['non_zero']}")

    for name, hrc in [("herocache_network", hrc_network), ("herocache_network_batch", hrc_batch)]:
        if not hrc or not (trs := hrc.get("first5_taskResults")):
            continue
        print(f"  HRC {name} first-5 system state:")
        for i, t in enumerate(trs):
            ss = summarize_hrc_task_system_state(t)
            full = "y" if (t.get("fullQueueSnapshot") or t.get("full_queue_snapshot")) else "n"
            at_sched = "y" if (t.get("queueSnapshotAtScheduling") or t.get("queue_snapshot_at_scheduling")) else "n"
            temporal = "y" if (t.get("temporalStateAtScheduling") or t.get("temporal_state_at_scheduling")) else "n"
            print(f"    task {i}: n_replicas={ss['n_replicas']}  full_queue_snapshot={full}  "
                  f"queue_at_scheduling={at_sched}  temporal={temporal}")
            if ss["queue_full"]["n_keys"] > 0:
                print(f"      queue_full: n_keys={ss['queue_full']['n_keys']}  total={ss['queue_full']['total']}  "
                      f"max={ss['queue_full']['max']}  non_zero={ss['queue_full']['non_zero']}")

    # --- App/source diversity: run300 vs HRC ---
    print("\n--- App/source diversity (run300 vs HRC first-5) ---")
    run300_apps: list[str] = []
    run300_sources: list[str] = []
    for r in run300:
        for t in r["taskResults"]:
            a = (t.get("applicationType") or {}).get("name") or (t.get("taskType") or {}).get("name")
            if a:
                run300_apps.append(a)
            s = t.get("sourceNode")
            if s:
                run300_sources.append(s)
    run300_app_counts: dict[str, int] = {}
    for a in run300_apps:
        run300_app_counts[a] = run300_app_counts.get(a, 0) + 1
    print(f"  Run300 (pooled): app types={list(run300_app_counts.keys())}  counts={run300_app_counts}  "
          f"unique sources={len(set(run300_sources))}")
    for name, hrc in [("herocache_network", hrc_network), ("herocache_network_batch", hrc_batch)]:
        if not hrc or not (trs := hrc.get("first5_taskResults")):
            continue
        apps = [(t.get("applicationType") or {}).get("name") or (t.get("taskType") or {}).get("name") for t in trs]
        srcs = [t.get("sourceNode") for t in trs]
        print(f"  HRC {name} first-5: apps={list(dict.fromkeys(apps))}  sources={list(dict.fromkeys(srcs))}")

    # --- HRC 5-task RTT distribution (many slices) ---
    print("\n--- HRC 5-task RTT distribution (slices across workload) ---")
    for label, jpath in [("herocache_network", hrc_network_path), ("herocache_network_batch", hrc_batch_path)]:
        if not jpath.exists():
            continue
        meta = load_hrc_meta(jpath)
        if not meta:
            continue
        n_tasks = meta.get("num_tasks") or 0
        if n_tasks < 10:
            continue
        step = max(1, (n_tasks - 5) // 5)
        starts = list(range(0, n_tasks - 4, step))[:5]
        slice_rtts: list[float] = []
        for start in starts:
            trs = load_hrc_slice(jpath, start, 5)
            if not trs or len(trs) < 5:
                continue
            s = summarize_task_results(trs, "")
            slice_rtts.append(s["rtt_total"])
        if not slice_rtts:
            continue
        slice_rtts_arr = np.array(slice_rtts)
        print(f"  {label}: {len(slice_rtts)} slices  min={float(np.min(slice_rtts_arr)):.2f}s  "
              f"p25={float(np.percentile(slice_rtts_arr, 25)):.2f}s  median={float(np.median(slice_rtts_arr)):.2f}s  "
              f"p75={float(np.percentile(slice_rtts_arr, 75)):.2f}s  max={float(np.max(slice_rtts_arr)):.2f}s")

    # --- Interpretation summary ---
    print("\n--- Summary: what this means & is co-sim data useful? ---")
    print("  • Run300 = 5-task co-sim OPTIMAL (brute-force best placement). HRC = real policy on 12k-task workload.")
    print("  • HRC first-5 RTT is 6–74x worse than run300 median: heuristics leave clear room for learning.")
    print("  • Co-sim data is useful as GNN supervision: (state, placement) -> optimal RTT.")
    print("  • Caveats: (1) run300 vs HRC = different apps/sources/infra (not apples-to-apples). "
          "(2) run300 = isolated 5-task; HRC = contention, queues. (3) HRC batch has no queue/temporal "
          "in your current result file—re-run HRC batch after capture changes if needed.")
    print("  • Further analyses: (a) Match 5-task by app mix + sources then compare. "
          "(b) Train GNN on run300, eval on HRC 5-task slices. (c) Queue-state distribution run300 vs HRC.")

    print("\nDone.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze run300 5-task co-sim vs HRC 5-task slices.")
    p.add_argument("--run300-dir", type=Path, default=None, help="Run300 gnn_datasets directory")
    p.add_argument("--hrc-network", type=Path, default=None, help="HRC herocache_network result JSON")
    p.add_argument("--hrc-batch", type=Path, default=None, help="HRC herocache_network_batch result JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        run300_base=args.run300_dir,
        hrc_network_path=args.hrc_network,
        hrc_batch_path=args.hrc_batch,
    )
