import json
import os
import time
from typing import Any, Dict, List, Tuple

MANIFEST_VERSION = 1
JOB_PENDING = "pending"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"
JOB_MERGED = "merged"


def safe_job_id(product_id: str, timeframe: str) -> str:
    return f"{str(product_id).replace('-', '_')}__{str(timeframe)}"


def default_worker_output_name(product_id: str, timeframe: str) -> str:
    return f"historical_shadow_replay.{safe_job_id(product_id, timeframe)}.csv"


def _default_job(product_id: str, timeframe: str, output_dir: str, now: float) -> Dict[str, Any]:
    job_id = safe_job_id(product_id, timeframe)
    return {
        "job_id": job_id,
        "product_id": product_id,
        "timeframe": timeframe,
        "status": JOB_PENDING,
        "attempts": 0,
        "created_ts": now,
        "updated_ts": now,
        "started_ts": 0.0,
        "finished_ts": 0.0,
        "merged_ts": 0.0,
        "output_path": os.path.join(output_dir, default_worker_output_name(product_id, timeframe)),
        "rows_written": 0,
        "evaluated": 0,
        "candle_rows": 0,
        "qualified_rows": 0,
        "accepted_rows": 0,
        "avg_net_pnl_bps": 0.0,
        "median_net_pnl_bps": 0.0,
        "error": "",
        "source": "",
    }


def build_default_manifest(*, products: List[str], timeframes: List[str], output_dir: str) -> Dict[str, Any]:
    now = time.time()
    jobs = {}
    for product_id in products:
        for timeframe in timeframes:
            job = _default_job(product_id, timeframe, output_dir, now)
            jobs[job["job_id"]] = job
    return {"version": MANIFEST_VERSION, "created_ts": now, "updated_ts": now, "jobs": jobs}


def load_manifest(path: str) -> Dict[str, Any]:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    manifest["updated_ts"] = time.time()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def ensure_manifest(*, path: str, products: List[str], timeframes: List[str], output_dir: str) -> Dict[str, Any]:
    existing = load_manifest(path)
    if not existing or "jobs" not in existing:
        manifest = build_default_manifest(products=products, timeframes=timeframes, output_dir=output_dir)
        save_manifest(path, manifest)
        return manifest
    jobs = existing.get("jobs", {})
    changed = False
    now = time.time()
    for product_id in products:
        for timeframe in timeframes:
            job_id = safe_job_id(product_id, timeframe)
            if job_id not in jobs:
                jobs[job_id] = _default_job(product_id, timeframe, output_dir, now)
                changed = True
    existing["jobs"] = jobs
    if changed:
        save_manifest(path, existing)
    return existing


def update_job(*, path: str, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    manifest = load_manifest(path)
    if not manifest or "jobs" not in manifest:
        raise RuntimeError(f"Manifest missing or invalid: {path}")
    jobs = manifest["jobs"]
    if job_id not in jobs:
        raise KeyError(f"Unknown historical replay job: {job_id}")
    jobs[job_id].update(updates)
    jobs[job_id]["updated_ts"] = time.time()
    manifest["jobs"] = jobs
    save_manifest(path, manifest)
    return jobs[job_id]


def jobs_by_status(manifest: Dict[str, Any], statuses: Tuple[str, ...]) -> List[Dict[str, Any]]:
    return [job for job in (manifest.get("jobs", {}) or {}).values() if str(job.get("status")) in statuses]


def manifest_progress(manifest: Dict[str, Any]) -> Dict[str, Any]:
    jobs = list((manifest.get("jobs", {}) or {}).values())
    total = len(jobs)
    if total <= 0:
        return {"total_jobs": 0, "done_jobs": 0, "merged_jobs": 0, "failed_jobs": 0, "running_jobs": 0, "pending_jobs": 0, "progress": 0.0, "progress_pct": 0.0}
    merged = sum(1 for j in jobs if j.get("status") == JOB_MERGED)
    done = sum(1 for j in jobs if j.get("status") in {JOB_DONE, JOB_MERGED})
    failed = sum(1 for j in jobs if j.get("status") == JOB_FAILED)
    running = sum(1 for j in jobs if j.get("status") == JOB_RUNNING)
    pending = sum(1 for j in jobs if j.get("status") == JOB_PENDING)
    weighted = merged + 0.5 * (done - merged)
    progress = weighted / max(1, total)
    return {"total_jobs": total, "done_jobs": done, "merged_jobs": merged, "failed_jobs": failed, "running_jobs": running, "pending_jobs": pending, "progress": max(0.0, min(1.0, progress)), "progress_pct": max(0.0, min(100.0, progress * 100.0))}
