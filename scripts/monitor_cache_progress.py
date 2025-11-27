#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from datetime import datetime

def monitor_progress(cache_dir: str, expected_files: int, update_interval: int, shards: int) -> None:
    cache_path = Path(cache_dir)
    log_dir = Path("logs")
    start_time = time.time()

    print("=== ESM-2 Cache Population Monitor ===")
    print(f"Cache directory: {cache_path}")
    print(f"Expected files: {expected_files}")
    print(f"Update interval: {update_interval}s")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        if cache_path.exists():
            completed = len(list(cache_path.glob("prot_emb_*.pt")))
        else:
            completed = 0
        progress_pct = (completed / expected_files) * 100 if expected_files else 0.0

        job_status = {}
        for i in range(shards):
            log_file = log_dir / f"esm2_cache_{i}.out"
            err_file = log_dir / f"esm2_cache_{i}.err"
            status = "PENDING"
            if log_file.exists():
                try:
                    content = log_file.read_text()
                    if "Summary:" in content:
                        status = "COMPLETED"
                    elif "ERROR" in content or "Traceback" in content:
                        status = "FAILED"
                    elif "Processed" in content:
                        status = "RUNNING"
                    else:
                        status = "STARTED"
                except Exception:
                    status = "UNKNOWN"
            if err_file.exists() and err_file.stat().st_size > 0 and status not in ("FAILED", "COMPLETED"):
                status = "ERROR"
            job_status[i] = status

        elapsed_hours = (time.time() - start_time) / 3600.0
        rate_per_hour = completed / elapsed_hours if elapsed_hours > 0 else 0.0
        remaining = max(0, expected_files - completed)
        eta_hours = remaining / rate_per_hour if rate_per_hour > 0 else 0.0
        eta_str = f"{eta_hours:.1f}h" if eta_hours < 24 else f"{eta_hours/24:.1f}d"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {completed}/{expected_files} ({progress_pct:.1f}%)")
        print(f"Rate: {rate_per_hour:.0f} files/hour, ETA: {eta_str}")
        print("Shard status:", ", ".join(f"{i}:{s}" for i, s in job_status.items()))

        if all(s == "COMPLETED" for s in job_status.values()) and completed >= expected_files:
            print("\nCache population completed")
            break

        time.sleep(update_interval)

def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor ESM-2 cache population progress")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--expected", type=int, default=21657)
    parser.add_argument("--update-interval", type=int, default=300)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    monitor_progress(args.cache_dir, args.expected, args.update_interval, args.shards)

if __name__ == "__main__":
    main()
