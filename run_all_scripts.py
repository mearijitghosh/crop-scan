"""Run all Python scripts in the crop_disease_classification directory.

This script executes all .py files in the specified directory.

Usage:
    python run_all_scripts.py [--output-dir OUTPUT_DIR]
"""

import re
import time
import threading
import argparse
import subprocess
import sys
from pathlib import Path

# ── Watchdog config ───────────────────────────────────────────────────────────
# Kill the script if it/s drops below this threshold after the warmup period.
# 0.01 it/s = ~100 seconds per step = broken (lazy loading bottleneck).
# 0.05 it/s = ~20 seconds per step  = acceptable minimum for cached loading.
MIN_ITS_THRESHOLD  = 0.05   # it/s — kill if slower than this
WARMUP_MINUTES     = 15     # don't check speed until this many minutes in
                            # (model load + cache build can take 10-15 min)
MAX_RUNTIME_HOURS  = 12     # hard kill after this many hours regardless


def parse_its(line: str):
    """Extract it/s from a tqdm progress line like '[ 5/30 00:10 < 03:20, 0.02 it/s]'"""
    match = re.search(r'([\d.]+)\s*it/s', line)
    if match:
        return float(match.group(1))
    return None


def run_script(script_path: str, output_dir: str = None) -> bool:
    """Run a Python script with real-time output streaming and speed watchdog."""
    script_path = Path(script_path)

    if output_dir is None:
        output_dir = script_path.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"{script_path.stem}.log"

    print(f"[RUN] Running: {script_path.name}")
    print(f"[LOG] Output: {log_path}")
    print(f"[WATCH] Speed threshold : {MIN_ITS_THRESHOLD} it/s  (after {WARMUP_MINUTES} min warmup)")
    print(f"[WATCH] Max runtime     : {MAX_RUNTIME_HOURS} hours")
    print("-" * 50)

    start_time = time.time()
    kill_reason = [None]   # mutable so the watchdog thread can set it

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            # ── Watchdog thread ───────────────────────────────────────
            # Runs in background, checks elapsed time and last known it/s.
            # Kills the process if either limit is exceeded.
            last_its = [None]

            def watchdog():
                while process.poll() is None:
                    elapsed_min = (time.time() - start_time) / 60
                    elapsed_hr  = elapsed_min / 60
                    h, rem = divmod(int(elapsed_min * 60), 3600)
                    m, s   = divmod(rem, 60)

                    # Print live elapsed every 30 seconds
                    its_str = f"  |  {last_its[0]:.3f} it/s" if last_its[0] is not None else ""
                    print(f"[ELAPSED] {h:02d}:{m:02d}:{s:02d}{its_str}", flush=True)

                    # Hard time limit
                    if elapsed_hr >= MAX_RUNTIME_HOURS:
                        kill_reason[0] = f"exceeded max runtime of {MAX_RUNTIME_HOURS}h"
                        print(f"\n[KILL] {kill_reason[0]} — terminating.")
                        process.kill()
                        return

                    # Speed check (only after warmup)
                    if elapsed_min >= WARMUP_MINUTES and last_its[0] is not None:
                        if last_its[0] < MIN_ITS_THRESHOLD:
                            kill_reason[0] = (
                                f"speed {last_its[0]:.3f} it/s is below "
                                f"threshold {MIN_ITS_THRESHOLD} it/s after "
                                f"{elapsed_min:.1f} min"
                            )
                            print(f"\n[KILL] {kill_reason[0]} — terminating.")
                            process.kill()
                            return

                    time.sleep(300)  # check every 5 minutes

            t = threading.Thread(target=watchdog, daemon=True)
            t.start()

            # ── Stream output ─────────────────────────────────────────
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

                # Parse it/s from progress lines
                its = parse_its(line)
                if its is not None:
                    last_its[0] = its

            process.wait()

        elapsed = time.time() - start_time
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print("-" * 50)
        print(f"[TIME] Duration: {duration_str}  ({elapsed/60:.1f} mins)")

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- TIMING ---\n")
            lf.write(f"Duration : {duration_str} ({elapsed/60:.1f} mins)\n")

        if kill_reason[0]:
            print(f"[KILL] Killed: {script_path.name} — {kill_reason[0]}")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"Killed   : {kill_reason[0]}\n")
            return False
        elif process.returncode == 0:
            print(f"[OK] Completed: {script_path.name}")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"Exit code: 0\n")
            return True
        else:
            print(f"[FAIL] Failed: {script_path.name} — exit code {process.returncode}")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"Exit code: {process.returncode}\n")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[FAIL] Error after {elapsed/60:.1f} mins: {script_path.name} — {e}")
        return False



def main():
    parser = argparse.ArgumentParser(description="Run all Python scripts")
    parser.add_argument("--scripts-dir", type=str, 
                        default="scripts/crop_disease_classification",
                        help="Directory containing scripts")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for logs")
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    scripts_dir = script_dir / args.scripts_dir
    
    if not scripts_dir.exists():
        print(f"[FAIL] Scripts directory not found: {scripts_dir}")
        sys.exit(1)
    
    # Find all Python scripts if not specified
    scripts = list(scripts_dir.glob("*.py"))
    
    if not scripts:
        print(f"[FAIL] No scripts found in {scripts_dir}")
        sys.exit(1)
    
    print(f"[DIR] Found {len(scripts)} scripts")
    print("=" * 50)
    
    # Run each script
    import time
    total_start = time.time()
    results = []
    for script in sorted(scripts):
        success = run_script(str(script), args.output_dir)
        results.append((script.name, success))

    total_elapsed = time.time() - total_start
    total_hours, rem = divmod(int(total_elapsed), 3600)
    total_mins, total_secs = divmod(rem, 60)

    # Summary
    print("=" * 50)
    print("[SUMMARY] Summary:")
    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)
    print(f"   [OK] Passed : {passed}")
    print(f"   [FAIL] Failed : {failed}")
    print(f"   [TIME] Total  : {total_hours:02d}:{total_mins:02d}:{total_secs:02d}  ({total_elapsed/60:.1f} mins)")
    
    if failed > 0:
        print("\n[FAIL] Scripts that failed:")
        for name, success in results:
            if not success:
                print(f"   - {name}")
        sys.exit(1)
    else:
        print("\n[OK] All scripts completed successfully!")


if __name__ == "__main__":
    main()