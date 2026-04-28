"""
Rebuild orchestrator.

Runs the full downstream pipeline after the master vault has been
refreshed by data_ingestion.py. Executes each step in the correct
order and stops on first failure.

Steps:
    1. matrix_builder.py        → pitch_matrix.parquet + contact_matrix.parquet
    2. enviroment_ingestion.py  → game_weather_log.parquet
    3. enviroment_merger.py     → contact_matrix_env.parquet
    4. recent_form.py           → recent_form_hitters + recent_form_pitchers

The early_inning_tracker.py step is NOT auto-run because it fetches
per-game day/night metadata from the MLB API which takes ~15-20
minutes the first time. Run it separately when convenient.

Usage:
    python rebuild.py               # run everything
    python rebuild.py --skip weather # run everything except weather refetch
"""
import argparse
import os
import subprocess
import sys


STEPS = [
    ('matrices', 'matrix_builder.py',
     'Build PQM and CQM from the master vault'),

    ('weather', 'enviroment_ingestion.py',
     'Fetch historical weather (slow first time; instant on resume)'),

    ('environment', 'enviroment_merger.py',
     'Merge weather into CQM to produce contact_matrix_env'),

    ('recent_form', 'recent_form.py',
     'Build exponentially-weighted recent-form snapshots'),
]


def run_step(script: str, label: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  RUNNING: {script}")
    print(f"  {label}")
    print('='*70)

    if not os.path.exists(script):
        print(f"[SKIP] {script} not found in current directory.")
        return False

    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}")
        return False

    print(f"\n[OK] {script} completed")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', nargs='+', default=[],
                        help='Step names to skip (e.g. --skip weather)')
    args = parser.parse_args()

    print("=" * 70)
    print("  DOWNSTREAM REBUILD — runs after data_ingestion.py updates the vault")
    print("=" * 70)
    print(f"\nSteps to run:")
    for name, script, desc in STEPS:
        marker = '[SKIP]' if name in args.skip else '[RUN ]'
        print(f"  {marker} {name:<12} → {script}")
    print()

    failures = []
    for name, script, desc in STEPS:
        if name in args.skip:
            continue
        ok = run_step(script, desc)
        if not ok:
            failures.append(script)
            print(f"\n[STOP] Halting because {script} failed. Fix and re-run.")
            break

    print("\n" + "=" * 70)
    if failures:
        print(f"  REBUILD INCOMPLETE — {len(failures)} step(s) failed")
    else:
        print(f"  REBUILD COMPLETE — app.py should work now")
        print(f"\n  Remember to also run (when convenient):")
        print(f"    python early_inning_tracker.py   (15-20 min on first run)")
    print("=" * 70)


if __name__ == "__main__":
    main()