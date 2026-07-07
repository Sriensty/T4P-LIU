"""Parse a sweep_lwf.sh runs.tsv index into a single results.csv.

For each row in the index, find the lightning log file inside the run's
output_dir, grep out the five final metrics, and write a tidy CSV.

Usage:
    python scripts/parse_sweep.py --index sweep_<ts>/runs.tsv --out sweep_<ts>/results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

METRICS = ["val_MR_epoch", "val_minADE1_epoch", "val_minADE6_epoch",
           "val_minFDE1_epoch", "val_minFDE6_epoch"]
METRIC_RE = re.compile(
    r"(val_(?:MR|minADE1|minADE6|minFDE1|minFDE6)_epoch):\s*([0-9.]+)"
)


def find_log(run_dir: Path) -> Path | None:
    """Locate the lightning info log under a Hydra output dir.

    Hydra writes to ``<dir>/<script>.log``. test.py uses logger
    ``lightning`` which by default goes to a file named ``test.log`` in the
    output dir (Hydra default). We fall back to any .log we can find.
    """
    if not run_dir.exists():
        return None
    candidates = sorted(run_dir.rglob("*.log"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def parse_metrics(log_path: Path) -> dict[str, float]:
    """Return the *last* occurrence of each metric (in case the log
    contains both per-step and final lines)."""
    result: dict[str, float] = {}
    text = log_path.read_text(errors="replace")
    for m in METRIC_RE.finditer(text):
        key, val = m.group(1), float(m.group(2))
        result[key] = val
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=Path,
                    help="runs.tsv produced by sweep_lwf.sh")
    ap.add_argument("--out", required=True, type=Path,
                    help="output CSV path")
    args = ap.parse_args()

    with args.index.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    header = ["run_id", "phase", "method", "lwf", "pi", "gamma", "output_dir"] + METRICS
    out_rows: list[list[str]] = []
    for row in rows:
        out_dir = Path(row["output_dir"]) if row.get("output_dir") else None
        log = find_log(out_dir) if out_dir else None
        metrics: dict[str, float] = parse_metrics(log) if log else {}
        out_rows.append([
            row.get("run_id", ""),
            row.get("phase", ""),
            row.get("method", ""),
            row.get("lwf", ""),
            row.get("pi", ""),
            row.get("gamma", ""),
            str(out_dir) if out_dir else "",
        ] + [f"{metrics.get(m, ''):.4f}" if isinstance(metrics.get(m), float) else "" for m in METRICS])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)

    # also print a tidy table to stdout
    print(f"\n[parse] wrote {len(out_rows)} rows to {args.out}\n")
    col_w = [max(len(h), max((len(str(r[i])) for r in out_rows), default=0)) for i, h in enumerate(header)]
    fmt = "  ".join("{:<" + str(w) + "}" for w in col_w)
    print(fmt.format(*header))
    print("-" * (sum(col_w) + 2 * (len(col_w) - 1)))
    for r in out_rows:
        print(fmt.format(*[str(x) for x in r]))


if __name__ == "__main__":
    main()
