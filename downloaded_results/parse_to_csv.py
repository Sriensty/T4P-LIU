#!/usr/bin/env python3
"""Parse all downloaded experiment logs into a master CSV.

Reads /tmp/exp_manifest.txt (path|regime|src|tgt|phase|lwf|pi|gamma|desc)
and for each row, extracts metrics from the log file and merges with the
manifest metadata. Also extracts pretrained_weights from .hydra/config.yaml
so the ckpt provenance is captured for every row.

Output: downloaded_results/master_table.csv
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent
MANIFEST = Path("/tmp/exp_manifest.txt")
OUT = ROOT / "master_table.csv"

METRIC_KEYS = [
    "val_MR_epoch",
    "val_minADE1_epoch",
    "val_minADE6_epoch",
    "val_minFDE1_epoch",
    "val_minFDE6_epoch",
]

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_log(log_path: Path) -> dict:
    """Extract final-epoch metrics from a lightning log."""
    metrics: dict = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(errors="ignore").splitlines():
        for key in METRIC_KEYS:
            if key in line:
                m = FLOAT_RE.findall(line.split(key, 1)[-1])
                if m:
                    metrics[key] = float(m[0])
    return metrics


def find_log(exp_dir: Path) -> Path | None:
    for cand in exp_dir.rglob("*.log"):
        if ".hydra" not in cand.parts and "checkpoints" not in cand.parts:
            return cand
    return None


def parse_config(cfg_path: Path) -> dict:
    """Extract selected fields from .hydra/config.yaml."""
    info: dict = {}
    if not cfg_path.exists():
        return info
    for line in cfg_path.read_text(errors="ignore").splitlines():
        for key in [
            "pretrained_weights",
            "lr",
            "lr2",
            "ttt_frequency",
        ]:
            if line.strip().startswith(f"{key}:"):
                info[key] = line.split(":", 1)[1].strip()
    return info


def main() -> None:
    rows = []
    manifest_lines = MANIFEST.read_text().strip().splitlines()
    for line in manifest_lines:
        parts = line.split("|")
        # Support both 9-col (legacy) and 10-col (with bug_note) manifests
        if len(parts) == 10:
            (path, regime, src, tgt, phase, lwf, pi, gamma, desc, bug_note) = parts
        else:
            (path, regime, src, tgt, phase, lwf, pi, gamma, desc) = parts
            bug_note = "-"
        exp_dir = ROOT / path
        log_path = find_log(exp_dir)
        cfg_path = exp_dir / ".hydra" / "config.yaml"

        row = {
            "path": path,
            "regime": regime,
            "src": src,
            "tgt": tgt,
            "phase": phase,
            "lwf_weight": lwf,
            "lwf_pi_weight": pi,
            "long_horizon_gamma": gamma,
            "desc": desc,
            "bug_note": bug_note,
            "log_file": str(log_path.relative_to(ROOT)) if log_path else "",
        }
        row.update(parse_log(log_path) if log_path else {})
        row.update(parse_config(cfg_path))
        rows.append(row)

    fieldnames = [
        "path",
        "regime",
        "src",
        "tgt",
        "phase",
        "lwf_weight",
        "lwf_pi_weight",
        "long_horizon_gamma",
        "val_MR_epoch",
        "val_minADE1_epoch",
        "val_minADE6_epoch",
        "val_minFDE1_epoch",
        "val_minFDE6_epoch",
        "pretrained_weights",
        "lr",
        "lr2",
        "ttt_frequency",
        "desc",
        "bug_note",
        "log_file",
    ]

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    filled = sum(1 for r in rows if r.get("val_minADE6_epoch") is not None)
    print(f"Wrote {OUT} — {len(rows)} rows, {filled} with metrics")


if __name__ == "__main__":
    main()
