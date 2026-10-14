#!/usr/bin/env python3
"""Plot measured reference coverage and the registered control qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.evidence / "final_audit.json").read_text())
    original = json.loads((args.evidence / "program_check_run1.json").read_text())
    projected = json.loads((args.evidence / "teacher_control_result.json").read_text())
    assert audit["status"] == "PASS" and audit["summary"]["completed"] == 500
    assert original["status"] == "PASS" and original["compared"] == original["expected"] == 3
    assert projected["compared"] == projected["expected"] == 3
    tasks = sorted(audit["summary"]["per_task"], key=int)
    rates = [100 * audit["summary"]["per_task"][t]["successes"] / 50 for t in tasks]
    baseline = {c["task"]: c["cost"]["policy_control_steps"] for c in original["comparisons"]}
    controls = {c["task"]: c for c in projected["cases"]}
    ids = sorted(baseline)
    assert ids == sorted(controls)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "libero-reference-and-control-v1",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    bars = ax.barh(range(10), rates, height=0.64, color="#237c78")
    ax.set_yticks(range(10), [f"Task {t}" for t in tasks])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Terminal success (%) · 50 official states per task")
    ax.set_title("a   Released-checkpoint reference", loc="left", fontweight="bold", pad=13)
    ax.grid(axis="x", alpha=0.13)
    ax.set_axisbelow(True)
    for bar, rate in zip(bars, rates):
        ax.text(
            rate + 1.2, bar.get_y() + bar.get_height() / 2, f"{rate:.0f}%", va="center", fontsize=9
        )
    ax.text(
        0.02,
        1.0,
        "476 / 500 completed successfully · 95.2%",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        color="#15504e",
    )
    ax = axes[1]
    x = np.arange(3)
    width = 0.33
    ax.bar(
        x - width / 2,
        [baseline[t] for t in ids],
        width,
        color="#237c78",
        label="Original 8-action program",
    )
    projected_bars = ax.bar(
        x + width / 2,
        [controls[t]["cost"]["policy_control_steps"] for t in ids],
        width,
        color="#d99242",
        label="Fresh first-action replanning",
    )
    for i, task in enumerate(ids):
        n = controls[task]["cost"]["policy_control_steps"]
        if not controls[task]["success"]:
            projected_bars[i].set_hatch("///")
            ax.text(i + width / 2, n + 10, "timeout", ha="center", fontsize=9, color="#9b4b0d")
        else:
            ax.text(i + width / 2, n + 10, str(n), ha="center", fontsize=9)
        ax.text(i - width / 2, baseline[task] + 10, str(baseline[task]), ha="center", fontsize=9)
    ax.set_xticks(x, [f"Task {t}\ninitial state 0" for t in ids])
    ax.set_ylim(0, 590)
    ax.set_ylabel("Policy control steps (timeout censored)")
    ax.set_title(
        "b   Executor qualification · 3 fixed cases", loc="left", fontweight="bold", pad=13
    )
    ax.grid(axis="y", alpha=0.13)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.text(
        0.06,
        0.025,
        "Fixed checkpoint; zero updates. "
        "Full-program adapter reproduces all 3 original traces exactly. "
        "No method efficacy or adaptation claim.",
        fontsize=8.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.19, top=0.84, wspace=0.29)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png", "pdf"):
        metadata = (
            {"Date": None}
            if suffix == "svg"
            else ({"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
        )
        fig.savefig(
            args.output.with_suffix("." + suffix), dpi=200, facecolor="white", metadata=metadata
        )
    svg = args.output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
