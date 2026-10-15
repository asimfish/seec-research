#!/usr/bin/env python3
"""Plot all four registered source rollouts using unchanged recorded pixels."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())
    audit = json.loads(args.audit.read_text())
    assert audit["status"] == "PASS" and len(audit["cases"]) == 4
    assert [r["task"] for r in selected["cases"]] == [r["task"] for r in audit["cases"]]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}
    )
    figure = plt.figure(figsize=(13.4, 6.9))
    grid = figure.add_gridspec(
        2, 2, left=0.025, right=0.975, top=0.84, bottom=0.10, hspace=0.34, wspace=0.08
    )
    with np.load(args.input, allow_pickle=False) as archive:
        for i, (spec, row) in enumerate(zip(selected["cases"], audit["cases"])):
            images = archive[row["task"]]
            assert images.shape == (3, 256, 256, 3) and images.dtype == np.uint8
            assert (
                spec["success"] == row["success"] and spec["control_steps"] == row["control_steps"]
            )
            panel = grid[i // 2, i % 2].subgridspec(1, 3, wspace=0.025)
            for j, image in enumerate(images):
                axis = figure.add_subplot(panel[0, j])
                axis.imshow(image, interpolation="nearest")
                axis.set_axis_off()
                axis.text(
                    0.5,
                    -0.055,
                    ["Start", "Midpoint", "Final"][j],
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    color="#526374",
                    fontsize=9,
                )
                if j == 0:
                    outcome = "Success" if row["success"] else "Failure"
                    title = (
                        f"({chr(97 + i)}) {row['task']}  |  {outcome}  |  "
                        f"{row['control_steps']} control steps"
                    )
                    axis.text(
                        0,
                        1.08,
                        title,
                        transform=axis.transAxes,
                        ha="left",
                        fontsize=11,
                        fontweight="medium",
                        color="#196347" if row["success"] else "#97511f",
                    )
    figure.text(
        0.025,
        0.955,
        "Released MME-VLA in closed-loop RoboMME control",
        fontsize=17,
        fontweight="medium",
        color="#172c40",
    )
    figure.text(
        0.025,
        0.907,
        "Four preregistered train instances · frozen checkpoint · all outcomes shown",
        fontsize=11,
        color="#526374",
    )
    figure.text(
        0.025,
        0.025,
        "Raw front-camera frames at 0%, 50%, 100% of executed control. No pixel edits. "
        "This qualifies execution; it is not a method result or full benchmark score.",
        fontsize=9,
        color="#526374",
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png", "pdf"):
        figure.savefig(str(args.output_prefix) + "." + extension, dpi=180, facecolor="white")
    svg = args.output_prefix.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_npz_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "selection": selected,
                "plan_id": audit["plan_id"],
                "pixel_editing": "NONE",
                "scope": "Four source cases; no method efficacy or full benchmark score",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
