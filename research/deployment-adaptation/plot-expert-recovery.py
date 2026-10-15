#!/usr/bin/env python3
"""Show the exact shared learner prefix and both recorded continuations."""

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
    for name in ("input", "result", "output-prefix"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    assert result["success"] is True and result["prefix_frames_checked"] == 113
    assert result["student_controls"] == 64 and result["expert_controls"] == 26
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}
    )
    fig = plt.figure(figsize=(12.6, 7.8))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.16,
        right=0.98,
        top=0.80,
        bottom=0.14,
        height_ratios=[1, 1, 0.27],
        hspace=0.33,
        wspace=0.045,
    )
    with np.load(args.input, allow_pickle=False) as source:
        assert np.array_equal(source["student_front_rgb"][0], source["expert_front_rgb"][0])
        for row, tag in enumerate(("student", "expert")):
            for col, control in enumerate((64, 66, 68) if row == 0 else (64, 77, 90)):
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(source[tag + "_front_rgb"][col], interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"Control step {control}", fontsize=10, color="#4e6173")
        timeline = fig.add_subplot(grid[2, :])
        timeline.set_xlim(0, 139)
        timeline.set_ylim(-0.1, 1.0)
        timeline.barh(0.58, 49, left=0, height=0.22, color="#dee8f0")
        timeline.barh(0.58, 64, left=49, height=0.22, color="#9caec1")
        timeline.barh(0.58, 26, left=113, height=0.22, color="#388b76")
        timeline.text(24, 0.58, "Conditioning video", ha="center", va="center", fontsize=8)
        timeline.text(80, 0.58, "Student actions", ha="center", va="center", fontsize=8)
        timeline.text(126, 0.58, "Expert", ha="center", va="center", fontsize=8, color="white")
        recent = np.arange(81, 113)
        full = np.linspace(0, 112, 32, dtype=np.int32)
        timeline.scatter(recent, np.full(32, 0.21), marker="|", color="#b77c34", s=35)
        timeline.scatter(full, np.full(32, -0.02), marker="|", color="#337ca0", s=35)
        timeline.text(-1, 0.21, "Recent 32", ha="right", va="center", fontsize=8)
        timeline.text(-1, -0.02, "Full-prefix 32", ha="right", va="center", fontsize=8)
        timeline.set_axis_off()
    fig.text(
        0.025, 0.94, "Can an expert recover a real learner failure?", fontsize=18, color="#183247"
    )
    fig.text(
        0.025,
        0.89,
        "PatternLock · train episode 0 · exact replay to the same intervention state",
        fontsize=11,
        color="#526374",
    )
    fig.text(
        0.025,
        0.68,
        "Released student\nFails at step 68",
        fontsize=11,
        color="#9a572e",
        va="center",
        linespacing=1.6,
    )
    fig.text(
        0.025,
        0.40,
        "Expert takeover\nSuccess: 64 + 26",
        fontsize=11,
        color="#216b56",
        va="center",
        linespacing=1.6,
    )
    fig.text(
        0.025,
        0.035,
        "Retrospective train-only diagnostic; 113 public frames matched exactly. "
        "Raw camera pixels are unchanged.\nMemory bars describe matched training views, "
        "not measured learning gains. "
        "No autonomous recovery trigger or method efficacy is claimed.",
        fontsize=9,
        color="#526374",
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(str(args.output_prefix) + "." + extension, dpi=180, facecolor="white")
    svg = args.output_prefix.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_npz_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "expert_result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
                "source_frame_indices": {"student": [112, 114, 116], "expert": [112, 125, 138]},
                "public_prefix_equal_frames": 113,
                "student_controls_before_intervention": 64,
                "expert_controls": 26,
                "pixel_editing": "NONE",
                "method_efficacy": "NOT_RUN",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
