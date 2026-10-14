#!/usr/bin/env python3
"""Render two unchanged, registered RGB frames as a benchmark input illustration."""

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
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.registration.read_text())["cases"]
    row = next(r for r in rows if r["task"] == "VideoUnmask")
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest != row["input_sha256"]:
        raise ValueError("Figure input differs from the recorded public observation")
    with np.load(args.input, allow_pickle=False) as archive:
        front = archive["front"].copy()
    assert len(front) == row["frames"] == 67
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    for axis, index, title in zip(axes, [0, 66], ["Frame 0: visible colour cue",
                                                "Frame 66: objects covered"]):
        axis.imshow(front[index], interpolation="nearest")
        axis.set_title(title, fontsize=11, pad=9)
        axis.axis("off")
    figure.suptitle("RoboMME: find the container hiding the blue cube", fontsize=13, y=.98)
    note = "Actual train episode 0 · 67 reset frames · input illustration, no policy result"
    figure.text(.5, .04, note,
                ha="center", fontsize=9, color="#526374")
    figure.subplots_adjust(left=.025, right=.975, top=.83, bottom=.13, wspace=.04)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        figure.savefig(str(args.output_prefix) + "." + extension, dpi=180, facecolor="white")
    plt.close(figure)
    args.output_prefix.with_suffix(".json").write_text(json.dumps({
        "source_task": "VideoUnmask", "split": "train", "episode": 0,
        "source_npz_sha256": digest, "frame_indices": [0, 66],
        "source_frame_count": 67, "pixel_editing": "NONE",
        "scope": "Benchmark input illustration, no learned-policy outcome",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
