"""Post-run scientific plots, never alter registered analysis or trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(folder):
    analysis = json.loads((folder / "analysis.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    names = {
        "visible_random": "Visible random",
        "bayes_reset": "Belief reset",
        "bayes_carry": "Belief retained",
        "privileged_greedy": "Privileged greedy",
    }
    colors = {
        "visible_random": "#8a97a8",
        "bayes_reset": "#d88b32",
        "bayes_carry": "#26877c",
        "privileged_greedy": "#3864a8",
    }
    for arm, name in names.items():
        axes[0].plot(
            range(1, 11),
            analysis["by_arm"][arm]["trial_reward_means"],
            label=name,
            color=colors[arm],
            marker="o",
            markersize=3,
            linestyle="--" if arm == "privileged_greedy" else "-",
        )
    axes[0].set(
        title="A   All ten trial means",
        xlabel="Trial / changing items",
        ylabel="Reward per trial",
        ylim=(0, 46),
    )
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    differences = analysis["paired_carry_minus_reset_last_five"]["by_seed"]
    axes[1].bar(
        range(8),
        differences,
        color=["#26877c" if v > 0 else "#d88b32" for v in differences],
        width=0.6,
    )
    axes[1].axhline(0, color="#67768a", linewidth=0.8)
    axes[1].axhline(
        2, color="#3864a8", linestyle="--", linewidth=1, label="Registered mean-gap target"
    )
    axes[1].set(
        title="B   Retained minus reset",
        xlabel="Chemistry seed",
        ylabel="Last-five mean reward difference",
        xticks=range(8),
        xticklabels=[str(i) for i in range(4101, 4109)],
    )
    axes[1].tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    for arm in ["bayes_reset", "bayes_carry"]:
        rows = [r for r in summary["episodes"] if r["arm"] == arm]
        median = np.median([r["belief_world_states_at_trial_start"] for r in rows], axis=0)
        axes[2].plot(
            range(1, 11), median, color=colors[arm], marker="o", markersize=3, label=names[arm]
        )
    axes[2].set(
        title="C   Information retained",
        xlabel="Trial start",
        ylabel="Median possible worlds (log scale)",
        yscale="log",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#e4e9ef", linewidth=0.6)
        ax.set_axisbelow(True)
    figure.suptitle(
        "Alchemy E0: belief narrows, registered performance gap not established", fontsize=12
    )
    figure.tight_layout()
    figure.text(
        0.02,
        -0.06,
        ("8 chemistries · fixed 20 actions/trial · zero gradients · "
         "myopic reference, not a full ideal observer"),
        color="#536176",
        fontsize=8,
    )
    output = folder.parent / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ["svg", "png", "pdf"]:
        figure.savefig(output / ("alchemy_e0." + suffix), dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    plot(parser.parse_args().run)
