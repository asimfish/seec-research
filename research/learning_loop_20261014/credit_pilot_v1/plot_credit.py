#!/usr/bin/env python3
"""Post-run vector plots. Read summaries; never alter selection or model evidence."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    "source_success": "Source success",
    "source_progress": "Source progress",
    "feedback_cold_gain": "Cold-start credit",
    "feedback_total_gain": "Total-performance credit",
    "meta_final_gain": "Final-adaptation credit",
    "adaptation_gain": "Adaptation-gain credit",
}
COLORS = ["#3864a8", "#d88b32", "#26877c"]


def save(figure, folder, name):
    for suffix in ["svg", "pdf", "png"]:
        figure.savefig(folder / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot(run_folder):
    analysis = json.loads((run_folder / "analysis.json").read_text())
    summary = json.loads((run_folder / "summary.json").read_text())
    seeds = sorted(analysis["by_seed"], key=int)
    folder = run_folder.parent / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
        }
    )
    scores = list(NAMES)
    figure, axes = plt.subplots(
        1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1.15, 1.15, 1]}
    )
    for i, seed in enumerate(seeds):
        values = analysis["by_seed"][seed]
        offset = (i - 1) * 0.12
        for j, score in enumerate(scores):
            rho = values["rank_correlations"][score]
            chosen = values["selections"][score]
            if rho is not None:
                axes[0].scatter(rho, j + offset, color=COLORS[i], s=28, zorder=3)
            else:
                axes[0].text(-1.04, j + offset, "n/a", fontsize=6, color=COLORS[i])
            if chosen:
                axes[1].scatter(
                    100 * chosen["long_audit_adaptation_gain"],
                    j + offset,
                    color=COLORS[i],
                    s=28,
                    zorder=3,
                )
        if values["uniform_adaptation_gain"] is not None:
            axes[1].scatter(
                100 * values["uniform_adaptation_gain"], 6 + offset, color=COLORS[i], s=28, zorder=3
            )
        rows = [r for r in analysis["candidates"] if str(r["seed"]) == seed]
        axes[2].scatter(
            [100 * r["adaptation_gain"] for r in rows],
            [100 * r["long_audit_adaptation_gain"] for r in rows],
            s=33,
            color=COLORS[i],
            label=f"Seed {seed}",
            alpha=0.85,
        )
    for ax in axes:
        ax.axvline(0, color="#b8c2cd", linewidth=0.8, zorder=0)
        ax.grid(axis="x", color="#e4e9ef", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set(
        yticks=range(6),
        yticklabels=list(NAMES.values()),
        xlim=(-1.1, 1.1),
        xlabel="Spearman ranking vs. longer-training utility",
        title="A   Ranking prediction",
    )
    axes[1].set(
        yticks=range(7),
        yticklabels=[*NAMES.values(), "Uniform eligible candidate"],
        xlabel="Long audit adaptation gain (percentage points)",
        title="B   Selected experience",
    )
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[2].axhline(0, color="#b8c2cd", linewidth=0.8)
    axes[2].set(
        xlabel="Short feedback adaptation gain (pp)",
        ylabel="Long audit adaptation gain (pp)",
        title="C   All 24 candidates",
    )
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Does short experience credit predict longer-training adaptation?", fontsize=12, y=1.03
    )
    figure.text(
        0.02,
        -0.045,
        (
            "Three training seeds · fixed candidate bank · training-side Cooking audit · "
            "no confirmatory inference"
        ),
        color="#536176",
        fontsize=8,
    )
    figure.tight_layout(w_pad=2.2)
    save(figure, folder, "credit_ranking")

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, seed in zip(axes, seeds):
        rows = [r for r in summary["results"] if str(r["seed"]) == seed]
        baseline = rows[0]["baseline"]["audit"]["success_by_attempt"]
        ax.plot(
            range(6),
            np.array(baseline) * 100,
            color="#68778a",
            linestyle="--",
            marker="o",
            markersize=3,
            label="Initial learner",
        )
        for name, color in [
            ("feedback_total_gain", "#3864a8"),
            ("meta_final_gain", "#d88b32"),
            ("adaptation_gain", "#26877c"),
        ]:
            selected = analysis["by_seed"][seed]["selections"][name]
            if selected is None:
                continue
            row = next(r for r in rows if r["candidate"] == selected["candidate"])
            ax.plot(
                range(6),
                np.array(row["long"]["audit"]["success_by_attempt"]) * 100,
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=NAMES[name],
            )
        ax.set(
            title=f"Training seed {seed}",
            xlabel="Previous own-policy attempts",
            ylim=(-3, 103),
            xticks=range(6),
        )
        ax.grid(color="#e4e9ef", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Audit success (%) · rule-macro average")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.11),
        fontsize=8,
    )
    figure.suptitle(
        "First-attempt competence and subsequent adaptation remain separate", fontsize=12
    )
    figure.tight_layout()
    save(figure, folder, "credit_adaptation_curves")

    diagnostic = json.loads((run_folder / "metric_diagnostic.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    mean = diagnostic["means"]
    labels = [
        "First-attempt success",
        "Later-attempt success",
        "Adaptation proxy",
        "Total performance",
    ]
    keys = [
        "first_change",
        "later_success_change",
        "adaptation_metric_change",
        "total_performance_change",
    ]
    values = [100 * mean[k] for k in keys]
    axes[0].barh(range(4), values, color=["#d88b32", "#3864a8", "#26877c", "#68778a"], height=0.55)
    axes[0].set(
        yticks=range(4),
        yticklabels=labels,
        xlabel="Mean change (percentage points)",
        title="A   Same selected experiences",
    )
    axes[0].invert_yaxis()
    for i, value in enumerate(values):
        inside = value < -5
        axes[0].text(
            value + (0.35 if value >= 0 or inside else -0.35),
            i,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 or inside else "right",
            color="white" if inside else "black",
            fontsize=9,
        )
    rows = diagnostic["by_seed"]
    positions = np.arange(len(rows))
    for shift, key, label, color in [
        (-0.25, "first_change", "First-attempt success", "#d88b32"),
        (0, "later_success_change", "Later success", "#3864a8"),
        (0.25, "adaptation_metric_change", "Adaptation proxy", "#26877c"),
    ]:
        axes[1].bar(
            positions + shift, [100 * r[key] for r in rows], width=0.24, color=color, label=label
        )
    axes[1].set(
        xticks=positions,
        xticklabels=[f"Seed {r['seed']}" for r in rows],
        ylabel="Change (percentage points)",
        title="B   All three seeds",
    )
    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, fontsize=8,
                  loc="lower center", ncol=3, bbox_to_anchor=(0.64, -0.105))
    axes[0].axvline(0, color="#64748b", linewidth=0.8)
    axes[1].axhline(0, color="#64748b", linewidth=0.8)
    for ax in axes:
        ax.grid(color="#e4e9ef", linewidth=0.6)
        ax.set_axisbelow(True)
    figure.suptitle("Larger adaptation gaps accompany first-attempt degradation", fontsize=12)
    figure.text(
        0.02,
        -0.16,
        ("Post-hoc arithmetic decomposition · registered competence-retention gate failed · "
         "no new-method efficacy"),
        fontsize=8,
        color="#536176",
    )
    figure.tight_layout()
    save(figure, folder, "credit_metric_decomposition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    plot(parser.parse_args().run)
