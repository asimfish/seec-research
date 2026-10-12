#!/usr/bin/env python3
"""Turn a verified fixed pilot into descriptive tables and standalone figures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ["action_only", "current_state_aux", "observed_facts_aux"]
NAMES = {"action_only": "Action-only history BC", "current_state_aux": "Current-state auxiliary",
         "observed_facts_aux": "Observed-fact auxiliary", "symbolic_full_history": "Symbolic full-history reference"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(folder):
    summary = json.loads((folder / "summary.json").read_text())
    verified = json.loads((folder / "verification.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    assert verified["status"] == "PASS" and verified["summary_sha256"] == sha(folder / "summary.json")
    table = {}
    for arm, records in summary["arms"].items():
        table[arm] = {"success_mean": float(np.mean([r["success_rate"] for r in records])),
                      "success_seed_values": [r["success_rate"] for r in records],
                      "mean_cost_steps": float(np.mean([r["mean_cost_steps"] for r in records])),
                      "by_trial": [float(np.mean([r["by_trial"][k]["success_rate"] for r in records])) for k in range(5)],
                      "episodes": sum(r["episodes"] for r in records),
                      "successes": sum(r["successes"] for r in records)}
        if arm != "symbolic_full_history":
            reset = [summary["evaluations"][f"evaluation/{arm}/{s}/reset_each_episode"]
                     for s in plan["training_seeds"]]
            table[arm]["reset_success_mean"] = float(np.mean([r["success_rate"] for r in reset]))
            table[arm]["memory_drop"] = table[arm]["success_mean"] - table[arm]["reset_success_mean"]
    candidate = table["observed_facts_aux"]
    differences = {}
    for baseline in ["action_only", "current_state_aux"]:
        values = [a - b for a, b in zip(candidate["success_seed_values"], table[baseline]["success_seed_values"])]
        differences[baseline] = {"mean": float(np.mean(values)), "by_seed": values,
                                 "all_seeds_positive": all(v > 0 for v in values)}
    h1 = all(r["mean"] >= 0.05 and r["all_seeds_positive"] for r in differences.values()) and candidate["memory_drop"] >= 0.05
    floor = all(table[a]["success_mean"] < 0.2 for a in ARMS)
    result = {"schema": "cooking-learning-pilot-analysis-v1", "plan_id": plan["plan_id"],
              "summary_sha256": sha(folder / "summary.json"), "verification_sha256": sha(folder / "verification.json"),
              "training_seeds": plan["training_seeds"], "arms": table,
              "paired_candidate_differences": differences,
              "prediction_checks": {"H1_grounded_update_supported_in_this_pilot": h1,
                                    "H3_all_learned_arms_below_20_percent": floor,
                                    "symbolic_reference_at_100_percent": table["symbolic_full_history"]["success_mean"] == 1.0},
              "stop_reason": "registered-budget-completed",
              "scope": "Three optimizer seeds; descriptive auxiliary-learning pilot, not new-method, causal-credit, OPD, robot or general task-space efficacy."}
    (folder / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Learned task-history pilot: actual results", "", "The fixed pilot used three optimizer seeds and nine training runs. It is recurrent behavioral cloning with a matched auxiliary-label ablation, not trained DREAM/AD/RL2 or OPD. No hidden assignment, symbolic memory, or teacher is supplied to a learned policy at evaluation.", "",
             "| Arm | Mean task success | Three seed success rates | Cost steps (failure = 20) | Reset-memory success |",
             "|---|---:|---|---:|---:|"]
    for arm, row in table.items():
        reset = f"{100*row['reset_success_mean']:.2f}%" if "reset_success_mean" in row else "—"
        lines.append(f"| {NAMES[arm]} | {100*row['success_mean']:.2f}% | {', '.join(f'{100*v:.2f}%' for v in row['success_seed_values'])} | {row['mean_cost_steps']:.4f} | {reset} |")
    lines += ["", "The primary metric averages the five task attempts. Success-by-attempt curves and each seed are retained in the JSON. The same 20-action exploration cap and 20-step execution horizon apply to every arm. All exploration actions come from each learned policy; there is no teacher prefix. Source is the same registered teacher/perturbed history pool, so this is offline supervision rather than student-state distillation.", "",
              f"Observed-fact auxiliary minus action-only: {100*differences['action_only']['mean']:+.2f} percentage points; minus current-state auxiliary: {100*differences['current_state_aux']['mean']:+.2f} points. Paired seed differences are {differences['action_only']['by_seed']} and {differences['current_state_aux']['by_seed']}.", "",
              f"Registered H1 (>=5 points over both, positive at every seed, memory drop >=5 points): **{'SUPPORTED IN THIS PILOT' if h1 else 'NOT SUPPORTED'}**. All-arm <20% floor: {floor}. Stop: registered-budget-completed; no horizon, loss or checkpoint tuning on this panel.", "",
              "The first run was interrupted for a terminal-feedback input bug and is INVALID. Its plan, code snapshot and partial source/training remain preserved. The valid run adds real terminal feedback-only tokens before episode resets, without executing extra simulator actions or changing task settings. All final checkpoint bytes were saved before development and test evaluation. Each recorded action, observation, reward, outcome, cost and aggregate has been independently recomputed.", "",
              "Three seeds do not support confirmatory inference. The mapping panel reuses the prior symbolic diagnostic split, so it is not an independent confirmatory test set. Symbolic reference numbers are skill/memory upper references, not another trained model. A grounded auxiliary benefit would still require stronger trained DREAM/AD/Bayes and student-state collection controls; a failure does not prove that task-rule learning is unnecessary.", "",
              f"Plan: `{plan['plan_id']}`; source tokens: {summary['dataset']['tokens']:,}; actual source steps: {summary['dataset']['physical_steps']:,}; optimizer updates: {summary['cost']['optimizer_updates']:,}; model decisions independently checked: {verified['recomputed_policy_decisions']:,}."]
    (folder.parent / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result, summary


def plot(result, summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1, 1.12]})
    colors = ["#6b778c", "#168c9c", "#285bd6"]
    labels = ["Action-only", "Current-state\nauxiliary", "Observed-fact\nauxiliary"]
    for i, arm in enumerate(ARMS):
        row = result["arms"][arm]
        axes[0].bar(i, row["success_mean"], color=colors[i], width=.52, alpha=.85)
        axes[0].scatter([i-.1, i, i+.1], row["success_seed_values"], color="#18243a", s=24, zorder=3)
        axes[0].text(i, row["success_mean"]+.035, f"{row['success_mean']:.1%}", ha="center", weight="bold")
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Task success, averaged over five attempts")
    axes[0].set_title("A  Same history, architecture and update budget", loc="left", fontsize=11, pad=14)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylim(0, 1.15)
    axes[0].set_yticks(np.arange(0, 1.01, .2))
    for i, arm in enumerate(ARMS):
        seed_curves = np.array([[r["by_trial"][k]["success_rate"] for k in range(5)] for r in summary["arms"][arm]])
        axes[1].plot(range(1, 6), seed_curves.mean(0), marker="o", lw=2, color=colors[i], label=labels[i].replace("\n", " "))
        axes[1].fill_between(range(1, 6), seed_curves.min(0), seed_curves.max(0), color=colors[i], alpha=.09)
    axes[1].set_title("B  Teacher-free adaptation after learned exploration", loc="left", fontsize=11, pad=14)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_yticks(np.arange(0, 1.01, .2))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xlabel("Attempt on the same unknown mapping and goal")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.5)
    for ax in axes:
        ax.axhline(1, color="#9aa4b4", linestyle="--", lw=1, zorder=0)
        ax.grid(axis="y", color="#e6eaf0", lw=.7)
        ax.set_axisbelow(True)
    fig.text(.07, .012, "24 unseen mappings · 3 optimizer seeds · dots/ranges show seeds, not confidence intervals · symbolic reference: 100%", fontsize=8.5, color="#526079")
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(destination / f"cooking_learned_pilot.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    result, summary = analyze(args.folder)
    if not args.no_plot:
        plot(result, summary, args.folder.parent / "figures")
    print(json.dumps(result, indent=2))
