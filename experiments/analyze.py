"""Analyze experiment results and generate figures and summary tables.

Usage:
    python experiments/analyze.py [--input results/results.csv] [--outdir results]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import site
import sys
from pathlib import Path

# Add user site packages if present
for p in [
    site.getusersitepackages(),
    r"C:\Users\Sai\AppData\Roaming\Python\Python315\site-packages",
    r"C:\Users\Sai\AppData\Local\Python\pythoncore-3.15-64\Lib\site-packages",
]:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def fig1_score_vs_budget(df: pd.DataFrame, outdir: str) -> None:
    """Figure 1: Best score vs token budget with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    policies = sorted(df["policy"].unique())
    # Distinct styling for cost_aware_sh (highlighted)
    palette = {
        "cost_aware_sh": ("#D9381E", 2.8, "-o", 10),
        "exhaustive_shared": ("#1E88E5", 2.0, "--s", 9),
        "exhaustive_no_sharing": ("#757575", 1.8, ":^", 8),
        "beam": ("#00897B", 1.8, "-.d", 7),
        "greedy": ("#FB8C00", 1.8, "-.v", 6),
        "sh_cost_blind": ("#8E24AA", 1.8, "-x", 5),
        "random_prune": ("#B0BEC5", 1.5, ":+", 4),
    }

    for policy in policies:
        pdf = df[df["policy"] == policy]
        stats_df = pdf.groupby("budget_frac")["best_score"].agg(["mean", "sem", "count"]).reset_index()

        color, lw, marker, zorder = palette.get(policy, ("#333333", 1.5, "-o", 5))
        # 95% CI
        ci95 = stats_df["sem"].fillna(0) * 1.96

        ax.plot(
            stats_df["budget_frac"] * 100,
            stats_df["mean"],
            marker,
            label=policy,
            color=color,
            linewidth=lw,
            markersize=6,
            zorder=zorder,
        )
        ax.fill_between(
            stats_df["budget_frac"] * 100,
            stats_df["mean"] - ci95,
            stats_df["mean"] + ci95,
            color=color,
            alpha=0.15,
            zorder=zorder - 1,
        )

    ax.set_title("Figure 1: Best Branch Score vs. Token Budget Fraction", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Token Budget (% of Exhaustive Unshared Cost)", fontsize=11)
    ax.set_ylabel("Best Score (Fraction of Unit Tests Passed)", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([25, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Policy", fontsize=9, title_fontsize=10, loc="lower right", framealpha=0.9)

    outpath = os.path.join(outdir, "fig1_score_vs_budget.png")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Generated: {outpath}")


def fig2_regret(df: pd.DataFrame, outdir: str) -> None:
    """Figure 2: Regret vs Oracle per policy."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    policies = sorted(df["policy"].unique())
    palette = {
        "cost_aware_sh": ("#D9381E", 2.8, "-o", 10),
        "exhaustive_shared": ("#1E88E5", 2.0, "--s", 9),
        "exhaustive_no_sharing": ("#757575", 1.8, ":^", 8),
        "beam": ("#00897B", 1.8, "-.d", 7),
        "greedy": ("#FB8C00", 1.8, "-.v", 6),
        "sh_cost_blind": ("#8E24AA", 1.8, "-x", 5),
        "random_prune": ("#B0BEC5", 1.5, ":+", 4),
    }

    for policy in policies:
        pdf = df[df["policy"] == policy]
        stats_df = pdf.groupby("budget_frac")["regret"].agg(["mean", "sem"]).reset_index()

        color, lw, marker, zorder = palette.get(policy, ("#333333", 1.5, "-o", 5))
        ci95 = stats_df["sem"].fillna(0) * 1.96

        ax.plot(
            stats_df["budget_frac"] * 100,
            stats_df["mean"],
            marker,
            label=policy,
            color=color,
            linewidth=lw,
            markersize=6,
            zorder=zorder,
        )
        ax.fill_between(
            stats_df["budget_frac"] * 100,
            stats_df["mean"] - ci95,
            stats_df["mean"] + ci95,
            color=color,
            alpha=0.15,
            zorder=zorder - 1,
        )

    ax.set_title("Figure 2: Regret vs. Oracle Across Token Budgets", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Token Budget (% of Exhaustive Unshared Cost)", fontsize=11)
    ax.set_ylabel("Regret (Oracle Score - Best Score)", fontsize=11)
    ax.set_xticks([25, 50, 75, 100])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Policy", fontsize=9, title_fontsize=10, loc="upper right", framealpha=0.9)

    outpath = os.path.join(outdir, "fig2_regret.png")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Generated: {outpath}")


def export_tables(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Compute summary statistics and export Markdown and LaTeX tables."""
    summary = df.groupby("policy").agg(
        mean_tokens=("tokens_used", "mean"),
        mean_best_score=("best_score", "mean"),
        oracle_hit_rate=("oracle_hit", "mean"),
        mean_regret=("regret", "mean"),
        mean_leaves=("leaves_explored", "mean"),
    ).reset_index()

    summary = summary.sort_values(by="mean_best_score", ascending=False)

    # Format numeric values
    formatted = summary.copy()
    formatted["mean_tokens"] = formatted["mean_tokens"].map(lambda x: f"{x:,.0f}")
    formatted["mean_best_score"] = formatted["mean_best_score"].map(lambda x: f"{x:.4f}")
    formatted["oracle_hit_rate"] = formatted["oracle_hit_rate"].map(lambda x: f"{x * 100:.1f}%")
    formatted["mean_regret"] = formatted["mean_regret"].map(lambda x: f"{x:.4f}")
    formatted["mean_leaves"] = formatted["mean_leaves"].map(lambda x: f"{x:.1f}")

    rename_map = {
        "policy": "Policy",
        "mean_tokens": "Mean Tokens",
        "mean_best_score": "Mean Best Score",
        "oracle_hit_rate": "Oracle Hit Rate",
        "mean_regret": "Mean Regret",
        "mean_leaves": "Mean Leaves Explored",
    }
    formatted = formatted.rename(columns=rename_map)

    # Export Markdown
    md_content = "# Policy Evaluation Summary\n\n" + formatted.to_markdown(index=False) + "\n"
    md_path = os.path.join(outdir, "table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Generated: {md_path}")

    # Export LaTeX (self-contained, no jinja2 dependency required)
    tex_path = os.path.join(outdir, "table.tex")
    cols = list(formatted.columns)
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Exploration Policy Benchmark: Tokens, Quality, and Oracle Hit Rate}",
        "\\label{tab:policy_comparison}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in formatted.iterrows():
        tex_lines.append(" & ".join(str(row[c]).replace("%", "\\%") for c in cols) + " \\\\")
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Generated: {tex_path}")

    return summary


def print_cost_savings(summary: pd.DataFrame) -> None:
    """Print comparative token savings and score differences."""
    by_policy = summary.set_index("policy")

    print("\n" + "=" * 65)
    print("EMPIRICAL FINDINGS: COST-AWARE SUCCESSIVE HALVING")
    print("=" * 65)

    if "cost_aware_sh" in by_policy.index:
        sh_tokens = by_policy.loc["cost_aware_sh", "mean_tokens"]
        sh_score = by_policy.loc["cost_aware_sh", "mean_best_score"]
        sh_hit = by_policy.loc["cost_aware_sh", "oracle_hit_rate"]

        print(f"cost_aware_sh performance:")
        print(f"  Mean Tokens:       {sh_tokens:,.0f}")
        print(f"  Mean Best Score:   {sh_score:.4f}")
        print(f"  Oracle Hit Rate:   {sh_hit * 100:.1f}%\n")

        if "exhaustive_no_sharing" in by_policy.index:
            ens_tokens = by_policy.loc["exhaustive_no_sharing", "mean_tokens"]
            ens_score = by_policy.loc["exhaustive_no_sharing", "mean_best_score"]
            sav_ens = ((ens_tokens - sh_tokens) / ens_tokens) * 100
            print(f"vs. exhaustive_no_sharing:")
            print(f"  Token Savings:     {sav_ens:.1f}%")
            print(f"  Score Delta:       {sh_score - ens_score:+.4f}")

        if "exhaustive_shared" in by_policy.index:
            es_tokens = by_policy.loc["exhaustive_shared", "mean_tokens"]
            es_score = by_policy.loc["exhaustive_shared", "mean_best_score"]
            sav_es = ((es_tokens - sh_tokens) / es_tokens) * 100
            print(f"\nvs. exhaustive_shared (AgentGit baseline):")
            print(f"  Token Savings:     {sav_es:.1f}%")
            print(f"  Score Delta:       {sh_score - es_score:+.4f}")

        if "sh_cost_blind" in by_policy.index:
            cb_tokens = by_policy.loc["sh_cost_blind", "mean_tokens"]
            cb_score = by_policy.loc["sh_cost_blind", "mean_best_score"]
            print(f"\nvs. sh_cost_blind (Cost-Blind Ablation):")
            print(f"  Token Delta:       {((sh_tokens - cb_tokens) / cb_tokens) * 100:+.1f}%")
            print(f"  Score Delta:       {sh_score - cb_score:+.4f}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results.")
    parser.add_argument("--input", type=str, default="results/results.csv", help="Input CSV path.")
    parser.add_argument("--outdir", type=str, default="results", help="Directory for output figures and tables.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist. Run run_budget_sweep.py first.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} experiment rows from {args.input}")

    fig1_score_vs_budget(df, args.outdir)
    fig2_regret(df, args.outdir)
    summary = export_tables(df, args.outdir)
    print_cost_savings(summary)


if __name__ == "__main__":
    main()

