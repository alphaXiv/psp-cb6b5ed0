import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    _intro_md = r"""# Progressive seed pruning, reproduced on SD1.5

A diffusion generator can make very different images from the same prompt because each run starts from different random noise. Progressive Seed Pruning (PSP) tries eight starts, scores partially formed images, then finishes only the most promising two. We tested whether that allocation beats fully denoising four candidates at the same 256-pass budget.

**Verdict: partially reproduced.** Across five fresh seed windows and 240 paired prompt/seed pools, PSP improved ImageReward by **+0.160** (95% bootstrap **+0.092 to +0.232**), close to the paper's +0.172. GenEval changed by **−0.004** (−0.042 to +0.033), rather than the paper's +0.032.
"""
    mo.md(_intro_md)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell
def _():
    evidence = {
        "paper": {
            "image_reward_gain": 0.172,
            "geneval_gain": 0.032,
        },
        "observed": {
            "image_reward_gain": 0.159979,
            "image_reward_ci": (0.091861, 0.232287),
            "geneval_gain": -0.004167,
            "geneval_ci": (-0.041667, 0.033333),
            "clip_gain": 0.002001,
            "clip_ci": (-0.000261, 0.004368),
        },
        "method_means": {
            "Standard": {"ImageReward": -0.065185, "GenEval": 0.250000, "CLIP": 0.317759},
            "Best-of-N=4": {"ImageReward": 0.775974, "GenEval": 0.366667, "CLIP": 0.329839},
            "PSP default": {"ImageReward": 0.935953, "GenEval": 0.362500, "CLIP": 0.331840},
            "PSP timing": {"ImageReward": 0.832375, "GenEval": 0.337500, "CLIP": 0.330110},
            "Oracle-8": {"ImageReward": 1.011862, "GenEval": 0.366667, "CLIP": 0.331922},
        },
        "predictiveness": {
            8: 0.323152,
            16: 0.658972,
            24: 0.828185,
            32: 0.898303,
            40: 0.949648,
            48: 0.976711,
            56: 0.990314,
        },
        "pruning": {
            "default_regret": 0.075909,
            "timing_regret": 0.179487,
            "default_oracle_survival": 0.7875,
            "timing_oracle_survival": 0.6500,
        },
        "compute": {
            "backend": "Kubernetes",
            "gpu": "NVIDIA RTX PRO 6000 Blackwell",
            "peak_concurrent_gpus": 16,
            "elapsed_wall_hours": 0.390545,
            "mean_terminal_job_seconds": 120.719441,
            "peak_allocated_gib_per_gpu": 28.033062,
        },
    }
    return (evidence,)


@app.cell
def _(evidence, np, plt):
    fig_head, axes_head = plt.subplots(1, 2, figsize=(11, 4.4))
    paper_color, observed_color = "#8795a1", "#2b7a78"
    panels_head = [
        ("ImageReward", "image_reward_gain", "image_reward_ci"),
        ("GenEval correctness", "geneval_gain", "geneval_ci"),
    ]
    for ax_head, (title_head, gain_key, ci_key) in zip(axes_head, panels_head):
        paper_gain = evidence["paper"][gain_key]
        observed_gain = evidence["observed"][gain_key]
        observed_ci = evidence["observed"][ci_key]
        ax_head.barh(["Paper", "Observed"], [paper_gain, observed_gain],
                     color=[paper_color, observed_color], height=0.55)
        ax_head.errorbar(
            observed_gain,
            1,
            xerr=np.array([[observed_gain - observed_ci[0]], [observed_ci[1] - observed_gain]]),
            color="#d35400",
            capsize=5,
            linewidth=2.2,
        )
        ax_head.axvline(0, color="#26343d", linewidth=1)
        ax_head.set_title(title_head)
        ax_head.set_xlabel("PSP − Best-of-N=4")
        ax_head.grid(axis="x", alpha=0.2)
    fig_head.suptitle("Matched-compute headline result", fontsize=15, fontweight="bold")
    fig_head.tight_layout()
    fig_head
    return


@app.cell
def _(mo):
    _budget_md = r"""The left panel aligns closely with the paper: selecting with ImageReward produced a robust gain. The right panel is the important qualification. On this bounded subset and substituted detector stack, PSP did not improve GenEval.

## Reconstructing the budget

The comparison is paired: every method sees the same eight initial seeds for a prompt.

| Rule | Denoising allocation | Passes |
|---|---:|---:|
| Standard | 1 seed × 64 steps | 64 |
| Best-of-N=4 | 4 seeds × 64 steps | 256 |
| PSP default | 8×16 + 4×16 + 2×32 | 256 |
| PSP timing | 8×8 + 4×40 + 2×16 | 256 |
| Oracle diagnostic | 8 seeds × 64 steps | 512 |

Oracle-8 is diagnostic only: completing all eight candidates reveals whether pruning discarded the eventual ImageReward winner.
"""
    mo.md(_budget_md)
    return


@app.cell
def _(mo):
    metric_choice = mo.ui.radio(
        options=["ImageReward", "GenEval", "CLIP"],
        value="ImageReward",
        label="Inspect method means",
        inline=True,
    )
    metric_choice
    return (metric_choice,)


@app.cell
def _(evidence, metric_choice, plt):
    chosen_metric = metric_choice.value
    labels_metric = list(evidence["method_means"])
    values_metric = [evidence["method_means"][name_metric][chosen_metric] for name_metric in labels_metric]
    colors_metric = ["#8795a1", "#557a95", "#2b7a78", "#d35400", "#6c5b7b"]
    fig_metric, ax_metric = plt.subplots(figsize=(9.5, 4.2))
    ax_metric.bar(labels_metric, values_metric, color=colors_metric)
    ax_metric.set_ylabel(chosen_metric)
    ax_metric.set_title(f"Final {chosen_metric} by selection rule")
    ax_metric.tick_params(axis="x", rotation=18)
    ax_metric.grid(axis="y", alpha=0.2)
    fig_metric.tight_layout()
    fig_metric
    return


@app.cell
def _(evidence, mo):
    clip_gain = evidence["observed"]["clip_gain"]
    clip_low, clip_high = evidence["observed"]["clip_ci"]
    mo.md(
        f"""
        ## Independent alignment check

        CLIP ViT-B/32 cosine was substituted for unavailable HPSv2 assets. Its paired gain was **{clip_gain:+.4f}** with a 95% interval **[{clip_low:+.4f}, {clip_high:+.4f}]**. The direction agrees with ImageReward, but the interval crosses zero, so it does not independently establish an alignment improvement.
        """
    )
    return


@app.cell
def _(evidence, plt):
    steps_pred = list(evidence["predictiveness"])
    correlations_pred = list(evidence["predictiveness"].values())
    fig_pred, ax_pred = plt.subplots(figsize=(9.5, 4.2))
    ax_pred.plot(steps_pred, correlations_pred, marker="o", linewidth=3, color="#2b7a78")
    ax_pred.axvline(16, color="#d35400", linestyle="--", label="default prune")
    ax_pred.axvline(32, color="#d35400", linestyle="--")
    ax_pred.set(xlabel="Denoising step (of 64)", ylabel="Spearman with final ImageReward",
                title="Intermediate estimates become predictive late")
    ax_pred.set_ylim(0, 1.05)
    ax_pred.grid(alpha=0.2)
    ax_pred.legend()
    fig_pred.tight_layout()
    fig_pred
    return


@app.cell
def _(evidence, mo):
    default_regret = evidence["pruning"]["default_regret"]
    timing_regret = evidence["pruning"]["timing_regret"]
    default_survival = evidence["pruning"]["default_oracle_survival"]
    timing_survival = evidence["pruning"]["timing_oracle_survival"]
    mo.md(
        f"""
        ## Why timing matters

        At step 8, intermediate-to-final rank correlation was only 0.323; it reached 0.659 at step 16 and 0.898 at step 32. The default schedule's oracle regret was **{default_regret:.3f}**, versus **{timing_regret:.3f}** when pruning at steps 8 and 48. Eventual-best-seed survival after the second cut fell from **{default_survival:.1%}** to **{timing_survival:.1%}**.

        This is the mechanism in miniature: exploring eight seeds is useful, but an early score must be predictive enough to avoid discarding the seed that would finish best.
        """
    )
    return


@app.cell
def _(evidence, mo):
    compute_nb = evidence["compute"]
    mo.md(
        f"""
        ## Compute, substitutions, and limits

        Formal runs used **{compute_nb["backend"]}** on **{compute_nb["gpu"]}** GPUs: four GPUs per run, **{compute_nb["peak_concurrent_gpus"]}** at peak concurrency, and **{compute_nb["elapsed_wall_hours"]:.3f} elapsed wall-hours** for the fresh attempt. Mean terminal job time was {compute_nb["mean_terminal_job_seconds"]:.1f} seconds and peak allocated memory was {compute_nb["peak_allocated_gib_per_gpu"]:.2f} GiB per GPU.

        The evidence covers five seed windows over 48 unique prompts, not the paper's full 553 prompts. GenEval decision rules were retained, but the original mmcv detector stack was replaced by a public Transformers Mask2Former implementation for Blackwell compatibility. SDXL and gated SD3.5 were not tested. These facts make the appropriate conclusion **partial reproduction**: reward selection and pruning dynamics align, while broader detector-measured alignment does not under this setup.

        Evidence runs: `3b6f5f67`, `4582e3dc`, `1cacce0c`, `0fa5ed77`, `98871d73`. All were created after the recovery cutoff and emitted nonempty terminal JSON logs.
        """
    )
    return


if __name__ == "__main__":
    app.run()
