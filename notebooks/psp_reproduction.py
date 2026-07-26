# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14.17",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Progressive Seed Pruning: a paired SD1.5 reproduction

    A diffusion generator can produce very different images from one prompt when
    only its initial random noise changes. Progressive Seed Pruning (PSP) tries
    eight noise seeds briefly, drops weaker early previews, and finishes two—using
    the same 256 generator forward passes as fully denoising four seeds.

    **Verdict: partially reproduced.** Across five fresh seed windows, PSP improved
    ImageReward over matched-compute Best-of-4 by **+0.160** (repeat-level 95%
    interval **+0.058 to +0.262**), close to the paper's **+0.172**. The bounded
    GenEval score was unchanged within uncertainty.

    This notebook embeds the completed evidence; opening it does not rerun
    diffusion inference.
    """)
    return


@app.cell
def _():
    evidence = {
        "methods": {
            "Standard": {"image_reward": -0.0651848312, "clip": 0.3177754720, "geneval": 0.2500000000, "forwards": 64},
            "Best-of-4": {"image_reward": 0.7759737460, "clip": 0.3298629761, "geneval": 0.3666666667, "forwards": 256},
            "PSP 8→4→2": {"image_reward": 0.9359526203, "clip": 0.3318613688, "geneval": 0.3625000000, "forwards": 256},
            "Timing 8→4→2": {"image_reward": 0.8323751913, "clip": 0.3301300049, "geneval": 0.3375000000, "forwards": 256},
            "Oracle-8": {"image_reward": 1.0118617079, "clip": 0.3319417318, "geneval": 0.3666666667, "forwards": 512},
        },
        "reward_delta": [0.1715834957, 0.2457090698, 0.1821980332, 0.0231470743, 0.1772566984],
        "clip_delta": [-0.0018615723, 0.0031687419, 0.0056050618, 0.0010401408, 0.0020395915],
        "geneval_delta": [0.0, 0.0, 0.0833333333, -0.0208333333, -0.0833333333],
        "predictiveness": {8: 0.3231524988, 16: 0.6589722472, 24: 0.8281854402, 32: 0.8983030087, 40: 0.9496476419, 48: 0.9767110335, 56: 0.9903136842},
        "regret": {"Default (16, 32)": 0.0759090877, "Timing (8, 48)": 0.1794865167},
        "survival": {"Default first prune": 0.8541666667, "Timing first prune": 0.6541666667},
        "compute": {
            "backend": "OpenResearch Kubernetes",
            "gpu": "NVIDIA RTX PRO 6000 Blackwell",
            "peak_gpus": 16,
            "gpus_per_run": 4,
            "job_seconds_mean": 120.5359308,
            "peak_memory_gib": 28.0330615,
            "attempt_wall_hours": 0.41,
        },
    }
    return (evidence,)


@app.function
def headline_svg(methods):
    names = ["Standard", "Best-of-4", "PSP 8→4→2"]
    colors = ["#9aa5b1", "#64748b", "#0f9d8a"]
    bars = []
    for index, name in enumerate(names):
        value = methods[name]["image_reward"]
        x = 105 + index * 180
        zero_y = 330
        height = abs(value) * 215
        y = zero_y - height if value >= 0 else zero_y
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="110" height="{height:.1f}" rx="6" fill="{colors[index]}"/>'
            f'<text x="{x + 55}" y="{y - 10 if value >= 0 else y + height + 22:.1f}" text-anchor="middle" font-weight="700">{value:.3f}</text>'
            f'<text x="{x + 55}" y="365" text-anchor="middle">{name}</text>'
        )
    return f"""
    <svg viewBox="0 0 650 410" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ImageReward by method">
      <style>text{{font-family:Inter,Arial,sans-serif;fill:#182230;font-size:16px}}</style>
      <rect width="650" height="410" fill="#fbfcfe"/>
      <text x="28" y="35" font-size="22" font-weight="700">Matched-compute ImageReward</text>
      <text x="28" y="60" fill="#52606d">Five seed windows × 48 prompts</text>
      <line x1="70" y1="330" x2="610" y2="330" stroke="#9aa5b1"/>
      <line x1="70" y1="90" x2="70" y2="350" stroke="#9aa5b1"/>
      {''.join(bars)}
      <text x="325" y="397" text-anchor="middle" fill="#087f6d" font-weight="700">PSP − Best-of-4 = +0.160 (paper: +0.172)</text>
    </svg>
    """


@app.cell
def _(evidence, mo):
    mo.Html(headline_svg(evidence["methods"]))
    return


@app.cell
def _(evidence, mo):
    method_rows = "\n".join(
        f"| {name} | {values['forwards']} | {values['image_reward']:.3f} | {values['clip']:.4f} | {values['geneval']:.3f} |"
        for name, values in evidence["methods"].items()
    )
    mo.md(
        f"""
        ## The paired experiment

        All methods select from the same initial seed pool. The released
        deterministic 64-step DDIM path produces intermediate clean-image
        estimates; ImageReward scores those estimates; schedule replay decides
        which candidates survive. Oracle-8 is diagnostic only.

        | Method | Generator forwards | ImageReward ↑ | CLIP ↑ | GenEval ↑ |
        |---|---:|---:|---:|---:|
        {method_rows}

        The 48-prompt subset contains eight prompts from each GenEval category:
        single object, two objects, counting, colors, position, and color
        attribution. Five non-overlapping seed windows make 240 paired
        prompt-window observations.
        """
    )
    return


@app.cell
def _(mo):
    replicate_pick = mo.ui.slider(0, 4, value=0, label="Inspect seed window")
    replicate_pick
    return (replicate_pick,)


@app.cell
def _(evidence, mo, replicate_pick):
    repeat_index = replicate_pick.value
    mo.md(
        f"""
        **Seed window {repeat_index}:**
        ImageReward difference = **{evidence["reward_delta"][repeat_index]:+.3f}**;
        CLIP difference = **{evidence["clip_delta"][repeat_index]:+.4f}**;
        GenEval difference = **{evidence["geneval_delta"][repeat_index]:+.3f}**.

        Every repeat favors PSP on its selection reward, but the independent
        scores move in both directions.
        """
    )
    return


@app.cell
def _(evidence, mo):
    correlation_rows = "\n".join(
        f"| {step} | {rho:.3f} |"
        for step, rho in evidence["predictiveness"].items()
    )
    mo.md(
        f"""
        ## Mechanism: score reliability controls pruning regret

        | Denoising step | Intermediate→final ImageReward rank correlation |
        |---:|---:|
        {correlation_rows}

        At step 8, correlation is only 0.323. The default first prune waits
        until step 16, where it is 0.659; by step 32 it is 0.898. Consequently,
        the eventual best seed survives the default first prune **85.4%** of
        the time versus **65.4%** for the step-8 ablation. Mean oracle regret is
        **0.076** for the default schedule and **0.179** for the timing
        ablation.

        Both schedules cost 256 generator forwards, yet default minus timing is
        +0.104 ImageReward, +0.0017 CLIP, and +0.025 GenEval. The allocation
        idea works when previews are predictive enough; an over-eager prune
        wastes the larger initial search.
        """
    )
    return


@app.cell
def _(evidence, mo):
    compute = evidence["compute"]
    mo.md(
        f"""
        ## Assessment and provenance

        The paper's **selection-reward claim aligns** in direction and size:
        observed +0.160 versus reported +0.172. Transfer to independent CLIP
        (+0.0020, interval crossing zero) and bounded GenEval (−0.004, wide
        interval) is **inconclusive under this setup**. The mechanism claim
        aligns: later previews predict final rankings better, and pruning at
        step 8 increases regret.

        Fresh evidence ran on **{compute["backend"]}**, using
        **{compute["gpu"]}** GPUs. Each run allocated {compute["gpus_per_run"]}
        GPUs; peak concurrency was **{compute["peak_gpus"]} GPUs**. Mean
        measured experiment time was {compute["job_seconds_mean"]:.1f} seconds
        per formal job, peak allocated memory was
        {compute["peak_memory_gib"]:.2f} GiB per GPU, and setup-to-last-evidence
        wall time was **{compute["attempt_wall_hours"]:.2f} hours**.

        Declared substitutions: 48/553 prompts; Transformers Mask2Former with
        original GenEval decision rules instead of mmcv 1.x; CLIP ViT-B/32
        instead of HPSv2; SD1.5 only. The exact fixed run command was
        `bash reproduction/run.sh`.

        [Detailed illustrated report](https://github.com/alphaXiv/psp-cb6b5ed0/blob/main/reports/psp-sd15/report.md)
        · [aggregated evidence](https://github.com/alphaXiv/psp-cb6b5ed0/blob/main/results/reproduction/aggregate.json)
        · [paper](https://arxiv.org/abs/2607.21591)
        """
    )
    return


if __name__ == "__main__":
    app.run()
