#!/usr/bin/env python3
"""Paired bounded reproduction of PSP on a stratified GenEval subset."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
T2I = ROOT / "Fk-Diffusion-Steering" / "text_to_image"
sys.path.insert(0, str(T2I))
sys.path.insert(0, str(T2I / "fkd_diffusers"))

# hpsv2's open_clip release imports turtle, which otherwise tries to import Tk.
if "turtle" not in sys.modules:
    turtle = types.ModuleType("turtle")
    turtle.forward = lambda *_args, **_kwargs: None
    sys.modules["turtle"] = turtle

from diffusers import DDIMScheduler
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion, latent_to_decode
from fkd_diffusers.rewards import do_human_preference_score, do_image_reward
from run_psp import estimate_x0_from_ddim_transition

from geneval_hf import ModernGenEval

METHODS = ("standard", "best_of_4", "psp_default", "psp_timing", "oracle_8")


def load_config() -> dict:
    return json.loads((ROOT / "reproduction" / "config.json").read_text())


def stratified_prompts(per_tag: int) -> list[dict]:
    path = T2I / "prompt_files" / "geneval_metadata.jsonl"
    rows = []
    with path.open() as handle:
        for prompt_id, line in enumerate(handle):
            row = json.loads(line)
            row["prompt_id"] = prompt_id
            rows.append(row)
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tag[row["tag"]].append(row)
    chosen = []
    for tag in sorted(by_tag):
        group = by_tag[tag]
        indices = np.linspace(0, len(group) - 1, per_tag).round().astype(int)
        chosen.extend(group[int(i)] for i in indices)
    return sorted(chosen, key=lambda row: row["prompt_id"])


def build_pipeline(model_id: str, device: str):
    pipe = FKDStableDiffusion.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def collect_pool(pipe, prompt: str, seeds: list[int], checkpoints: list[int], steps: int):
    device = pipe.device
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    prompt_list = [prompt] * len(seeds)
    prev_latents = pipe.prepare_latents(
        batch_size=len(seeds),
        num_channels_latents=pipe.unet.config.in_channels,
        height=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        width=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        dtype=pipe.unet.dtype,
        device=device,
        generator=generators,
        latents=None,
    )
    scores: dict[int, list[float]] = {}
    wanted = set(checkpoints)

    def callback(_pipe, step_idx, timestep, kwargs):
        nonlocal prev_latents
        latents_after = kwargs["latents"]
        step_number = step_idx + 1
        if step_number in wanted:
            x0 = estimate_x0_from_ddim_transition(
                scheduler=pipe.scheduler,
                timestep=timestep,
                sample_before_step=prev_latents,
                sample_after_step=latents_after,
                num_inference_steps=steps,
            )
            decoded = latent_to_decode(model=pipe, output_type="pil", latents=x0).detach()
            images = pipe.image_processor.postprocess(decoded, output_type="pil")
            scores[step_number] = [
                float(x)
                for x in do_image_reward(images=images, prompts=prompt_list)
            ]
        prev_latents = latents_after
        return {"latents": latents_after}

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = pipe(
            prompt=prompt_list,
            num_inference_steps=steps,
            eta=0.0,
            generator=generators,
            latents=prev_latents,
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=[
                "latents",
                "prompt_embeds",
                "negative_prompt_embeds",
            ],
        )
    final_latents = output.images if hasattr(output, "images") else output[0]
    decoded = latent_to_decode(model=pipe, output_type="pil", latents=final_latents).detach()
    final_images = pipe.image_processor.postprocess(decoded, output_type="pil")
    scores[steps] = [
        float(x)
        for x in do_image_reward(images=final_images, prompts=prompt_list)
    ]
    torch.cuda.synchronize()
    return final_images, scores, time.perf_counter() - started


def replay_schedule(scores: dict[int, list[float]], schedule: list[list[int]], final_step: int):
    survivors = list(range(len(scores[final_step])))
    trace = []
    for step, keep in schedule:
        ranked = sorted(survivors, key=lambda idx: scores[int(step)][idx], reverse=True)
        survivors = ranked[: int(keep)]
        trace.append({"step": int(step), "survivors": survivors.copy()})
    winner = max(survivors, key=lambda idx: scores[final_step][idx])
    return winner, trace


def clip_alignment(images, prompts, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    batch = torch.stack([preprocess(image) for image in images]).to(device)
    tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        image_features = model.encode_image(batch)
        text_features = model.encode_text(tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    result = (image_features * text_features).sum(-1).float().cpu().tolist()
    del model, batch, tokens, image_features, text_features
    torch.cuda.empty_cache()
    return [float(x) for x in result]


def score_independent(selected_records: list[dict], device: str) -> str:
    images = [row["_image"] for row in selected_records]
    prompts = [row["prompt"] for row in selected_records]
    try:
        values = do_human_preference_score(images=images, prompts=prompts)
        metric = "HPSv2.1"
    except Exception as exc:
        print(f"[rank {dist.get_rank()}] HPS unavailable; using CLIP cosine: {exc!r}", flush=True)
        values = clip_alignment(images, prompts, device)
        metric = "CLIP-ViT-B-32-cosine"
    for row, value in zip(selected_records, values):
        row["independent_alignment"] = float(value)
    return metric


def per_rank(config: dict, rank: int, world: int) -> dict:
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    rank_started = time.time()
    prompts = stratified_prompts(int(config["prompts_per_tag"]))
    local_prompts = prompts[rank::world]
    print(
        f"[rank {rank}] prompts={len(local_prompts)} replicate={config['replicate']} "
        f"ids={[p['prompt_id'] for p in local_prompts]}",
        flush=True,
    )
    pipe = build_pipeline(config["model_id"], device)
    torch.cuda.reset_peak_memory_stats(rank)
    rows = []
    intermediate_pairs: dict[int, list[tuple[float, float]]] = defaultdict(list)
    generation_s = 0.0

    for prompt_index, metadata in enumerate(local_prompts):
        prompt_id = int(metadata["prompt_id"])
        seed_base = int(config["replicate"]) * 1_000_000 + prompt_id * int(config["pool_size"])
        seeds = [seed_base + i for i in range(int(config["pool_size"]))]
        final_images, scores, elapsed = collect_pool(
            pipe,
            metadata["prompt"],
            seeds,
            list(config["checkpoints"]),
            int(config["num_steps"]),
        )
        generation_s += elapsed
        final_step = int(config["num_steps"])
        final_scores = scores[final_step]
        default_winner, default_trace = replay_schedule(
            scores, config["default_schedule"], final_step
        )
        timing_winner, timing_trace = replay_schedule(
            scores, config["timing_schedule"], final_step
        )
        winners = {
            "standard": 0,
            "best_of_4": max(range(4), key=lambda idx: final_scores[idx]),
            "psp_default": default_winner,
            "psp_timing": timing_winner,
            "oracle_8": max(range(8), key=lambda idx: final_scores[idx]),
        }
        oracle_idx = winners["oracle_8"]
        for step in config["checkpoints"]:
            for idx in range(8):
                intermediate_pairs[int(step)].append((scores[int(step)][idx], final_scores[idx]))
        base = {
            "prompt_id": prompt_id,
            "tag": metadata["tag"],
            "prompt": metadata["prompt"],
            "seed_base": seed_base,
            "oracle_index": oracle_idx,
            "oracle_survived_default_first": oracle_idx in default_trace[0]["survivors"],
            "oracle_survived_default_second": oracle_idx in default_trace[1]["survivors"],
            "oracle_survived_timing_first": oracle_idx in timing_trace[0]["survivors"],
            "oracle_survived_timing_second": oracle_idx in timing_trace[1]["survivors"],
            "default_regret_ir": float(final_scores[oracle_idx] - final_scores[default_winner]),
            "timing_regret_ir": float(final_scores[oracle_idx] - final_scores[timing_winner]),
            "pool_final_ir": [float(x) for x in final_scores],
        }
        for method, idx in winners.items():
            row = dict(base)
            row.update(
                {
                    "method": method,
                    "candidate_index": int(idx),
                    "seed": int(seeds[idx]),
                    "image_reward": float(final_scores[idx]),
                    "_image": final_images[idx],
                    "_metadata": metadata,
                }
            )
            rows.append(row)
        print(
            f"[rank {rank}] {prompt_index + 1}/{len(local_prompts)} prompt={prompt_id} "
            f"IR bon4={rows[-4]['image_reward']:.4f} psp={rows[-3]['image_reward']:.4f} "
            f"regret={base['default_regret_ir']:.4f} elapsed={elapsed:.2f}s",
            flush=True,
        )

    independent_metric = score_independent(rows, device)
    evaluator = ModernGenEval(device)
    for idx, row in enumerate(rows):
        ge = evaluator.evaluate(row["_image"], row["_metadata"])
        row["geneval_correct"] = bool(ge["correct"])
        row["geneval_reason"] = ge["reason"]
        del row["_image"], row["_metadata"]
        if (idx + 1) % 10 == 0:
            print(f"[rank {rank}] GenEval {idx + 1}/{len(rows)}", flush=True)

    correlations = {}
    for step, pairs in intermediate_pairs.items():
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        correlations[str(step)] = {
            "pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic),
            "n": len(x),
        }
    payload = {
        "rank": rank,
        "replicate": int(config["replicate"]),
        "independent_metric": independent_metric,
        "rows": rows,
        "correlations": correlations,
        "generation_s": generation_s,
        "rank_elapsed_s": time.time() - rank_started,
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(rank) / (1024**3),
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(rank) / (1024**3),
    }
    out = ROOT / "reproduction" / "outputs"
    out.mkdir(exist_ok=True)
    (out / f"rank_{rank}.json").write_text(json.dumps(payload))
    return payload


def bootstrap_ci(values: list[float], seed: int = 260721591) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    draws = rng.choice(arr, size=(4000, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(config: dict, payloads: list[dict], started: float) -> dict:
    rows = [row for payload in payloads for row in payload["rows"]]
    method_summary = {}
    for method in METHODS:
        cur = [row for row in rows if row["method"] == method]
        tag_scores = {
            tag: mean(float(row["geneval_correct"]) for row in cur if row["tag"] == tag)
            for tag in sorted({row["tag"] for row in cur})
        }
        method_summary[method] = {
            "n_prompts": len(cur),
            "mean_image_reward": mean(row["image_reward"] for row in cur),
            "mean_independent_alignment": mean(row["independent_alignment"] for row in cur),
            "geneval_overall": mean(tag_scores.values()),
            "geneval_by_tag": tag_scores,
        }
    by_prompt = defaultdict(dict)
    for row in rows:
        by_prompt[row["prompt_id"]][row["method"]] = row
    paired_ir = [
        values["psp_default"]["image_reward"] - values["best_of_4"]["image_reward"]
        for values in by_prompt.values()
    ]
    paired_ge = [
        float(values["psp_default"]["geneval_correct"])
        - float(values["best_of_4"]["geneval_correct"])
        for values in by_prompt.values()
    ]
    default_rows = [row for row in rows if row["method"] == "psp_default"]
    timing_rows = [row for row in rows if row["method"] == "psp_timing"]
    correlations = {}
    for step in config["checkpoints"]:
        weighted = [
            payload["correlations"][str(step)]
            for payload in payloads
            if str(step) in payload["correlations"]
        ]
        correlations[str(step)] = {
            "pearson": mean(item["pearson"] for item in weighted),
            "spearman": mean(item["spearman"] for item in weighted),
            "n": sum(item["n"] for item in weighted),
        }
    ir_ci = bootstrap_ci(paired_ir)
    ge_ci = bootstrap_ci(paired_ge)
    return {
        "paper_id": "2607.21591",
        "attempt_cutoff_utc": "2026-07-26T20:40:10.852Z",
        "backend": "kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpus": 4,
        "replicate": int(config["replicate"]),
        "subset": {
            "name": "GenEval stratified bounded subset",
            "prompts": len(by_prompt),
            "per_tag": int(config["prompts_per_tag"]),
            "tags": sorted({row["tag"] for row in rows}),
        },
        "selection_forward_passes": {
            "standard": 64,
            "best_of_4": 256,
            "psp_default": 256,
            "psp_timing": 256,
            "oracle_8": 512,
        },
        "diagnostic_generation_forward_passes_per_prompt": 512,
        "methods": method_summary,
        "paired_psp_minus_bon4": {
            "mean_image_reward": mean(paired_ir),
            "image_reward_95pct_prompt_bootstrap": list(ir_ci),
            "mean_geneval_correct": mean(paired_ge),
            "geneval_95pct_prompt_bootstrap": list(ge_ci),
        },
        "predictiveness": correlations,
        "pruning": {
            "default_mean_oracle_regret_ir": mean(row["default_regret_ir"] for row in default_rows),
            "timing_mean_oracle_regret_ir": mean(row["timing_regret_ir"] for row in timing_rows),
            "default_oracle_survival_step16": mean(
                float(row["oracle_survived_default_first"]) for row in default_rows
            ),
            "default_oracle_survival_step32": mean(
                float(row["oracle_survived_default_second"]) for row in default_rows
            ),
            "timing_oracle_survival_step8": mean(
                float(row["oracle_survived_timing_first"]) for row in timing_rows
            ),
            "timing_oracle_survival_step48": mean(
                float(row["oracle_survived_timing_second"]) for row in timing_rows
            ),
        },
        "independent_metric": payloads[0]["independent_metric"],
        "rank_generation_s": [payload["generation_s"] for payload in payloads],
        "rank_elapsed_s": [payload["rank_elapsed_s"] for payload in payloads],
        "job_elapsed_s": time.time() - started,
        "peak_memory_allocated_gib": max(
            payload["peak_memory_allocated_gib"] for payload in payloads
        ),
        "peak_memory_reserved_gib": max(
            payload["peak_memory_reserved_gib"] for payload in payloads
        ),
        "geneval_backend": (
            "GenEval decision rules with facebook/mask2former-swin-small-coco-instance "
            "via Transformers (mmcv 1.x replaced for Blackwell compatibility)"
        ),
        "per_prompt": rows,
    }


def main():
    started = time.time()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    config = load_config()
    assert world == 4, f"expected four GPUs, got world size {world}"
    payload = per_rank(config, rank, world)
    dist.barrier()
    if rank == 0:
        out = ROOT / "reproduction" / "outputs"
        payloads = [json.loads((out / f"rank_{idx}.json").read_text()) for idx in range(world)]
        result = summarize(config, payloads, started)
        compact = {k: v for k, v in result.items() if k != "per_prompt"}
        print("FINAL_SUMMARY_JSON=" + json.dumps(compact, sort_keys=True), flush=True)
        print("FINAL_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
