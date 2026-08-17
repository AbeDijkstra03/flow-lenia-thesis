#!/usr/bin/env python3
"""
Thesis Ablation Experiment 2A: Stochastic Gene-Wise Sampling (Gumbel-Max) & Mutation Multi-Seed Runner.

Evaluates how discrete categorical sampling over incoming directional mass fluxes
prevents parameter blurring into inert gray averages and enables porous multicellular
colonies, self-organizing dividing solitons, and branching lineages.
"""

import os
import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List

from core.flow_lenia_jax import (
    FlowLeniaParams, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.visualization import save_experiment_artifacts
from core.metrics import evaluate_run_metrics

def run_single_gene_mutation_seed(
    seed: int,
    grid_size: int = 384,
    steps: int = 3600,
    sample_interval: int = 3,
    n_patches: int = 6,
    mutation_interval: int = 50,
    base_output_dir: str = "results/gene_mutation"
) -> Dict[str, Any]:
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    rng_key = random.PRNGKey(seed)
    
    H, W = grid_size, grid_size
    K = 9
    
    print(f"\n=======================================================")
    print(f"=== Gene Mutation Ablation (Seed {seed}) ===")
    print(f"=======================================================")
    
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.140, maxval=0.168)
    sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.012, maxval=0.017)
    
    rng_key, subk = random.split(rng_key)
    state = initialize_multi_patch_state(subk, H, W, 1, K, n_patches, radii, mu_presets, sigma_presets)
    params = FlowLeniaParams(
        mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K),
        v_scale=5.8, alpha_diffusion=0.050
    )
    
    print("[Gene Mutation] Running continuous simulation (3600 steps = 1 min HD)...")
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk,
        num_steps=steps,
        sample_interval=sample_interval,
        mixing_rule='gene_wise',
        enable_mutation=True,
        mutation_interval=mutation_interval
    )
    
    sampled_mass_np = np.array(sampled_mass)
    sampled_gid_np = np.array(sampled_gid)
    final_mass_np = sampled_mass_np[-1, 0] if sampled_mass_np.ndim == 4 else sampled_mass_np[-1]
    
    metrics = evaluate_run_metrics(final_mass_np, sampled_mass_np, sampled_gid_np, total_steps=steps, n_genomes=n_patches)
    metrics["method"] = "gene_mutation"
    metrics["mixing_rule"] = "gene_wise"
    metrics["mutation_interval"] = mutation_interval
    
    config = {
        "grid_size": grid_size,
        "steps": steps,
        "patches": n_patches,
        "mutation_interval": mutation_interval,
        "v_scale": 5.8,
        "alpha_diffusion": 0.050,
        "seed": seed
    }
    
    art_paths = save_experiment_artifacts(
        sampled_mass_frames=sampled_mass_np,
        metrics=metrics,
        config=config,
        output_dir=base_output_dir,
        prefix=f"seed_{seed}",
        fps=20,
        genome_id_maps=sampled_gid_np
    )
    print(f"Saved 1-minute video to: {art_paths['video']}")
    
    summary = {
        "seed": seed,
        "metrics": metrics,
        "config": config,
        "artifacts": art_paths
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Gene-Wise Mutation Ablation Multi-Seed Runner")
    parser.add_argument("--grid_size", type=int, default=384, help="Grid resolution")
    parser.add_argument("--steps", type=int, default=3600, help="Simulation steps (3600 = 1 min HD)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Sampling interval")
    parser.add_argument("--patches", type=int, default=6, help="Species count")
    parser.add_argument("--mutation_interval", type=int, default=50, help="Mutation interval")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/gene_mutation", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    multiseed_summaries = []
    for s in args.seeds:
        summary = run_single_gene_mutation_seed(
            seed=s,
            grid_size=args.grid_size,
            steps=args.steps,
            sample_interval=args.sample_interval,
            n_patches=args.patches,
            mutation_interval=args.mutation_interval,
            base_output_dir=args.output_dir
        )
        multiseed_summaries.append(summary)
        
    with open(os.path.join(args.output_dir, "multiseed_summary.json"), "w") as f:
        json.dump(multiseed_summaries, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else str(x))
        
    print(f"\n=======================================================")
    print(f"Gene Mutation completed across all {len(args.seeds)} seeds!")
    print(f"Aggregated summary written to: {args.output_dir}/multiseed_summary.json")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
