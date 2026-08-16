#!/usr/bin/env python3
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from core.flow_lenia_jax import (
    FlowLeniaParams, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.metrics import evaluate_run_metrics
from core.visualization import save_experiment_artifacts

def main():
    parser = argparse.ArgumentParser(description="Run Long Multi-Blob Ecosystem Simulation for Flow Lenia")
    parser.add_argument("--patches", type=int, default=6, help="Number of distinct biological patches/species")
    parser.add_argument("--grid_size", type=int, default=384, help="Grid size resolution (e.g. 384 or 512)")
    parser.add_argument("--steps", type=int, default=4000, help="Total simulation steps (e.g. 4000)")
    parser.add_argument("--sample_interval", type=int, default=20, help="Sampling frame interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/hero_ecosystem", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    rng_key = random.PRNGKey(args.seed)
    
    print(f"=== Flow-Lenia Long Ecosystem Hero Run (SOTA Broadcast Quality) ===")
    print(f"Grid: {args.grid_size}x{args.grid_size} | Steps: {args.steps} | Patches: {args.patches} | JAX: {jax.devices()}")
    
    # 9 kernels continuous Gaussian rings
    K = 9
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    
    H, W = args.grid_size, args.grid_size
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    # Motile biological parameters per patch
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (args.patches, K), minval=0.13, maxval=0.22)
    sigma_presets = random.uniform(k_sigma, (args.patches, K), minval=0.011, maxval=0.024)
    
    rng_key, subk = random.split(rng_key)
    state = initialize_multi_patch_state(
        subk, H, W, C=1, K=K,
        n_patches=args.patches,
        kernel_radii=radii,
        mu_presets=mu_presets,
        sigma_presets=sigma_presets
    )
    
    params = FlowLeniaParams(
        mu=mu_presets[0],
        sigma=sigma_presets[0],
        weights=jnp.full((K,), 1.0 / K)
    )
    
    print("[Hero Run] Running Flow Lenia physics simulation...")
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk,
        num_steps=args.steps,
        sample_interval=args.sample_interval,
        mixing_rule='gene_wise',
        enable_mutation=True,
        mutation_interval=50
    )
    
    sampled_mass_np = np.array(sampled_mass)
    sampled_gid_np = np.array(sampled_gid)
    final_mass_np = np.array(final_state.mass)
    
    metrics = evaluate_run_metrics(final_mass_np, sampled_mass_np, sampled_gid_np, total_steps=args.steps, n_genomes=args.patches)
    
    print(f"\n=== HERO RUN METRIC SUMMARY ===")
    print(f"CoM Displacement: {metrics['com_displacement']:.2f} px")
    print(f"Evolutionary Activity (EA): {metrics['ea_raw']:.6f}")
    print(f"Compression Complexity: {metrics['complexity_raw']:.0f} bytes")
    print(f"Multi-Scale Entropy: {metrics['entropy_raw']:.4f} bits")
    
    config = {
        "grid_size": args.grid_size,
        "steps": args.steps,
        "patches": args.patches,
        "seed": args.seed,
        "radii": np.array(radii).tolist()
    }
    
    prefix = f"hero_ecosystem_seed{args.seed}"
    art_paths = save_experiment_artifacts(
        sampled_mass_frames=sampled_mass_np,
        metrics=metrics,
        config=config,
        output_dir=args.output_dir,
        prefix=prefix,
        fps=20,
        genome_id_maps=sampled_gid_np
    )
    
    print(f"\nHero Ecosystem MP4 video ready: {art_paths['video']}")
    print(f"Trajectory Filmstrip ready: {art_paths['filmstrip']}")
    print(f"Motion Heatmap ready: {art_paths['heatmap']}")

if __name__ == "__main__":
    main()
