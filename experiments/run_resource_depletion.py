#!/usr/bin/env python3
"""
Thesis Experiment 3: Reactive Resource Depletion & Niche Construction.

Systematically evaluates how continuous dynamic resource depletion and regeneration
alter self-organizing soliton dynamics compared to a static, infinite-nutrient baseline.
Measures organism motility, spatial foraging radius, and active lifespan, and generates
comparative transmission plots and dual-panel MP4 videos.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.metrics import evaluate_run_metrics, compute_center_of_mass
from core.visualization import save_experiment_artifacts

def run_depletion_experiment(
    grid_size: int = 256,
    steps: int = 3600,
    sample_interval: int = 3,
    seed: int = 42,
    output_dir: str = "results/resource_depletion"
):
    os.makedirs(output_dir, exist_ok=True)
    rng_key = random.PRNGKey(seed)
    H, W = grid_size, grid_size
    K = 9
    
    print(f"=== Flow-Lenia Resource Depletion & Niche Construction Experiment ===")
    print(f"Grid: {H}x{W} | Steps: {steps} | JAX: {jax.devices()}")
    
    # 1. Initialize fixed tuned genome for controlled comparison
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    n_patches = 2
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.14, maxval=0.19)
    sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.012, maxval=0.018)
    
    v_scale = 5.2
    alpha_diff = 0.06
    
    # Condition A: Static Infinite-Nutrient Baseline (depletion_rate = 0.0)
    params_static = FlowLeniaParams(
        mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K),
        v_scale=v_scale, alpha_diffusion=alpha_diff,
        depletion_rate=0.0, regen_rate=0.0
    )
    
    # Condition B: Dynamic Resource Depletion (depletion = 0.04, regen = 0.01)
    params_depletion = FlowLeniaParams(
        mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K),
        v_scale=v_scale, alpha_diffusion=alpha_diff,
        depletion_rate=0.04, regen_rate=0.01
    )
    
    # Common Initial State
    rng_key, subk_init = random.split(rng_key)
    init_state = initialize_multi_patch_state(
        subk_init, H, W, C=1, K=K, n_patches=n_patches, kernel_radii=radii,
        mu_presets=mu_presets, sigma_presets=sigma_presets
    )
    
    conditions = [
        ("static_baseline", False, params_static),
        ("dynamic_depletion", True, params_depletion)
    ]
    
    results = {}
    
    for cond_name, enable_dep, p in conditions:
        print(f"\n--- Running Condition: {cond_name} ---")
        rng_key, subk_rollout = random.split(rng_key)
        final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
            init_state, kernel_ffts, p, subk_rollout,
            num_steps=steps,
            sample_interval=sample_interval,
            enable_depletion=enable_dep,
            enable_mutation=False
        )
        
        sampled_mass_np = np.array(sampled_mass)
        sampled_gid_np = np.array(sampled_gid)
        final_mass_arr = sampled_mass_np[-1, 0] if sampled_mass_np.ndim == 4 else sampled_mass_np[-1]
        metrics = evaluate_run_metrics(final_mass_arr, sampled_mass_np, sampled_gid_np, steps)
        
        # Calculate spatial trajectory radius (foraging coverage)
        com_coords = []
        S = sampled_mass_np.shape[0]
        for f in range(S):
            mf = sampled_mass_np[f, 0] if sampled_mass_np.ndim == 4 else sampled_mass_np[f]
            cy, cx = compute_center_of_mass(mf)
            com_coords.append([cy, cx])
        com_arr = np.array(com_coords)
        com_variance = float(np.var(com_arr[:, 0]) + np.var(com_arr[:, 1]))
        
        print(f"Motility (CoM Shift): {metrics['com_displacement']:.2f} px")
        print(f"Evolutionary Activity (EA): {metrics['ea_raw']:.6f}")
        print(f"Compression Complexity: {metrics['complexity_raw']:.0f} bytes")
        print(f"Foraging Spatial Variance: {com_variance:.2f} px^2")
        
        config_dict = {
            "condition": cond_name,
            "enable_depletion": enable_dep,
            "depletion_rate": float(p.depletion_rate),
            "regen_rate": float(p.regen_rate),
            "grid_size": grid_size,
            "steps": steps,
            "v_scale": v_scale
        }
        
        metrics["foraging_variance"] = com_variance
        
        art_paths = save_experiment_artifacts(
            sampled_mass_frames=sampled_mass_np,
            metrics=metrics,
            config=config_dict,
            output_dir=output_dir,
            prefix=cond_name,
            fps=20
        )
        results[cond_name] = {
            "metrics": metrics,
            "artifacts": art_paths
        }
        
    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)
    conds = ["Static Baseline", "Dynamic Depletion"]
    colors = ["#2b5c8f", "#e05a47"]
    
    # 1. Motility (CoM)
    coms = [results["static_baseline"]["metrics"]["com_displacement"], results["dynamic_depletion"]["metrics"]["com_displacement"]]
    axes[0].bar(conds, coms, color=colors, width=0.5)
    axes[0].set_title("Motility (CoM Displacement)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Pixels (px)", fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 2. Evolutionary Activity
    eas = [results["static_baseline"]["metrics"]["ea_raw"], results["dynamic_depletion"]["metrics"]["ea_raw"]]
    axes[1].bar(conds, eas, color=colors, width=0.5)
    axes[1].set_title("Evolutionary Activity (EA)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel(r"Mean $(\Delta A)^2$", fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 3. Foraging Variance (Coverage)
    vars_ = [results["static_baseline"]["metrics"]["foraging_variance"], results["dynamic_depletion"]["metrics"]["foraging_variance"]]
    axes[2].bar(conds, vars_, color=colors, width=0.5)
    axes[2].set_title(r"Foraging Spatial Coverage ($\sigma_{\text{CoM}}^2$)", fontsize=11, fontweight='bold')
    axes[2].set_ylabel(r"$\text{px}^2$", fontsize=10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.suptitle("Impact of Dynamic Niche Depletion on Flow-Lenia Soliton Locomotion", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    comp_plot_path = os.path.join(output_dir, "depletion_comparison_metrics.png")
    plt.savefig(comp_plot_path)
    plt.close()
    
    summary_path = os.path.join(output_dir, "depletion_experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else str(x))
        
    print(f"\n=== Depletion Experiment Complete ===")
    print(f"Comparison Plot: {comp_plot_path}")
    print(f"Summary JSON: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Resource Depletion Thesis Experiment")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid resolution")
    parser.add_argument("--steps", type=int, default=3600, help="Simulation steps (default: 3600 -> 1 min video)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Sampling interval (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/resource_depletion", help="Output directory")
    args = parser.parse_args()
    
    run_depletion_experiment(
        grid_size=args.grid_size,
        steps=args.steps,
        sample_interval=args.sample_interval,
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
