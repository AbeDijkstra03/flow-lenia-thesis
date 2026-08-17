#!/usr/bin/env python3
"""
Thesis Experiment 3B: Chemotaxis Baseline Calibration & The Cohesion-Fission Phase Transition.

Evaluates how internal surface tension (alpha_diffusion, sigma) and chemotactic gradient pull (chi)
govern the morphological phase transition between:
1. Unbaited Control (chi = 0.0): Stationary baseline soliton.
2. Cohesive Foraging (chi = 18.0, alpha = 0.065, sigma = 0.013): Unitary elastomeric droplet migration.
3. Proliferative Fission (chi = 25.0, alpha = 0.035, sigma = 0.015): Amoeboid shear-induced cell division / mitosis.
"""

import os
import json
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts,
    compute_sobel_gradients, moroz_reintegration_tracking
)
from core.metrics import evaluate_run_metrics, compute_center_of_mass
from core.visualization import save_experiment_artifacts

def run_single_calibration_seed(
    seed: int,
    grid_size: int = 256,
    steps: int = 3600,
    sample_interval: int = 3,
    base_output_dir: str = "results/chemotaxis_calibration"
) -> Dict[str, Any]:
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    rng_key = random.PRNGKey(seed)
    
    H, W = grid_size, grid_size
    K = 9
    radii = jnp.linspace(6.0, 15.0, K)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    # Setup Food Reservoir Field: 0.0 at x=40, ramping smoothly to 1.0 at x=200
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    resource_map_np = np.clip((xx - 40.0) / 160.0, 0.0, 1.0).astype(np.float32)
    resource_jnp = jnp.array(resource_map_np, dtype=jnp.float32)
    rx, ry = compute_sobel_gradients(resource_jnp)
    
    print(f"\n==================================================================")
    print(f"=== Chemotaxis Calibration & Cohesion-Fission Sweep (Seed {seed}) ===")
    print(f"Evaluating 3 Morphological Regimes Across Horizon = {steps} (1 min HD)")
    print(f"==================================================================")
    
    conditions = [
        ("unbaited_control", 0.0, 0.065, 0.013, 8.5, "Stationary Baseline"),
        ("cohesive_foraging", 18.0, 0.065, 0.013, 8.5, "Unitary Droplet Migration"),
        ("dividing_fission", 25.0, 0.035, 0.015, 9.5, "Amoeboid Fission / Mitosis")
    ]
    
    condition_results = {}
    
    for cond_name, chi_val, alpha_val, sigma_val, v_s, desc in conditions:
        print(f"\n--- [Seed {seed}] Condition: {cond_name} ({desc}) ---")
        print(f"Parameters: chi = {chi_val:.1f} | alpha = {alpha_val:.3f} | sigma = {sigma_val:.3f} | v_scale = {v_s:.1f}")
        
        # Spawn standard circular soliton at (y=128, x=50)
        cy, cx = 128, 50
        r_dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        blob = np.exp(-r_dist**2 / (2.0 * 14.0**2))
        init_mass_2d = np.where(r_dist < 22.0, np.clip(blob, 0.0, 1.0), 0.0)
        
        mu_presets = jnp.full((1, K), 0.150)
        sigma_presets = jnp.full((1, K), sigma_val)
        weights_preset = jnp.full((K,), 1.0 / K)
        
        init_state = FlowLeniaState(
            jnp.array(init_mass_2d[None, :, :], dtype=jnp.float32),
            jnp.full((K, H, W), 0.150, dtype=jnp.float32),
            jnp.full((K, H, W), sigma_val, dtype=jnp.float32),
            jnp.full((K, H, W), 1.0 / K, dtype=jnp.float32),
            resource_jnp,
            jnp.zeros((H, W), dtype=jnp.int32)
        )
        
        def _step_fn(curr_st):
            mass_p = curr_st.mass[0]
            fft_m = jnp.fft.rfft2(mass_p)
            U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
            G_k = 2.0 * jnp.exp(-((U_stack - curr_st.mu_map)**2) / (2.0 * jnp.square(curr_st.sigma_map) + 1e-8)) - 1.0
            G = jnp.sum(curr_st.weights_map * G_k, axis=0)
            gx, gy = compute_sobel_gradients(G)
            ax, ay = compute_sobel_gradients(mass_p)
            vx = v_s * ((1.0 - alpha_val) * gx - alpha_val * ax) + chi_val * rx
            vy = v_s * ((1.0 - alpha_val) * gy - alpha_val * ay) + chi_val * ry
            vx = jnp.tanh(vx)
            vy = jnp.tanh(vy)
            new_mass_p, _, _, _, _, _ = moroz_reintegration_tracking(mass_p, vx, vy)
            return FlowLeniaState(
                new_mass_p[None, :, :],
                curr_st.mu_map,
                curr_st.sigma_map,
                curr_st.weights_map,
                curr_st.resource_map,
                curr_st.genome_id_map
            )
            
        def _scan_fn(c, step_idx):
            nxt = _step_fn(c)
            return nxt, nxt.mass[0]
            
        final_state, all_mass_frames = jax.lax.scan(_scan_fn, init_state, jnp.arange(steps))
        
        sampled_indices = np.arange(0, steps, sample_interval)
        sampled_mass_np = np.array(all_mass_frames)[sampled_indices]
        sampled_mass_np = sampled_mass_np[:, None, :, :]
        final_mass_arr = sampled_mass_np[-1, 0]
        sampled_gid_np = np.zeros((sampled_mass_np.shape[0], H, W), dtype=np.int32)
        
        metrics = evaluate_run_metrics(final_mass_arr, sampled_mass_np, sampled_gid_np, steps)
        
        cy_init, cx_init = compute_center_of_mass(init_mass_2d)
        cy_final, cx_final = compute_center_of_mass(final_mass_arr)
        com_dx = float(cx_final - cx_init)
        
        print(f"Horizontal CoM Shift (dx) : {com_dx:+.2f} px (Start X={cx_init:.1f} -> End X={cx_final:.1f})")
        print(f"Solid Core Preservation   : {metrics['solid_core_ratio'] * 100:.1f}%")
        print(f"Complexity (gzip)         : {metrics['complexity_raw']} bytes")
        
        config_dict = {
            "condition": cond_name,
            "description": desc,
            "chemotaxis_chi": chi_val,
            "alpha_diffusion": alpha_val,
            "sigma": sigma_val,
            "v_scale": v_s,
            "seed": seed,
            "com_dx": com_dx,
            "steps": steps
        }
        
        art_paths = save_experiment_artifacts(
            sampled_mass_frames=sampled_mass_np,
            metrics=metrics,
            config=config_dict,
            output_dir=seed_dir,
            prefix=cond_name,
            fps=20
        )
        
        condition_results[cond_name] = {
            "condition": cond_name,
            "description": desc,
            "chemotaxis_chi": chi_val,
            "alpha_diffusion": alpha_val,
            "sigma": sigma_val,
            "v_scale": v_s,
            "com_dx": com_dx,
            "start_x": float(cx_init),
            "end_x": float(cx_final),
            "metrics": metrics,
            "artifacts": art_paths
        }
        
    # Generate 3-way morphological comparison chart
    cond_labels = [
        "Unbaited Control\n($\chi=0.0$)",
        "Cohesive Foraging\n(Unitary Droplet)",
        "Dividing Fission\n(Amoeboid Mitosis)"
    ]
    dx_values = [
        condition_results["unbaited_control"]["com_dx"],
        condition_results["cohesive_foraging"]["com_dx"],
        condition_results["dividing_fission"]["com_dx"]
    ]
    colors = ['#757575', '#185ADB', '#FF5722']
    
    plt.figure(figsize=(9, 5), dpi=300)
    bars = plt.bar(cond_labels, dx_values, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    plt.title(f"Chemotactic Morphological Regimes (Seed {seed})", fontsize=13, fontweight='bold')
    plt.ylabel(r"Horizontal Displacement $\Delta x$ (pixels)", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (2.0 if yval >= 0 else -6.0), f"{yval:+.1f} px", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(seed_dir, "chemotaxis_calibration_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    
    summary_data = {
        "experiment_name": "Chemotaxis Baseline Calibration & Cohesion-Fission Transition",
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "seed": seed,
        "results": condition_results
    }
    with open(os.path.join(seed_dir, "calibration_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
        
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="Chemotaxis Calibration & Fission Multi-Seed Runner")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size")
    parser.add_argument("--steps", type=int, default=3600, help="Simulation steps (3600 = 1 min HD)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Sampling interval")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/chemotaxis_calibration", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    multiseed_summaries = []
    for s in args.seeds:
        summary = run_single_calibration_seed(
            seed=s,
            grid_size=args.grid_size,
            steps=args.steps,
            sample_interval=args.sample_interval,
            base_output_dir=args.output_dir
        )
        multiseed_summaries.append(summary)
        
    with open(os.path.join(args.output_dir, "multiseed_summary.json"), "w") as f:
        json.dump(multiseed_summaries, f, indent=2)
        
    print(f"\n==================================================================")
    print(f"Chemotaxis calibration & fission sweep complete for seeds {args.seeds}!")
    print(f"Aggregated summary written to: {args.output_dir}/multiseed_summary.json")
    print(f"==================================================================")

if __name__ == "__main__":
    main()
