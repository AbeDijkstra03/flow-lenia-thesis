import argparse
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from core.imgep import (
    IMGEPArchive, sample_random_config, evaluate_single_config_jax
)
from core.flow_lenia_jax import (
    FlowLeniaParams, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.metrics import evaluate_run_metrics
from core.environment import create_wall_obstacle_mask
from core.visualization import save_experiment_artifacts

def main():
    parser = argparse.ArgumentParser(description="Scaled-Up Reruns on 512x512+ Grids with Proportional Patch Tiling")
    parser.add_argument("--k_reruns", type=int, default=3, help="Number of FPS selected candidates to scale up")
    parser.add_argument("--search_trials", type=int, default=30, help="Initial search trial count if archive missing")
    parser.add_argument("--scale_grid_size", type=int, default=512, help="Scaled-up grid size resolution")
    parser.add_argument("--scale_steps", type=int, default=3600, help="Scaled-up simulation steps horizon (default: 3600 -> 1 min)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Frame sampling interval (default: 3)")
    parser.add_argument("--env", type=str, choices=["open", "wall"], default="open", help="Environment type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/scaleup", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    rng_key = random.PRNGKey(args.seed)
    
    print(f"=== Flow-Lenia Scaled-Up Reruns (Proportional Tiling) ===")
    print(f"Grid Resolution: {args.scale_grid_size}x{args.scale_grid_size} | Horizon: {args.scale_steps} steps")
    print(f"FPS Candidates: {args.k_reruns} | JAX Device: {jax.devices()}")
    
    if args.env == "wall":
        wall_mask = create_wall_obstacle_mask(H=args.scale_grid_size, W=args.scale_grid_size)
    else:
        wall_mask = None
        
    archive = IMGEPArchive()
    print("[Scale-up] Generating search archive candidates...")
    for t in range(args.search_trials):
        rng_key, cfg = sample_random_config(rng_key)
        rng_key, metrics, _ = evaluate_single_config_jax(
            rng_key, cfg, grid_size=256, num_steps=2000, sample_interval=args.sample_interval, wall_mask=None
        )
        archive.add_trial(cfg, metrics)
        
    fps_indices = archive.select_farthest_point_sampling(args.k_reruns)
    print(f"[Scale-up] Selected FPS Archive Indices: {fps_indices}")
    
    rerun_results = []
    
    for idx, archive_idx in enumerate(fps_indices):
        trial = archive.trials[archive_idx]
        cfg = trial["config"]
        print(f"\n--- Rerun {idx+1}/{args.k_reruns} (Archive Trial #{archive_idx}) ---")
        print(f"Patch Count: {cfg['n_patches']} | Kernel Radii: {cfg['radii']}")
        
        radii = jnp.array(cfg["radii"], dtype=jnp.float32)
        K = len(radii)
        H, W = args.scale_grid_size, args.scale_grid_size
        
        kernel_ffts = precompute_kernel_ffts(radii, H, W)
        
        mu_presets = jnp.array(cfg["mu_presets"], dtype=jnp.float32)
        sigma_presets = jnp.array(cfg["sigma_presets"], dtype=jnp.float32)
        
        rng_key, subk = random.split(rng_key)
        state = initialize_multi_patch_state(
            subk, H, W, C=1, K=K,
            n_patches=cfg["n_patches"],
            kernel_radii=radii,
            mu_presets=mu_presets,
            sigma_presets=sigma_presets,
            wall_mask=wall_mask
        )
        
        params = FlowLeniaParams(
            mu=mu_presets[0],
            sigma=sigma_presets[0],
            weights=jnp.full((K,), 1.0 / K),
            v_scale=float(cfg.get("v_scale", 5.4)),
            alpha_diffusion=float(cfg.get("alpha_diffusion", 0.055))
        )
        
        rng_key, subk = random.split(rng_key)
        final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
            state, kernel_ffts, params, subk,
            num_steps=args.scale_steps,
            sample_interval=args.sample_interval,
            wall_mask=wall_mask,
            mixing_rule='gene_wise',
            enable_mutation=True,
            mutation_interval=60
        )
        
        final_mass_np = np.array(final_state.mass)
        sampled_mass_np = np.array(sampled_mass)
        sampled_gid_np = np.array(sampled_gid)
        
        metrics = evaluate_run_metrics(
            final_mass_np, sampled_mass_np, sampled_gid_np,
            total_steps=args.scale_steps, n_genomes=cfg["n_patches"]
        )
        
        print(f"Raw EA: {metrics['ea_raw']:.6f} | Normalized EA: {metrics['ea_norm']:.6f}")
        print(f"Raw Complexity: {metrics['complexity_raw']:.0f} bytes | Normalized Comp: {metrics['complexity_norm']:.4f}")
        print(f"Raw Entropy: {metrics['entropy_raw']:.4f} bits | Normalized Entropy: {metrics['entropy_norm']:.4f}")
        
        prefix = f"rerun_{idx+1}"
        art_paths = save_experiment_artifacts(
            sampled_mass_frames=sampled_mass_np,
            metrics=metrics,
            config=cfg,
            output_dir=args.output_dir,
            prefix=prefix,
            fps=20,
            wall_mask=wall_mask
        )
        print(f"Saved artifacts to: {art_paths['video']}")
        
        rerun_results.append({
            "rerun_id": idx + 1,
            "archive_index": archive_idx,
            "config": cfg,
            "metrics": metrics,
            "artifacts": art_paths
        })
        
    report_path = os.path.join(args.output_dir, "scaleup_report.json")
    with open(report_path, "w") as f:
        json.dump(rerun_results, f, indent=2)
        
    print(f"\nScaled-up reruns complete! Report written to: {report_path}")

if __name__ == "__main__":
    main()
