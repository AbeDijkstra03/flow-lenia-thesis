import argparse
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from core.imgep import run_imgep_experiment, run_random_search_experiment
from core.flow_lenia_jax import (
    FlowLeniaParams, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.environment import create_homogeneous_mask, create_wall_obstacle_mask
from core.visualization import save_experiment_artifacts

def run_single_imgep_seed(
    seed: int,
    env_type: str = "open",
    trials: int = 40,
    bootstrap: int = 10,
    grid_size: int = 256,
    search_steps: int = 2000,
    search_sample_interval: int = 50,
    elite_steps: int = 3600,
    elite_sample_interval: int = 3,
    base_output_dir: str = "results/baseline_imgep"
):
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    rng_key = random.PRNGKey(seed)
    
    if env_type == "wall":
        wall_mask = create_wall_obstacle_mask(H=grid_size, W=grid_size)
    else:
        wall_mask = None
        
    print(f"\n=======================================================")
    print(f"=== IMGEP Search Seed {seed} (Env: {env_type.upper()}) ===")
    print(f"=======================================================")
    
    # 1. Run IMGEP Exploration
    rng_key, subk = random.split(rng_key)
    imgep_archive, _ = run_imgep_experiment(
        subk, n_trials=trials, n_bootstrap=bootstrap,
        grid_size=grid_size, num_steps=search_steps,
        sample_interval=search_sample_interval, wall_mask=wall_mask
    )
    
    # 2. Run Random Search Baseline
    rng_key, subk = random.split(rng_key)
    random_archive, _ = run_random_search_experiment(
        subk, n_trials=trials, grid_size=grid_size,
        num_steps=search_steps, sample_interval=search_sample_interval, wall_mask=wall_mask
    )
    
    imgep_metrics = imgep_archive.get_metrics_array()
    random_metrics = random_archive.get_metrics_array()
    
    summary = {
        "seed": seed,
        "env": env_type,
        "trials": trials,
        "grid_size": grid_size,
        "imgep_com_mean": float(np.mean(imgep_metrics[:, 0])),
        "imgep_com_std": float(np.std(imgep_metrics[:, 0])),
        "random_com_mean": float(np.mean(random_metrics[:, 0])),
        "random_com_std": float(np.std(random_metrics[:, 0])),
        "imgep_ea_mean": float(np.mean(imgep_metrics[:, 1])),
        "imgep_ea_std": float(np.std(imgep_metrics[:, 1])),
        "random_ea_mean": float(np.mean(random_metrics[:, 1])),
        "random_ea_std": float(np.std(random_metrics[:, 1])),
        "imgep_comp_mean": float(np.mean(imgep_metrics[:, 2])),
        "imgep_comp_std": float(np.std(imgep_metrics[:, 2])),
        "random_comp_mean": float(np.mean(random_metrics[:, 2])),
        "random_comp_std": float(np.std(random_metrics[:, 2]))
    }
    
    with open(os.path.join(seed_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    np.save(os.path.join(seed_dir, "imgep_metrics.npy"), imgep_metrics)
    np.save(os.path.join(seed_dir, "random_metrics.npy"), random_metrics)
    
    # 3. Select Top 3 IMGEP Elites and re-render full 1-minute (1200 frames) HD videos!
    norm_mat = imgep_archive.get_normalized_metrics_array()
    rank_scores = norm_mat[:, 0] + norm_mat[:, 1]
    top_indices = np.argsort(rank_scores)[::-1][:3]
    
    print(f"\n[Visualization] Rerunning top 3 elites for full 1-minute video rollouts (3600 steps)...")
    for r_rank, top_idx in enumerate(top_indices):
        trial = imgep_archive.trials[top_idx]
        cfg = trial["config"]
        prefix = f"elite_{r_rank+1}"
        
        # High-res 1-minute rollout (3600 steps @ sample_interval=3 -> 1200 frames @ 20 FPS = 60.0s)
        radii = jnp.array(cfg["radii"], dtype=jnp.float32)
        K = len(radii)
        H, W = grid_size, grid_size
        kernel_ffts = precompute_kernel_ffts(radii, H, W)
        
        mu_presets = jnp.array(cfg["mu_presets"], dtype=jnp.float32)
        sigma_presets = jnp.array(cfg["sigma_presets"], dtype=jnp.float32)
        
        rng_key, subk_init = random.split(rng_key)
        state = initialize_multi_patch_state(
            subk_init, H, W, C=1, K=K,
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
        
        rng_key, subk_roll = random.split(rng_key)
        final_st, sampled_mass, sampled_gid = run_flow_lenia_rollout(
            state, kernel_ffts, params, subk_roll,
            num_steps=elite_steps,
            sample_interval=elite_sample_interval,
            wall_mask=wall_mask,
            mixing_rule='gene_wise',
            enable_mutation=True,
            mutation_interval=60
        )
        
        save_experiment_artifacts(
            sampled_mass_frames=np.array(sampled_mass),
            metrics=trial["metrics"],
            config=cfg,
            output_dir=seed_dir,
            prefix=prefix,
            fps=20,
            wall_mask=wall_mask
        )
        print(f"Saved 1-minute rollout for: {prefix} in {seed_dir}/{prefix}/")
        
    return summary

def main():
    parser = argparse.ArgumentParser(description="Flow-Lenia IMGEP Search Multi-Seed Scientific Runner")
    parser.add_argument("--trials", type=int, default=40, help="Total trial budget per search method")
    parser.add_argument("--bootstrap", type=int, default=10, help="IMGEP bootstrap phase random trials")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size resolution")
    parser.add_argument("--steps", type=int, default=2000, help="Search exploration horizon steps")
    parser.add_argument("--sample_interval", type=int, default=50, help="Search frame sampling interval")
    parser.add_argument("--elite_steps", type=int, default=3600, help="Elite rollout horizon (3600 = 1 min)")
    parser.add_argument("--elite_sample_interval", type=int, default=3, help="Elite sampling interval")
    parser.add_argument("--env", type=str, choices=["open", "wall"], default="open", help="Environment type")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/baseline_imgep", help="Directory to store results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    multiseed_summaries = []
    for s in args.seeds:
        summary = run_single_imgep_seed(
            seed=s,
            env_type=args.env,
            trials=args.trials,
            bootstrap=args.bootstrap,
            grid_size=args.grid_size,
            search_steps=args.steps,
            search_sample_interval=args.sample_interval,
            elite_steps=args.elite_steps,
            elite_sample_interval=args.elite_sample_interval,
            base_output_dir=args.output_dir
        )
        multiseed_summaries.append(summary)
        
    with open(os.path.join(args.output_dir, "multiseed_summary.json"), "w") as f:
        json.dump(multiseed_summaries, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"All {len(args.seeds)} seeds completed for {args.output_dir}!")
    print(f"Aggregated summary written to: {args.output_dir}/multiseed_summary.json")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
