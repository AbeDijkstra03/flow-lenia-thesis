import argparse
import os
import json
import numpy as np
import jax
from jax import random

from core.imgep import run_imgep_experiment, run_random_search_experiment
from core.environment import create_homogeneous_mask, create_wall_obstacle_mask
from core.visualization import save_experiment_artifacts, save_rollout_mp4, extract_trajectory_filmstrip

def main():
    parser = argparse.ArgumentParser(description="Flow-Lenia IMGEP Search vs Random Search Experiment")
    parser.add_argument("--trials", type=int, default=50, help="Total trial budget per search method")
    parser.add_argument("--bootstrap", type=int, default=10, help="IMGEP bootstrap phase random trials")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size resolution")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation horizon steps (e.g. 2000)")
    parser.add_argument("--sample_interval", type=int, default=50, help="Frame sampling interval")
    parser.add_argument("--env", type=str, choices=["open", "wall"], default="open", help="Environment type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/imgep_search", help="Directory to store results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    rng_key = random.PRNGKey(args.seed)
    
    if args.env == "wall":
        wall_mask = create_wall_obstacle_mask(H=args.grid_size, W=args.grid_size)
    else:
        wall_mask = None
        
    print(f"=== Flow-Lenia SOTA Search Experiment ===")
    print(f"Environment: {args.env.upper()} | Grid Size: {args.grid_size}x{args.grid_size} | Horizon: {args.steps} steps")
    print(f"Trials: {args.trials} | JAX Device: {jax.devices()}")
    
    # 1. Run IMGEP Exploration
    rng_key, subk = random.split(rng_key)
    imgep_archive, imgep_rollouts = run_imgep_experiment(
        subk, n_trials=args.trials, n_bootstrap=args.bootstrap,
        grid_size=args.grid_size, num_steps=args.steps,
        sample_interval=args.sample_interval, wall_mask=wall_mask
    )
    
    # 2. Run Random Search Baseline
    rng_key, subk = random.split(rng_key)
    random_archive, random_rollouts = run_random_search_experiment(
        subk, n_trials=args.trials, grid_size=args.grid_size,
        num_steps=args.steps, sample_interval=args.sample_interval, wall_mask=wall_mask
    )
    
    imgep_metrics = imgep_archive.get_metrics_array()
    random_metrics = random_archive.get_metrics_array()
    
    summary = {
        "trials": args.trials,
        "grid_size": args.grid_size,
        "steps": args.steps,
        "seed": args.seed,
        "env": args.env,
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
    
    print("\n=== SEARCH SUMMARY RESULTS ===")
    print(f"IMGEP  -> Mean CoM: {summary['imgep_com_mean']:.2f} px | Mean EA: {summary['imgep_ea_mean']:.6f} | Mean Complexity: {summary['imgep_comp_mean']:.1f} bytes")
    print(f"RANDOM -> Mean CoM: {summary['random_com_mean']:.2f} px | Mean EA: {summary['random_ea_mean']:.6f} | Mean Complexity: {summary['random_comp_mean']:.1f} bytes")
    
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    np.save(os.path.join(args.output_dir, "imgep_metrics.npy"), imgep_metrics)
    np.save(os.path.join(args.output_dir, "random_metrics.npy"), random_metrics)
    
    # Export artifacts (MP4 video, trajectory filmstrip, motion heatmap, NPZ, JSON) for top 3 IMGEP elites
    norm_mat = imgep_archive.get_normalized_metrics_array()
    rank_scores = norm_mat[:, 0] + norm_mat[:, 1]
    top_indices = np.argsort(rank_scores)[::-1][:3]
    
    print(f"\n[Visualization] Saving SOTA MP4 videos, filmstrips, & heatmaps for top 3 IMGEP elites...")
    for r_rank, top_idx in enumerate(top_indices):
        trial = imgep_archive.trials[top_idx]
        prefix = f"elite_{r_rank+1}_trial{top_idx}"
        save_experiment_artifacts(
            sampled_mass_frames=imgep_rollouts[top_idx],
            metrics=trial["metrics"],
            config=trial["config"],
            output_dir=args.output_dir,
            prefix=prefix,
            fps=20,
            wall_mask=wall_mask
        )
        print(f"Saved artifacts for: {prefix} (MP4 + Filmstrip + Heatmap + NPZ)")
        
    print(f"\nAll search experiment artifacts successfully saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
