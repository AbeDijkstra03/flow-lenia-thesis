import argparse
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import List, Dict, Any

from core.imgep import IMGEPArchive, sample_random_config, evaluate_single_config_jax
from core.flow_lenia_jax import (
    FlowLeniaParams, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.metrics import evaluate_run_metrics
from core.environment import create_wall_obstacle_mask
from core.visualization import save_experiment_artifacts

def run_single_scaleup_seed(
    seed: int,
    k_reruns: int = 3,
    scale_grid_size: int = 512,
    scale_steps: int = 3600,
    sample_interval: int = 3,
    env_type: str = "open",
    base_output_dir: str = "results/scaleup"
) -> Dict[str, Any]:
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    rng_key = random.PRNGKey(seed)
    
    print(f"\n=======================================================")
    print(f"=== Flow-Lenia Scaled-Up 512x512 Reruns (Seed {seed}) ===")
    print(f"=======================================================")
    
    if env_type == "wall":
        wall_mask = create_wall_obstacle_mask(H=scale_grid_size, W=scale_grid_size)
    else:
        wall_mask = None
        
    # Prioritize loading discovered elite champion genomes from agentic loop archive
    state_file = "results/agentic_loop/agentic_loop_state.json"
    selected_configs = []
    
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
            elites = state_data.get("elites", [])
            if len(elites) > 0:
                sorted_elites = sorted(elites, key=lambda x: x.get("watertight_score", 0), reverse=True)
                # Take diverse elites based on seed offset
                seed_offset = {42: 0, 101: 3, 2024: 6}.get(seed, 0)
                selected_configs = [e["config"] for e in sorted_elites[seed_offset:seed_offset+k_reruns]]
                if len(selected_configs) < k_reruns:
                    selected_configs = [e["config"] for e in sorted_elites[:k_reruns]]
                print(f"[Scale-up] Loaded {len(selected_configs)} elite champion genomes from {state_file}")
        except Exception as e:
            print(f"[Scale-up] Warning reading {state_file}: {e}")
            
    if len(selected_configs) == 0:
        archive = IMGEPArchive()
        print("[Scale-up] Generating search archive candidates...")
        for t in range(25):
            rng_key, cfg = sample_random_config(rng_key)
            rng_key, metrics, _ = evaluate_single_config_jax(
                rng_key, cfg, grid_size=256, num_steps=2000, sample_interval=sample_interval, wall_mask=None
            )
            archive.add_trial(cfg, metrics)
            
        fps_indices = archive.select_farthest_point_sampling(k_reruns)
        selected_configs = [archive.trials[i]["config"] for i in fps_indices]
    
    rerun_results = []
    
    for idx, cfg in enumerate(selected_configs):
        print(f"\n--- [Seed {seed}] Rerun {idx+1}/{len(selected_configs)} ---")
        
        # On 512x512, tile 6 to 8 interactive patches so the canvas is full of life!
        n_patches = max(6, int(cfg.get("n_patches", 4) * 2))
        radii = jnp.array(cfg["radii"], dtype=jnp.float32)
        K = len(radii)
        H, W = scale_grid_size, scale_grid_size
        
        print(f"Canvas: {H}x{W} | Interactive Species Patches: {n_patches} | Horizon: {scale_steps} steps (1 min HD)")
        kernel_ffts = precompute_kernel_ffts(radii, H, W)
        
        rng_key, k_mu, k_sigma = random.split(rng_key, 3)
        mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.140, maxval=0.168)
        sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.012, maxval=0.017)
        
        rng_key, subk = random.split(rng_key)
        state = initialize_multi_patch_state(
            subk, H, W, C=1, K=K,
            n_patches=n_patches,
            kernel_radii=radii,
            mu_presets=mu_presets,
            sigma_presets=sigma_presets,
            wall_mask=wall_mask
        )
        
        v_scale = float(cfg.get("v_scale", 5.8))
        if v_scale < 5.4: v_scale = 5.8
        
        params = FlowLeniaParams(
            mu=mu_presets[0],
            sigma=sigma_presets[0],
            weights=jnp.full((K,), 1.0 / K),
            v_scale=v_scale,
            alpha_diffusion=0.050
        )
        
        rng_key, subk = random.split(rng_key)
        final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
            state, kernel_ffts, params, subk,
            num_steps=scale_steps,
            sample_interval=sample_interval,
            wall_mask=wall_mask,
            mixing_rule='gene_wise',
            enable_mutation=True,
            mutation_interval=50
        )
        
        sampled_mass_np = np.array(sampled_mass)
        final_mass_np = sampled_mass_np[-1, 0] if sampled_mass_np.ndim == 4 else sampled_mass_np[-1]
        sampled_gid_np = np.array(sampled_gid)
        
        metrics = evaluate_run_metrics(
            final_mass_np, sampled_mass_np, sampled_gid_np,
            total_steps=scale_steps, n_genomes=n_patches
        )
        
        print(f"Motility (CoM): {metrics['com_displacement']:.2f} px | EA: {metrics['ea_raw']:.6f} | Complexity: {metrics['complexity_raw']:.0f} bytes")
        
        prefix = f"rerun_{idx+1}"
        art_paths = save_experiment_artifacts(
            sampled_mass_frames=sampled_mass_np,
            metrics=metrics,
            config=cfg,
            output_dir=seed_dir,
            prefix=prefix,
            fps=20,
            wall_mask=wall_mask
        )
        print(f"Saved 1-minute 512x512 HD video to: {art_paths['video']}")
        
        rerun_results.append({
            "rerun_id": idx + 1,
            "seed": seed,
            "n_patches": n_patches,
            "metrics": metrics,
            "artifacts": art_paths
        })
        
    report_path = os.path.join(seed_dir, "scaleup_report.json")
    with open(report_path, "w") as f:
        json.dump(rerun_results, f, indent=2)
        
    return {"seed": seed, "reruns": rerun_results}

def main():
    parser = argparse.ArgumentParser(description="Scaled-Up Multi-Seed Reruns on 512x512+ Grids with Proportional Patch Tiling")
    parser.add_argument("--k_reruns", type=int, default=3, help="Number of candidate reruns per seed")
    parser.add_argument("--scale_grid_size", type=int, default=512, help="Scaled-up grid size resolution")
    parser.add_argument("--scale_steps", type=int, default=3600, help="Scaled-up simulation steps (3600 = 1 min HD)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Frame sampling interval")
    parser.add_argument("--env", type=str, choices=["open", "wall"], default="open", help="Environment type")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/scaleup", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    multiseed_summaries = []
    for s in args.seeds:
        summary = run_single_scaleup_seed(
            seed=s,
            k_reruns=args.k_reruns,
            scale_grid_size=args.scale_grid_size,
            scale_steps=args.scale_steps,
            sample_interval=args.sample_interval,
            env_type=args.env,
            base_output_dir=args.output_dir
        )
        multiseed_summaries.append(summary)
        
    with open(os.path.join(args.output_dir, "multiseed_summary.json"), "w") as f:
        json.dump(multiseed_summaries, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"Scale-up complete across all {len(args.seeds)} seeds!")
    print(f"Aggregated summary written to: {args.output_dir}/multiseed_summary.json")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
