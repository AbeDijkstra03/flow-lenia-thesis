#!/usr/bin/env python3
"""
Autonomous Agentic Discovery Loop Harness for Flow-Lenia.

Orchestrates multi-generation IMGEP goal exploration, watertight quality filtering,
dual-panel visual frame extraction for AI agent multimodal vision inspection,
broadcast MP4 video generation, and persistent JSON state logging.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp

from core.flow_lenia_jax import (
    FlowLeniaParams, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.imgep import run_imgep_experiment
from core.metrics import evaluate_watertight_quality_score
from core.visualization import save_rollout_mp4, extract_trajectory_filmstrip, save_motion_heatmap, colorize_frame_plasma_log, render_physical_frame

def save_dual_panel_frames_and_video(frames: np.ndarray, output_prefix: str, fps: int = 20) -> List[str]:
    """
    Extract dual-panel trajectory frames (Plasma palette + Absolute physical scale)
    for AI agent visual inspection, plus high-definition MP4 video and composite filmstrip.
    """
    if frames.ndim == 4:
        frames = frames[:, 0, :, :]
        
    S, H, W = frames.shape
    frame_dir = os.path.dirname(output_prefix)
    os.makedirs(frame_dir, exist_ok=True)
    
    # 1. Save broadcast MP4 video
    mp4_path = f"{output_prefix}_rollout.mp4"
    save_rollout_mp4(frames, mp4_path, fps=fps, dual_panel=True)
    
    # 2. Save 6 key trajectory frames (0%, 20%, 40%, 60%, 80%, 100%) for AI vision models
    pcts = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]
    frame_paths = []
    
    for idx, p in enumerate(pcts):
        f_idx = min(int(p * (S - 1)), S - 1)
        f_path = f"{output_prefix}_frame_step_{idx+1}_pct{int(p*100)}.png"
        
        left_img = colorize_frame_plasma_log(frames[f_idx])
        right_img = render_physical_frame(frames[f_idx])
        
        dual_img = np.concatenate([left_img, right_img], axis=1)
        Image.fromarray(dual_img).save(f_path)
        frame_paths.append(f_path)
        
    # 3. Save motion heatmap
    heatmap_path = f"{output_prefix}_motion_heatmap.png"
    save_motion_heatmap(frames, heatmap_path)
    frame_paths.append(heatmap_path)
    
    # 4. Save composite filmstrip
    filmstrip_path = f"{output_prefix}_filmstrip.png"
    extract_trajectory_filmstrip(frames, filmstrip_path, num_frames=6, dual_panel=True)
    frame_paths.append(filmstrip_path)
    frame_paths.append(mp4_path)
    
    return frame_paths

def run_agentic_loop(
    generations: int = 3,
    trials_per_gen: int = 20,
    grid_size: int = 384,
    num_steps: int = 3000,
    n_patches: int = 6,
    output_dir: str = "results/agentic_loop"
):
    os.makedirs(output_dir, exist_ok=True)
    state_file = os.path.join(output_dir, "agentic_loop_state.json")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"=== Autonomous Flow-Lenia Discovery Loop Harness ===")
    print(f"Generations: {generations} | Trials/Gen: {trials_per_gen} | Grid: {grid_size}x{grid_size} | Steps: {num_steps}")
    
    rng_key = jax.random.PRNGKey(42)
    H, W = grid_size, grid_size
    K = 9
    
    # Load persistent state if exists
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            loop_state = json.load(f)
        print(f"[Harness State] Loaded existing loop state with {len(loop_state.get('elites', []))} elites.")
    else:
        loop_state = {
            "completed_generations": 0,
            "elites": [],
            "generation_history": []
        }
        
    start_gen = loop_state.get("completed_generations", 0) + 1
    
    for gen in range(start_gen, start_gen + generations):
        print(f"\n--- Generation {gen}/{start_gen + generations - 1} ---")
        rng_key, subk = jax.random.split(rng_key)
        
        imgep_archive, _ = run_imgep_experiment(
            subk,
            n_trials=trials_per_gen,
            n_bootstrap=min(8, trials_per_gen // 2),
            grid_size=256,
            num_steps=1000,
            sample_interval=50
        )
        
        evaluated_candidates = []
        
        for idx, trial in enumerate(imgep_archive.trials):
            cfg = trial["config"]
            radii = jnp.array(cfg["radii"], dtype=jnp.float32)
            kernel_ffts = precompute_kernel_ffts(radii, H, W)
            
            parent_mu = np.array(cfg["mu_presets"])
            parent_sigma = np.array(cfg["sigma_presets"])
            repeat_factor = (n_patches // len(parent_mu)) + 1
            mu_presets = jnp.array(np.tile(parent_mu, (repeat_factor, 1))[:n_patches], dtype=jnp.float32)
            sigma_presets = jnp.array(np.tile(parent_sigma, (repeat_factor, 1))[:n_patches], dtype=jnp.float32)
            
            rng_key, subk_init, subk_rollout = jax.random.split(rng_key, 3)
            state = initialize_multi_patch_state(
                subk_init, H, W, 1, K, n_patches, radii, mu_presets, sigma_presets
            )
            
            v_scale = float(cfg.get("v_scale", 5.2))
            alpha_diff = float(cfg.get("alpha_diffusion", 0.06))
            params = FlowLeniaParams(
                mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K),
                v_scale=v_scale, alpha_diffusion=alpha_diff
            )
            
            _, sampled_mass, _ = run_flow_lenia_rollout(
                state, kernel_ffts, params, subk_rollout, num_steps=num_steps, sample_interval=20,
                mixing_rule='gene_wise', enable_mutation=True
            )
            
            sampled_mass_np = np.array(sampled_mass)
            wt_res = evaluate_watertight_quality_score(sampled_mass_np)
            
            cand_info = {
                "generation": gen,
                "trial_idx": idx,
                "config": cfg,
                "watertight_res": wt_res,
                "sampled_mass": sampled_mass_np
            }
            evaluated_candidates.append(cand_info)
            
        # Filter valid candidates (watertight_score > 0.0)
        valid_cands = [c for c in evaluated_candidates if c["watertight_res"]["is_valid"]]
        valid_cands.sort(key=lambda c: c["watertight_res"]["watertight_score"], reverse=True)
        
        print(f"Generation {gen} Summary: Evaluated={len(evaluated_candidates)} | Valid Elites={len(valid_cands)}")
        
        if valid_cands:
            top_elite = valid_cands[0]
            wt = top_elite["watertight_res"]
            print(f"  Top Elite Score: {wt['watertight_score']:.4f} | CoM Shift: {wt['com_shift']:.2f} px | Mass Pres: {wt['mass_preservation_ratio']:.4f} | Solid Core: {wt['solid_core_ratio_end']:.4f}")
            
            # Save dual-panel frames + MP4 video for top elite
            prefix = os.path.join(frames_dir, f"gen_{gen}_top1")
            frame_paths = save_dual_panel_frames_and_video(top_elite["sampled_mass"], prefix)
            
            loop_state["elites"].append({
                "generation": gen,
                "watertight_score": wt["watertight_score"],
                "status": wt["status"],
                "com_shift": wt["com_shift"],
                "solid_core_ratio": wt["solid_core_ratio_end"],
                "mass_preservation_ratio": wt["mass_preservation_ratio"],
                "config": top_elite["config"],
                "frame_paths": frame_paths
            })
            
        loop_state["completed_generations"] = gen
        loop_state["generation_history"].append({
            "generation": gen,
            "total_trials": len(evaluated_candidates),
            "valid_count": len(valid_cands),
            "top_score": valid_cands[0]["watertight_res"]["watertight_score"] if valid_cands else 0.0
        })
        
        # Persist updated state JSON
        with open(state_file, "w") as f:
            f.write(json.dumps(loop_state, indent=2))
            
    print(f"\n=== Autonomous Discovery Loop Complete ===")
    print(f"State file saved to: {state_file}")
    print(f"Dual-panel trajectory frames & MP4s in: {frames_dir}")

def main():
    parser = argparse.ArgumentParser(description="Autonomous Flow-Lenia Discovery Loop Harness")
    parser.add_argument("--generations", type=int, default=3, help="Number of exploration generations")
    parser.add_argument("--trials_per_gen", type=int, default=20, help="IMGEP trials per generation")
    parser.add_argument("--grid_size", type=int, default=384, help="Grid size")
    parser.add_argument("--steps", type=int, default=3000, help="Simulation rollout steps")
    parser.add_argument("--patches", type=int, default=6, help="Number of species patches")
    parser.add_argument("--output_dir", type=str, default="results/agentic_loop", help="Output directory")
    args = parser.parse_args()
    
    run_agentic_loop(
        generations=args.generations,
        trials_per_gen=args.trials_per_gen,
        grid_size=args.grid_size,
        num_steps=args.steps,
        n_patches=args.patches,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
