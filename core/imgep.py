import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import List, Dict, Any, Tuple, Optional
import json

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts,
    initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.metrics import evaluate_run_metrics

class IMGEPArchive:
    """
    Archive storing candidate trial configurations and their metric vectors.
    """
    def __init__(self, metric_names: List[str] = None):
        if metric_names is None:
            metric_names = ["com_displacement", "ea_raw", "complexity_raw"]
        self.metric_names = metric_names
        self.trials: List[Dict[str, Any]] = []
        self.metrics_matrix: List[np.ndarray] = []
        
    def add_trial(self, config: Dict[str, Any], metrics: Dict[str, float]):
        vec = np.array([metrics[m] for m in self.metric_names], dtype=np.float32)
        self.trials.append({
            "config": config,
            "metrics": metrics
        })
        self.metrics_matrix.append(vec)
        
    def size(self) -> int:
        return len(self.trials)
        
    def get_metrics_array(self) -> np.ndarray:
        if len(self.metrics_matrix) == 0:
            return np.zeros((0, len(self.metric_names)), dtype=np.float32)
        return np.array(self.metrics_matrix, dtype=np.float32)
        
    def get_normalized_metrics_array(self) -> np.ndarray:
        mat = self.get_metrics_array()
        if mat.shape[0] == 0:
            return mat
        min_vals = np.min(mat, axis=0, keepdims=True)
        max_vals = np.max(mat, axis=0, keepdims=True)
        denom = np.maximum(max_vals - min_vals, 1e-8)
        return (mat - min_vals) / denom

    def find_nearest(self, goal_vec_norm: np.ndarray) -> int:
        norm_mat = self.get_normalized_metrics_array()
        if norm_mat.shape[0] == 0:
            return -1
        dists = np.linalg.norm(norm_mat - goal_vec_norm, axis=1)
        return int(np.argmin(dists))

    def select_farthest_point_sampling(self, k: int) -> List[int]:
        norm_mat = self.get_normalized_metrics_array()
        N = norm_mat.shape[0]
        if N <= k:
            return list(range(N))
            
        selected_indices = [np.random.randint(0, N)]
        min_dists = np.linalg.norm(norm_mat - norm_mat[selected_indices[0]], axis=1)
        
        for _ in range(1, k):
            far_idx = int(np.argmax(min_dists))
            selected_indices.append(far_idx)
            new_dists = np.linalg.norm(norm_mat - norm_mat[far_idx], axis=1)
            min_dists = np.minimum(min_dists, new_dists)
            
        return selected_indices

def sample_random_config(rng_key: jnp.ndarray, K: int = 9) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    rng_key, k1, k2, k3, k4, k5, k6 = random.split(rng_key, 7)
    
    # Kernel radii in continuous CA range [6, 15]
    radii = jnp.sort(random.uniform(k1, (K,), minval=6.0, maxval=15.0))
    n_patches = int(random.randint(k2, (), 3, 7))
    
    # Sample mu and sigma strictly in the canonical solid glider regime (Plantec et al. 2025)
    mu_presets = random.uniform(k3, (n_patches, K), minval=0.14, maxval=0.18)
    sigma_presets = random.uniform(k4, (n_patches, K), minval=0.012, maxval=0.018)
    
    v_scale = float(random.uniform(k5, (), minval=4.2, maxval=6.4))
    alpha_diff = float(random.uniform(k6, (), minval=0.04, maxval=0.075))
    
    config = {
        "radii": np.array(radii).tolist(),
        "n_patches": n_patches,
        "mu_presets": np.array(mu_presets).tolist(),
        "sigma_presets": np.array(sigma_presets).tolist(),
        "v_scale": v_scale,
        "alpha_diffusion": alpha_diff
    }
    return rng_key, config

def mutate_config(rng_key: jnp.ndarray, parent_config: Dict[str, Any], std: float = 0.02) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    rng_key, k1, k2, k3, k4, k5, k6 = random.split(rng_key, 7)
    
    parent_radii = np.array(parent_config["radii"])
    radii_noise = np.array(random.normal(k1, parent_radii.shape)) * 0.3
    mut_radii = np.sort(np.clip(parent_radii + radii_noise, 5.0, 16.0))
    
    n_patches = int(parent_config.get("n_patches", 4))
    # 20% probability to alter patch count by +/- 1
    p_patch_mut = float(random.uniform(k6, ()))
    if p_patch_mut < 0.15 and n_patches < 6:
        new_n_patches = n_patches + 1
    elif p_patch_mut > 0.85 and n_patches > 2:
        new_n_patches = n_patches - 1
    else:
        new_n_patches = n_patches
        
    parent_mu = np.array(parent_config["mu_presets"])
    parent_sigma = np.array(parent_config["sigma_presets"])
    
    if new_n_patches > len(parent_mu):
        # Add new patch inheriting from an existing patch with perturbation
        extra_mu = parent_mu[-1:] + np.array(random.normal(k2, (1, parent_mu.shape[1]))) * std
        parent_mu = np.vstack([parent_mu, extra_mu])
        extra_sigma = parent_sigma[-1:] + np.array(random.normal(k3, (1, parent_sigma.shape[1]))) * (std * 0.25)
        parent_sigma = np.vstack([parent_sigma, extra_sigma])
    elif new_n_patches < len(parent_mu):
        parent_mu = parent_mu[:new_n_patches]
        parent_sigma = parent_sigma[:new_n_patches]
        
    mu_noise = np.array(random.normal(k2, parent_mu.shape)) * std
    mut_mu = np.clip(parent_mu + mu_noise, 0.13, 0.20)
    
    sigma_noise = np.array(random.normal(k3, parent_sigma.shape)) * (std * 0.25)
    mut_sigma = np.clip(parent_sigma + sigma_noise, 0.010, 0.020)
    
    v_s_noise = float(random.normal(k4, ())) * 0.35
    mut_v_scale = float(np.clip(parent_config.get("v_scale", 5.2) + v_s_noise, 4.2, 6.5))
    
    a_diff_noise = float(random.normal(k5, ())) * 0.01
    mut_alpha_diff = float(np.clip(parent_config.get("alpha_diffusion", 0.06) + a_diff_noise, 0.04, 0.08))
    
    mut_config = {
        "radii": mut_radii.tolist(),
        "n_patches": new_n_patches,
        "mu_presets": mut_mu.tolist(),
        "sigma_presets": mut_sigma.tolist(),
        "v_scale": mut_v_scale,
        "alpha_diffusion": mut_alpha_diff
    }
    return rng_key, mut_config

def evaluate_single_config_jax(
    rng_key: jnp.ndarray,
    config: Dict[str, Any],
    grid_size: int = 256,
    num_steps: int = 2000,
    sample_interval: int = 250,
    wall_mask: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Dict[str, float], np.ndarray]:
    radii = jnp.array(config["radii"], dtype=jnp.float32)
    K = len(radii)
    H, W = grid_size, grid_size
    C = 1
    
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    mu_presets = jnp.array(config["mu_presets"], dtype=jnp.float32)
    sigma_presets = jnp.array(config["sigma_presets"], dtype=jnp.float32)
    
    rng_key, subk = random.split(rng_key)
    state = initialize_multi_patch_state(
        subk, H, W, C, K,
        n_patches=config["n_patches"],
        kernel_radii=radii,
        mu_presets=mu_presets,
        sigma_presets=sigma_presets,
        wall_mask=wall_mask
    )
    
    v_scale = float(config.get("v_scale", 5.0))
    alpha_diff = float(config.get("alpha_diffusion", 0.05))
    
    params = FlowLeniaParams(
        mu=mu_presets[0],
        sigma=sigma_presets[0],
        weights=jnp.full((K,), 1.0 / K),
        v_scale=v_scale,
        alpha_diffusion=alpha_diff
    )
    
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk,
        num_steps=num_steps,
        sample_interval=sample_interval,
        wall_mask=wall_mask,
        enable_mutation=True
    )
    
    final_mass_np = np.array(final_state.mass)
    sampled_mass_np = np.array(sampled_mass)
    sampled_gid_np = np.array(sampled_gid)
    
    metrics = evaluate_run_metrics(
        final_mass_np, sampled_mass_np, sampled_gid_np,
        total_steps=num_steps, n_genomes=config["n_patches"]
    )
    
    return rng_key, metrics, sampled_mass_np

def run_imgep_experiment(
    rng_key: jnp.ndarray,
    n_trials: int = 50,
    n_bootstrap: int = 10,
    grid_size: int = 256,
    num_steps: int = 2000,
    sample_interval: int = 250,
    wall_mask: Optional[jnp.ndarray] = None,
    seed_configs: Optional[List[Dict[str, Any]]] = None
) -> Tuple[IMGEPArchive, List[np.ndarray]]:
    if wall_mask is not None:
        metric_names = ["com_norm", "corridor_coverage", "ea_norm"]
    else:
        metric_names = ["com_displacement", "ea_raw", "complexity_raw"]
        
    archive = IMGEPArchive(metric_names=metric_names)
    rollouts: List[np.ndarray] = []
    
    print(f"[IMGEP] Starting Bootstrap Phase ({n_bootstrap} trials)... Goal Metrics: {metric_names}")
    
    # 1. Evaluate seed configs if provided (elite lineage continuity)
    seed_count = 0
    if seed_configs:
        for sc in seed_configs:
            if seed_count >= n_bootstrap:
                break
            rng_key, metrics, sampled_mass = evaluate_single_config_jax(
                rng_key, sc, grid_size=grid_size, num_steps=num_steps,
                sample_interval=sample_interval, wall_mask=wall_mask
            )
            archive.add_trial(sc, metrics)
            rollouts.append(sampled_mass)
            seed_count += 1
            
    # 2. Fill remaining bootstrap slots with random sampling
    for b in range(seed_count, n_bootstrap):
        rng_key, cfg = sample_random_config(rng_key)
        rng_key, metrics, sampled_mass = evaluate_single_config_jax(
            rng_key, cfg, grid_size=grid_size, num_steps=num_steps,
            sample_interval=sample_interval, wall_mask=wall_mask
        )
        archive.add_trial(cfg, metrics)
        rollouts.append(sampled_mass)
        
    print(f"[IMGEP] Starting Goal Exploration Phase ({n_trials - n_bootstrap} trials)...")
    for t in range(n_bootstrap, n_trials):
        rng_key, k_goal = random.split(rng_key)
        # Sample goals across behavior space with focus on high motility & evolutionary dynamism
        goal_norm = np.array(random.uniform(k_goal, (len(metric_names),), minval=0.1, maxval=1.0))
        
        nearest_idx = archive.find_nearest(goal_norm)
        parent_cfg = archive.trials[nearest_idx]["config"]
        
        rng_key, mut_cfg = mutate_config(rng_key, parent_cfg)
        
        rng_key, metrics, sampled_mass = evaluate_single_config_jax(
            rng_key, mut_cfg, grid_size=grid_size, num_steps=num_steps,
            sample_interval=sample_interval, wall_mask=wall_mask
        )
        archive.add_trial(mut_cfg, metrics)
        rollouts.append(sampled_mass)
        
    return archive, rollouts

def run_random_search_experiment(
    rng_key: jnp.ndarray,
    n_trials: int = 50,
    grid_size: int = 256,
    num_steps: int = 2000,
    sample_interval: int = 250,
    wall_mask: Optional[jnp.ndarray] = None
) -> Tuple[IMGEPArchive, List[np.ndarray]]:
    if wall_mask is not None:
        metric_names = ["com_norm", "corridor_coverage", "ea_norm"]
    else:
        metric_names = ["com_displacement", "ea_raw", "complexity_raw"]
        
    archive = IMGEPArchive(metric_names=metric_names)
    rollouts: List[np.ndarray] = []
    print(f"[Random Search] Starting Random Search ({n_trials} trials)...")
    
    for t in range(n_trials):
        rng_key, cfg = sample_random_config(rng_key)
        rng_key, metrics, sampled_mass = evaluate_single_config_jax(
            rng_key, cfg, grid_size=grid_size, num_steps=num_steps,
            sample_interval=sample_interval, wall_mask=wall_mask
        )
        archive.add_trial(cfg, metrics)
        rollouts.append(sampled_mass)
        
    return archive, rollouts
