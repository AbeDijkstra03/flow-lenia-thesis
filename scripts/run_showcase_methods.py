#!/usr/bin/env python3
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.visualization import save_rollout_mp4, extract_trajectory_filmstrip

def run_method_1_gene_mutation(rng_key, grid_size=384, steps=3600, n_patches=6):
    """Method 1: Gene-wise mixing + periodic spatial parameter mutation sweeps."""
    print("\n=== Method 1: Gene Mutation Sweep ===")
    H, W = grid_size, grid_size
    K = 9
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.13, maxval=0.22)
    sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.011, maxval=0.024)
    
    rng_key, subk = random.split(rng_key)
    state = initialize_multi_patch_state(subk, H, W, 1, K, n_patches, radii, mu_presets, sigma_presets)
    params = FlowLeniaParams(mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K))
    
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, _ = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk, num_steps=steps, sample_interval=3, mixing_rule='gene_wise', enable_mutation=True, mutation_interval=100
    )
    return np.array(sampled_mass)

def run_method_2_negotiation_competition(rng_key, grid_size=384, steps=3600, n_patches=6):
    """Method 2: Negotiation Rule (growth affinity territorial competition)."""
    print("\n=== Method 2: Negotiation Competition Rule ===")
    H, W = grid_size, grid_size
    K = 9
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.13, maxval=0.22)
    sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.011, maxval=0.024)
    
    rng_key, subk = random.split(rng_key)
    state = initialize_multi_patch_state(subk, H, W, 1, K, n_patches, radii, mu_presets, sigma_presets)
    params = FlowLeniaParams(mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K), beta=3.0)
    
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, _ = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk, num_steps=steps, sample_interval=3, mixing_rule='negotiation', enable_mutation=True, mutation_interval=100
    )
    return np.array(sampled_mass)

def run_method_3_resource_depletion(rng_key, grid_size=384, steps=3600, n_patches=6):
    """Method 3: Dynamic Resource Depletion Wake (forces continuous migration)."""
    print("\n=== Method 3: Resource Depletion Foraging Wake ===")
    H, W = grid_size, grid_size
    K = 9
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.13, maxval=0.22)
    sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.011, maxval=0.024)
    
    rng_key, subk = random.split(rng_key)
    state = initialize_multi_patch_state(subk, H, W, 1, K, n_patches, radii, mu_presets, sigma_presets)
    params = FlowLeniaParams(mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K), depletion_rate=0.015, regen_rate=0.005)
    
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, _ = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk, num_steps=steps, sample_interval=3, mixing_rule='gene_wise', enable_mutation=True, enable_depletion=True, mutation_interval=100
    )
    return np.array(sampled_mass)

def main():
    output_dir = "results/showcase"
    os.makedirs(output_dir, exist_ok=True)
    rng_key = random.PRNGKey(42)
    
    # Method 1
    rng_key, subk = random.split(rng_key)
    mass1 = run_method_1_gene_mutation(subk)
    save_rollout_mp4(mass1, os.path.join(output_dir, "showcase_1_gene_mutation.mp4"), fps=20, dual_panel=True)
    extract_trajectory_filmstrip(mass1, os.path.join(output_dir, "showcase_1_filmstrip.png"))
    
    # Method 2
    rng_key, subk = random.split(rng_key)
    mass2 = run_method_2_negotiation_competition(subk)
    save_rollout_mp4(mass2, os.path.join(output_dir, "showcase_2_negotiation.mp4"), fps=20, dual_panel=True)
    extract_trajectory_filmstrip(mass2, os.path.join(output_dir, "showcase_2_filmstrip.png"))
    
    # Method 3
    rng_key, subk = random.split(rng_key)
    mass3 = run_method_3_resource_depletion(subk)
    save_rollout_mp4(mass3, os.path.join(output_dir, "showcase_3_resource_depletion.mp4"), fps=20, dual_panel=True)
    extract_trajectory_filmstrip(mass3, os.path.join(output_dir, "showcase_3_filmstrip.png"))

    print(f"\nAll 3 showcase MP4 videos & filmstrips saved to: {output_dir}/")

if __name__ == "__main__":
    main()
