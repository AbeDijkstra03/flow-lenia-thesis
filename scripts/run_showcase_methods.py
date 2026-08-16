#!/usr/bin/env python3
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.visualization import save_experiment_artifacts

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
    params = FlowLeniaParams(mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K), v_scale=5.4, alpha_diffusion=0.055)
    
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk, num_steps=steps, sample_interval=3, mixing_rule='gene_wise', enable_mutation=True, mutation_interval=60
    )
    return np.array(sampled_mass), np.array(sampled_gid)

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
    params = FlowLeniaParams(mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K), beta=3.0, v_scale=5.4, alpha_diffusion=0.055)
    
    rng_key, subk = random.split(rng_key)
    final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
        state, kernel_ffts, params, subk, num_steps=steps, sample_interval=3, mixing_rule='negotiation', enable_mutation=True, mutation_interval=60
    )
    return np.array(sampled_mass), np.array(sampled_gid)

def main():
    output_dir = "results/showcase"
    os.makedirs(output_dir, exist_ok=True)
    rng_key = random.PRNGKey(42)
    
    # Method 1: Gene Mutation Sweep
    print("\n--- Running Showcase Method 1: Gene Mutation ---")
    rng_key, subk = random.split(rng_key)
    mass1, gid1 = run_method_1_gene_mutation(subk)
    save_experiment_artifacts(
        sampled_mass_frames=mass1,
        metrics={"method": "gene_mutation", "mixing_rule": "gene_wise", "mutation_interval": 60},
        config={"grid_size": 384, "steps": 3600, "patches": 6, "v_scale": 5.4, "alpha_diffusion": 0.055},
        output_dir=output_dir,
        prefix="method_1_gene_mutation",
        fps=20,
        genome_id_maps=gid1
    )
    print("Saved complete artifacts to: results/showcase/method_1_gene_mutation/")
    
    # Method 2: Softmax Negotiation Competition Rule
    print("\n--- Running Showcase Method 2: Negotiation Competition Rule ---")
    rng_key, subk = random.split(rng_key)
    mass2, gid2 = run_method_2_negotiation_competition(subk)
    save_experiment_artifacts(
        sampled_mass_frames=mass2,
        metrics={"method": "negotiation_rule", "mixing_rule": "negotiation", "beta": 3.0},
        config={"grid_size": 384, "steps": 3600, "patches": 6, "v_scale": 5.4, "alpha_diffusion": 0.055, "beta": 3.0},
        output_dir=output_dir,
        prefix="method_2_negotiation_rule",
        fps=20,
        genome_id_maps=gid2
    )
    print("Saved complete artifacts to: results/showcase/method_2_negotiation_rule/")
    
    print("\nAll showcase experiment artifacts saved successfully to: results/showcase/")

if __name__ == "__main__":
    main()
