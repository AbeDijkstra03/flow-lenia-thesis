import gzip
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

def compute_evolutionary_activity(
    sampled_mass_frames: np.ndarray,
    sampled_genome_maps: np.ndarray,
    n_genomes: int = 10
) -> float:
    """
    Compute non-neutral evolutionary activity:
    Sum of squared positive increases in per-genome mass-share:
    EA = sum_{t} sum_{g} max(0, s_g(t) - s_g(t-1))^2
    sampled_mass_frames: shape (S, C, H, W)
    sampled_genome_maps: shape (S, H, W)
    """
    S, C, H, W = sampled_mass_frames.shape
    mass_0 = sampled_mass_frames[:, 0, :, :] # (S, H, W)
    
    genome_shares = np.zeros((S, n_genomes), dtype=np.float32)
    
    for t in range(S):
        total_m = np.sum(mass_0[t]) + 1e-8
        for g in range(n_genomes):
            mask_g = (sampled_genome_maps[t] == g)
            mass_g = np.sum(mass_0[t] * mask_g)
            genome_shares[t, g] = mass_g / total_m
            
    diffs = np.diff(genome_shares, axis=0) # (S-1, n_genomes)
    pos_diffs = np.maximum(0.0, diffs)
    
    # Quadratic activity formula: max(0, delta_s)^2
    total_ea = float(np.sum(pos_diffs**2))
    return total_ea

def compute_compression_complexity(mass_grid: np.ndarray, quant_levels: int = 256) -> float:
    """
    Compute compression-based complexity:
    gzip size of quantized rollout/final state in bytes.
    mass_grid: shape (H, W) or (C, H, W)
    """
    quantized = np.clip(mass_grid * (quant_levels - 1), 0, quant_levels - 1).astype(np.uint8)
    compressed = gzip.compress(quantized.tobytes())
    return float(len(compressed))

def compute_shannon_entropy(prob_dist: np.ndarray) -> float:
    """Compute Shannon entropy H in bits."""
    p_flat = prob_dist.ravel()
    p_flat = p_flat[p_flat > 1e-12]
    p_flat = p_flat / (np.sum(p_flat) + 1e-12)
    return float(-np.sum(p_flat * np.log2(p_flat)))

def compute_multi_scale_entropy(
    mass_grid: np.ndarray,
    scales: Tuple[int, ...] = (2, 4, 8, 16, 32)
) -> float:
    """
    Compute multi-scale entropy across downsampled block resolutions.
    """
    H, W = mass_grid.shape[-2:]
    entropies = []
    
    for scale in scales:
        if scale > H or scale > W:
            continue
        h_blocks = H // scale
        w_blocks = W // scale
        cropped = mass_grid[:h_blocks * scale, :w_blocks * scale]
        
        downsampled = cropped.reshape(h_blocks, scale, w_blocks, scale).mean(axis=(1, 3))
        ent = compute_shannon_entropy(downsampled)
        entropies.append(ent)
        
    if len(entropies) == 0:
        return 0.0
    return float(np.mean(entropies))

def compute_center_of_mass(mass_grid: np.ndarray) -> Tuple[float, float]:
    """Compute (y, x) center of mass coordinates."""
    H, W = mass_grid.shape[-2:]
    total_m = np.sum(mass_grid) + 1e-8
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cy = np.sum(mass_grid * yy) / total_m
    cx = np.sum(mass_grid * xx) / total_m
    return float(cy), float(cx)

def compute_wall_corridor_metrics(
    init_mass: np.ndarray,
    final_mass: np.ndarray
) -> Tuple[float, float]:
    """
    Compute CoM displacement and grid coverage beyond initial area for Wall Obstacle experiments.
    """
    cy_init, cx_init = compute_center_of_mass(init_mass[0] if init_mass.ndim == 3 else init_mass)
    cy_final, cx_final = compute_center_of_mass(final_mass[0] if final_mass.ndim == 3 else final_mass)
    
    com_displacement = float(np.sqrt((cy_final - cy_init)**2 + (cx_final - cx_init)**2))
    
    # Coverage: fraction of cells on right side of grid (x > W // 2) with active mass > 0.05
    H, W = final_mass.shape[-2:]
    right_side_mass = (final_mass[0] if final_mass.ndim == 3 else final_mass)[:, W // 2:]
    coverage = float(np.mean(right_side_mass > 0.05))
    
    return com_displacement, coverage

def evaluate_watertight_quality_score(
    sampled_mass_frames: np.ndarray
) -> Dict[str, Any]:
    """
    Watertight quality evaluation suite for Flow-Lenia rollouts.
    Disqualifies:
    1. Mass Dissipation (R_mass < 0.60)
    2. Mass Explosion (R_mass > 5.00)
    3. Hollow Outline Degeneration (R_core < 0.50)
    4. Frozen Still-Lifes (v_CoM < 5.0 px)
    5. Unconstrained Chaos (Coverage > 0.25)
    """
    if sampled_mass_frames.ndim == 4:
        sampled_mass_frames = sampled_mass_frames[:, 0, :, :]
        
    S, H, W = sampled_mass_frames.shape
    
    init_mass_sum = float(np.sum(sampled_mass_frames[0])) + 1e-8
    final_mass_sum = float(np.sum(sampled_mass_frames[-1])) + 1e-8
    mass_preservation_ratio = final_mass_sum / init_mass_sum
    
    solid_core_mask_end = (sampled_mass_frames[-1] >= 0.15)
    solid_core_mass_end = float(np.sum(sampled_mass_frames[-1] * solid_core_mask_end))
    solid_core_ratio_end = solid_core_mass_end / final_mass_sum
    
    cy0, cx0 = compute_center_of_mass(sampled_mass_frames[0])
    cy_end, cx_end = compute_center_of_mass(sampled_mass_frames[-1])
    com_shift = float(np.sqrt((cy_end - cy0)**2 + (cx_end - cx0)**2))
    
    active_coverage = float(np.mean(sampled_mass_frames[-1] > 0.05))
    heatmap_activity = float(np.mean(np.abs(np.diff(sampled_mass_frames, axis=0))))
    
    status = "SOLID_GLIDER_SOLITON"
    is_valid = True
    
    min_required_com_shift = 5.0
    if mass_preservation_ratio < 0.60:
        status = "MASS_DISSIPATED"
        is_valid = False
    elif mass_preservation_ratio > 5.00:
        status = "MASS_EXPLODED"
        is_valid = False
    elif solid_core_ratio_end < 0.50:
        status = "HOLLOW_OUTLINE_DEGENERATION"
        is_valid = False
    elif com_shift < min_required_com_shift:
        status = "FROZEN_STILL_LIFE"
        is_valid = False
    elif active_coverage > 0.25:
        status = "UNCONSTRAINED_CHAOS"
        is_valid = False
        
    if is_valid:
        watertight_score = com_shift * solid_core_ratio_end * (1.0 + heatmap_activity * 50.0)
    else:
        watertight_score = 0.0
        
    return {
        "watertight_score": watertight_score,
        "is_valid": is_valid,
        "status": status,
        "mass_preservation_ratio": mass_preservation_ratio,
        "solid_core_ratio_end": solid_core_ratio_end,
        "com_shift": com_shift,
        "active_coverage": active_coverage,
        "heatmap_activity": heatmap_activity
    }

def evaluate_run_metrics(
    final_mass: np.ndarray,
    sampled_mass_frames: np.ndarray,
    sampled_genome_maps: np.ndarray,
    total_steps: int,
    n_genomes: int = 10
) -> Dict[str, float]:
    """
    Compute complete 3-D metric suite (raw & normalized) + Watertight Quality Metrics.
    """
    # 1. Non-Neutral Evolutionary Activity (Quadratic formula)
    ea_raw = compute_evolutionary_activity(sampled_mass_frames, sampled_genome_maps, n_genomes=n_genomes)
    
    # 2. Compression Complexity
    comp_raw = compute_compression_complexity(final_mass[0] if final_mass.ndim == 3 else final_mass)
    
    # 3. Multi-Scale Entropy
    entropy_raw = compute_multi_scale_entropy(final_mass[0] if final_mass.ndim == 3 else final_mass)
    
    # 4. Center of Mass & Coverage Metrics
    com_displacement, coverage = compute_wall_corridor_metrics(sampled_mass_frames[0], final_mass)
    
    # 5. Watertight Quality Filter Evaluation
    watertight_res = evaluate_watertight_quality_score(sampled_mass_frames)
    
    H, W = final_mass.shape[-2:]
    total_cells = float(H * W)
    total_mass = float(np.sum(final_mass[0] if final_mass.ndim == 3 else final_mass)) + 1e-8
    horizon = float(total_steps)
    
    ea_norm = ea_raw / (horizon / 1000.0) # normalized per 1k steps
    comp_norm = comp_raw / total_cells # bytes per cell
    entropy_norm = entropy_raw / np.log2(total_cells + 1e-8)
    com_norm = com_displacement / np.sqrt(H**2 + W**2) # normalized displacement in [0, 1]
    
    return {
        "ea_raw": ea_raw,
        "complexity_raw": comp_raw,
        "entropy_raw": entropy_raw,
        "com_displacement": com_displacement,
        "corridor_coverage": coverage,
        "ea_norm": ea_norm,
        "complexity_norm": comp_norm,
        "entropy_norm": entropy_norm,
        "com_norm": com_norm,
        "total_mass": total_mass,
        "watertight_score": watertight_res["watertight_score"],
        "watertight_valid": float(watertight_res["is_valid"]),
        "solid_core_ratio": watertight_res["solid_core_ratio_end"],
        "mass_preservation_ratio": watertight_res["mass_preservation_ratio"],
        "grid_size": H,
        "horizon": total_steps
    }
