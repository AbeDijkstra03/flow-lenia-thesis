"""
Scientific Visualization & Video Export Engine for Flow-Lenia.

Adheres to SOTA literature standards (Michel et al. 2025/2026, Plantec et al. 2025):
- Broadcast-grade H.264 MP4 video encoding (libx264, yuv420p, CRF 18)
- Dual-panel scientific layout:
  * Left: Multi-Species categorical palette (or Perceptually uniform Plasma with soft log contrast)
  * Right: Absolute physical mass density [0, 1] with obstacle boundaries and CoM trajectory
- 6-frame horizontal composite trajectory filmstrips for LaTeX/publication figures
- Cumulative motion displacement heatmaps
- Compressed NumPy (.npz) and JSON metadata archiving
"""

import os
import json
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import imageio.v3 as iio

# Vibrant 10-color categorical species palette for ecological tracking
SPECIES_PALETTE = np.array([
    [0, 230, 255],   # 0: Vibrant Cyan
    [255, 0, 128],   # 1: Vibrant Magenta
    [50, 255, 50],   # 2: Vibrant Lime
    [255, 170, 0],   # 3: Amber Orange
    [180, 50, 255],  # 4: Electric Purple
    [255, 80, 80],   # 5: Coral Red
    [0, 255, 180],   # 6: Aquamarine
    [255, 220, 0],   # 7: Canary Yellow
    [100, 150, 255], # 8: Sky Blue
    [255, 105, 180]  # 9: Hot Pink
], dtype=np.float32)

def colorize_frame_plasma_log(
    frame_2d: np.ndarray,
    wall_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply fixed logarithmic contrast scaling and perceptually uniform Plasma colormap.
    frame_2d: (H, W) array of float density values.
    Returns: (H, W, 3) uint8 RGB array.
    """
    # Fixed absolute physical reference scaling (zero frame-to-frame flicker)
    log_f = np.log1p(np.maximum(0.0, frame_2d) * 1.8)
    norm_f = np.clip(log_f / np.log1p(1.8), 0.0, 1.0)
        
    plasma_rgba = cm.plasma(norm_f)
    plasma_rgb = (plasma_rgba[:, :, :3] * 255.0).astype(np.uint8)
    
    if wall_mask is not None:
        wall_bool = (wall_mask < 0.5)
        plasma_rgb[wall_bool] = [30, 144, 255] # DodgerBlue obstacle wall
        
    return plasma_rgb

def colorize_multi_species_frame(
    mass_2d: np.ndarray,
    genome_id_map: Optional[np.ndarray] = None,
    wall_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Render multi-species ecological frame where each discrete genome/species
    has a distinct vibrant categorical color, modulated by local mass density.
    """
    if genome_id_map is None:
        return colorize_frame_plasma_log(mass_2d, wall_mask=wall_mask)
        
    H, W = mass_2d.shape
    gid_safe = np.clip(genome_id_map, 0, len(SPECIES_PALETTE) - 1)
    species_colors = SPECIES_PALETTE[gid_safe] # (H, W, 3)
    
    # Fixed absolute physical reference intensity (zero frame-to-frame flicker)
    log_f = np.log1p(np.maximum(0.0, mass_2d) * 1.8)
    intensity = np.clip(log_f / np.log1p(1.8), 0.0, 1.0)[:, :, None]
    
    rgb = np.clip(species_colors * intensity, 0, 255).astype(np.uint8)
    
    # Vacuum cells (gid == -1 or mass < 0.01) remain dark midnight navy background
    vacuum_mask = (genome_id_map < 0) | (mass_2d < 0.01)
    rgb[vacuum_mask] = [5, 5, 22]
    
    if wall_mask is not None:
        wall_bool = (wall_mask < 0.5)
        rgb[wall_bool] = [30, 144, 255] # DodgerBlue obstacle wall
        
    return rgb

def render_physical_frame(
    frame_2d: np.ndarray,
    wall_mask: Optional[np.ndarray] = None,
    com_pos: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Render absolute physical mass density [0.0, 1.0] in grayscale with
    obstacle walls and center-of-mass overlay.
    """
    H, W = frame_2d.shape
    gray = np.clip(frame_2d * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    
    if wall_mask is not None:
        wall_bool = (wall_mask < 0.5)
        rgb[wall_bool] = [30, 144, 255] # DodgerBlue obstacle wall
        
    if com_pos is not None:
        cy, cx = int(round(com_pos[0])), int(round(com_pos[1]))
        if 0 <= cy < H and 0 <= cx < W:
            y_min, y_max = max(0, cy - 2), min(H, cy + 3)
            x_min, x_max = max(0, cx - 2), min(W, cx + 3)
            rgb[y_min:y_max, cx] = [255, 69, 0] # Red-Orange crosshair
            rgb[cy, x_min:x_max] = [255, 69, 0]
            
    return rgb

def render_dual_panel_rgb(
    mass_2d: np.ndarray,
    genome_id_map: Optional[np.ndarray] = None,
    n_patches: Optional[int] = None,
    wall_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Render a single dual-panel composite RGB image.
    Left: Multi-species or Plasma log colorization.
    Right: Absolute physical mass density [0, 1] with obstacle boundaries.
    """
    H, W = mass_2d.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    tot_m = np.sum(mass_2d) + 1e-8
    cy = float(np.sum(mass_2d * yy) / tot_m)
    cx = float(np.sum(mass_2d * xx) / tot_m)
    
    left_img = colorize_multi_species_frame(mass_2d, genome_id_map, wall_mask=wall_mask)
    right_img = render_physical_frame(mass_2d, wall_mask=wall_mask, com_pos=(cy, cx))
    combined = np.concatenate([left_img, right_img], axis=1)
    return combined

def save_rollout_mp4(
    frames: np.ndarray,
    filepath: str,
    fps: int = 20,
    dual_panel: bool = True,
    wall_mask: Optional[np.ndarray] = None,
    genome_id_maps: Optional[np.ndarray] = None,
    crf: int = 18
) -> str:
    """
    Save simulation rollout frames to broadcast-grade H.264 MP4 video.
    frames: shape (S, C, H, W) or (S, H, W)
    genome_id_maps: optional shape (S, H, W) for multi-species colorization
    """
    if frames.ndim == 4:
        frames = frames[:, 0, :, :]
        
    S, H, W = frames.shape
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    if not filepath.endswith(".mp4"):
        filepath = os.path.splitext(filepath)[0] + ".mp4"
        
    rgb_frames = []
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    for f in range(S):
        frame_2d = frames[f]
        tot_m = np.sum(frame_2d) + 1e-8
        cy = float(np.sum(frame_2d * yy) / tot_m)
        cx = float(np.sum(frame_2d * xx) / tot_m)
        
        gid_f = genome_id_maps[f] if genome_id_maps is not None else None
        w_mask_f = wall_mask[f] if (wall_mask is not None and np.asarray(wall_mask).ndim == 3) else wall_mask
        left_img = colorize_multi_species_frame(frame_2d, gid_f, wall_mask=w_mask_f)
        
        if dual_panel:
            right_img = render_physical_frame(frame_2d, wall_mask=w_mask_f, com_pos=(cy, cx))
            combined = np.concatenate([left_img, right_img], axis=1)
            rgb_frames.append(combined)
        else:
            rgb_frames.append(left_img)
            
    stacked = np.stack(rgb_frames, axis=0)
    
    iio.imwrite(
        filepath,
        stacked,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", str(crf), "-preset", "medium"]
    )
    return filepath

def extract_trajectory_filmstrip(
    frames: np.ndarray,
    output_path: str,
    num_frames: int = 6,
    dual_panel: bool = True,
    wall_mask: Optional[np.ndarray] = None,
    genome_id_maps: Optional[np.ndarray] = None
) -> str:
    """
    Extract a publication-ready horizontal composite filmstrip PNG (e.g. 6 evenly spaced steps)
    ideal for LaTeX subfigures and thesis chapters. Supports static (H, W) or dynamic (S, H, W) wall_mask.
    """
    if frames.ndim == 4:
        frames = frames[:, 0, :, :]
        
    S, H, W = frames.shape
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    pcts = np.linspace(0.0, 1.0, num_frames)
    panel_list = []
    
    for p in pcts:
        idx = min(int(p * (S - 1)), S - 1)
        frame_2d = frames[idx]
        gid_f = genome_id_maps[idx] if genome_id_maps is not None else None
        w_mask_f = wall_mask[idx] if (wall_mask is not None and np.asarray(wall_mask).ndim == 3) else wall_mask
        
        left_rgb = colorize_multi_species_frame(frame_2d, gid_f, wall_mask=w_mask_f)
        if dual_panel:
            right_rgb = render_physical_frame(frame_2d, wall_mask=w_mask_f)
            cell_img = np.concatenate([left_rgb, right_rgb], axis=1)
        else:
            cell_img = left_rgb
            
        pil_cell = Image.fromarray(cell_img)
        draw = ImageDraw.Draw(pil_cell)
        
        pct_label = f"t = {int(p * 100)}%"
        draw.rectangle([(5, 5), (65, 22)], fill=(0, 0, 0, 180))
        draw.text((10, 7), pct_label, fill=(255, 255, 255))
        
        panel_list.append(np.array(pil_cell))
        
    composite_strip = np.concatenate(panel_list, axis=1)
    Image.fromarray(composite_strip).save(output_path)
    return output_path

def save_motion_heatmap(frames: np.ndarray, output_path: str) -> str:
    """
    Compute and save cumulative absolute mass displacement heatmap.
    """
    if frames.ndim == 4:
        frames = frames[:, 0, :, :]
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    diffs = np.sum(np.abs(np.diff(frames, axis=0)), axis=0)
    max_d = float(np.max(diffs)) + 1e-8
    norm_diffs = np.clip(diffs / max_d, 0.0, 1.0)
    
    magma_rgba = cm.magma(norm_diffs)
    magma_rgb = (magma_rgba[:, :, :3] * 255.0).astype(np.uint8)
    
    Image.fromarray(magma_rgb).save(output_path)
    return output_path

def save_experiment_artifacts(
    sampled_mass_frames: np.ndarray,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
    prefix: str = "",
    fps: int = 20,
    wall_mask: Optional[np.ndarray] = None,
    genome_id_maps: Optional[np.ndarray] = None,
    use_subfolder: bool = True
) -> Dict[str, str]:
    """
    One-shot comprehensive scientific artifact generator:
    Saves clean per-simulation package containing:
    - rollout.mp4 (Broadcast dual-panel video)
    - trajectory_filmstrip.png (6-frame composite LaTeX figure)
    - motion_heatmap.png (Cumulative motion trajectory)
    - data.npz (Raw simulation arrays)
    - metadata.json (Full hyperparameter & metric logs)
    """
    if use_subfolder and prefix:
        target_dir = os.path.join(output_dir, prefix)
        file_prefix = ""
    else:
        target_dir = output_dir
        file_prefix = f"{prefix}_" if prefix else ""
        
    os.makedirs(target_dir, exist_ok=True)
    
    video_path = os.path.join(target_dir, f"{file_prefix}rollout.mp4")
    filmstrip_path = os.path.join(target_dir, f"{file_prefix}trajectory_filmstrip.png")
    heatmap_path = os.path.join(target_dir, f"{file_prefix}motion_heatmap.png")
    data_path = os.path.join(target_dir, f"{file_prefix}data.npz")
    meta_path = os.path.join(target_dir, f"{file_prefix}metadata.json")
    
    save_rollout_mp4(
        sampled_mass_frames, video_path, fps=fps, dual_panel=True,
        wall_mask=wall_mask, genome_id_maps=genome_id_maps
    )
    extract_trajectory_filmstrip(
        sampled_mass_frames, filmstrip_path, num_frames=6, dual_panel=True,
        wall_mask=wall_mask, genome_id_maps=genome_id_maps
    )
    save_motion_heatmap(sampled_mass_frames, heatmap_path)
    
    save_dict = {"sampled_mass": sampled_mass_frames}
    if genome_id_maps is not None:
        save_dict["sampled_gid"] = genome_id_maps
    np.savez_compressed(data_path, **save_dict, allow_pickle=False)
    
    metadata = {
        "prefix": prefix,
        "config": config,
        "metrics": metrics,
        "video_file": os.path.basename(video_path),
        "filmstrip_file": os.path.basename(filmstrip_path),
        "heatmap_file": os.path.basename(heatmap_path),
        "data_file": os.path.basename(data_path)
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    return {
        "video": video_path,
        "filmstrip": filmstrip_path,
        "heatmap": heatmap_path,
        "data": data_path,
        "metadata": meta_path
    }
