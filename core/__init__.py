"""
Core Flow Lenia JAX Simulation Engine, Metrics, Configs, and Visualization Framework.
"""
import os
import sys

# Prepend virtualenv's CUDA NVCC to PATH if available to guarantee Blackwell (RTX 5090 sm_120) compatibility
venv_nvcc = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cuda_nvcc", "bin")
if os.path.isdir(venv_nvcc) and venv_nvcc not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_nvcc + ":" + os.environ.get("PATH", "")

from core.flow_lenia_jax import (
    FlowLeniaParams,
    FlowLeniaState,
    precompute_kernel_ffts,
    initialize_multi_patch_state,
    flow_lenia_step_single,
    run_flow_lenia_rollout
)
from core.metrics import evaluate_run_metrics, evaluate_watertight_quality_score
from core.imgep import IMGEPArchive, run_imgep_experiment, run_random_search_experiment
from core.environment import create_homogeneous_mask, create_wall_obstacle_mask
from core.visualization import (
    save_rollout_mp4,
    extract_trajectory_filmstrip,
    save_motion_heatmap,
    save_experiment_artifacts
)
from core.config import (
    SimulationConfig,
    PhysicsConfig,
    BiologyConfig,
    EnvironmentConfig,
    load_config,
    save_config
)

__all__ = [
    "FlowLeniaParams",
    "FlowLeniaState",
    "precompute_kernel_ffts",
    "initialize_multi_patch_state",
    "flow_lenia_step_single",
    "run_flow_lenia_rollout",
    "evaluate_run_metrics",
    "evaluate_watertight_quality_score",
    "IMGEPArchive",
    "run_imgep_experiment",
    "run_random_search_experiment",
    "create_homogeneous_mask",
    "create_wall_obstacle_mask",
    "save_rollout_mp4",
    "extract_trajectory_filmstrip",
    "save_motion_heatmap",
    "save_experiment_artifacts",
    "SimulationConfig",
    "PhysicsConfig",
    "BiologyConfig",
    "EnvironmentConfig",
    "load_config",
    "save_config"
]
