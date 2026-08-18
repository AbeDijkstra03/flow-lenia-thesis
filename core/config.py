"""
Typed Configuration Engine for Flow-Lenia Simulations.

Parses, validates, and manages simulation, physics, biology, and environmental configurations
from YAML files with seamless command-line override support.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
import os
import yaml

@dataclass
class PhysicsConfig:
    dt: float = 0.05
    beta: float = 2.0
    depletion_rate: float = 0.04
    regen_rate: float = 0.01
    v_scale: float = 5.2
    alpha_diffusion: float = 0.06
    n_kernels: int = 9
    kernel_radii_range: Tuple[float, float] = (6.0, 15.0)

@dataclass
class BiologyConfig:
    n_patches: int = 6
    mu_range: Tuple[float, float] = (0.13, 0.22)
    sigma_range: Tuple[float, float] = (0.011, 0.024)
    mixing_rule: str = "gene_wise" # "gene_wise", "negotiation", "constant"
    enable_mutation: bool = True
    mutation_interval: int = 50
    mutation_patch_radius: int = 10
    mutation_std: float = 0.01

@dataclass
class EnvironmentConfig:
    env_type: str = "open" # "open", "wall", "constriction"
    num_barriers: int = 2
    wall_thickness: int = 8
    passage_width: int = 20
    enable_depletion: bool = False
    depletion_rate: float = 0.04
    regeneration_rate: float = 0.01

@dataclass
class SimulationConfig:
    name: str = "flow_lenia_run"
    grid_size: int = 256
    steps: int = 2000
    sample_interval: int = 50
    seed: int = 42
    trials: int = 50
    bootstrap: int = 10
    output_dir: str = "results/experiment_run"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    biology: BiologyConfig = field(default_factory=BiologyConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    metrics_goal_space: List[str] = field(default_factory=lambda: ["com_displacement", "ea_raw", "complexity_raw"])

def load_config(yaml_path: Optional[str] = None) -> SimulationConfig:
    """
    Load a SimulationConfig from a YAML file, filling missing keys with defaults.
    """
    if yaml_path is None or not os.path.exists(yaml_path):
        return SimulationConfig()
        
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
        
    exp_section = raw.get("experiment", {})
    sim_section = raw.get("simulation", {})
    phys_section = raw.get("physics", {})
    bio_section = raw.get("biology", {})
    env_section = raw.get("environment", {})
    env_params = raw.get("environment_params", {})
    metrics_section = raw.get("metrics", {})
    
    physics_cfg = PhysicsConfig(
        dt=float(phys_section.get("dt", sim_section.get("dt", 0.05))),
        beta=float(phys_section.get("beta", 2.0)),
        depletion_rate=float(phys_section.get("depletion_rate", env_params.get("depletion_rate", 0.04))),
        regen_rate=float(phys_section.get("regen_rate", env_params.get("regeneration_rate", 0.01))),
        v_scale=float(phys_section.get("v_scale", 5.2)),
        alpha_diffusion=float(phys_section.get("alpha_diffusion", 0.06)),
        n_kernels=int(phys_section.get("n_kernels", 9)),
        kernel_radii_range=tuple(phys_section.get("kernel_radii_range", (6.0, 15.0)))
    )
    
    biology_cfg = BiologyConfig(
        n_patches=int(bio_section.get("n_patches", 6)),
        mu_range=tuple(bio_section.get("mu_range", (0.13, 0.22))),
        sigma_range=tuple(bio_section.get("sigma_range", (0.011, 0.024))),
        mixing_rule=str(bio_section.get("mixing_rule", "gene_wise")),
        enable_mutation=bool(bio_section.get("enable_mutation", True)),
        mutation_interval=int(bio_section.get("mutation_interval", 50)),
        mutation_patch_radius=int(bio_section.get("mutation_patch_radius", 10)),
        mutation_std=float(bio_section.get("mutation_std", 0.01))
    )
    
    env_cfg = EnvironmentConfig(
        env_type=str(exp_section.get("env", sim_section.get("environment", "open"))),
        num_barriers=int(env_section.get("num_barriers", 2)),
        wall_thickness=int(env_section.get("wall_thickness", 8)),
        passage_width=int(env_section.get("passage_width", 20)),
        enable_depletion=bool(env_params.get("enable_depletion", False)),
        depletion_rate=float(env_params.get("depletion_rate", 0.04)),
        regeneration_rate=float(env_params.get("regeneration_rate", 0.01))
    )
    
    sim_cfg = SimulationConfig(
        name=str(exp_section.get("name", "flow_lenia_run")),
        grid_size=int(sim_section.get("grid_size", 256)),
        steps=int(sim_section.get("steps", 2000)),
        sample_interval=int(sim_section.get("sample_interval", 50)),
        seed=int(exp_section.get("seed", 42)),
        trials=int(exp_section.get("trials", 50)),
        bootstrap=int(exp_section.get("bootstrap", 10)),
        output_dir=str(exp_section.get("output_dir", f"results/{exp_section.get('name', 'experiment_run')}")),
        physics=physics_cfg,
        biology=biology_cfg,
        environment=env_cfg,
        metrics_goal_space=metrics_section.get("goal_space", ["com_displacement", "ea_raw", "complexity_raw"])
    )
    
    return sim_cfg

def save_config(config: SimulationConfig, yaml_path: str):
    """
    Save SimulationConfig dataclass to YAML file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
    raw_dict = asdict(config)
    with open(yaml_path, "w") as f:
        yaml.dump(raw_dict, f, default_flow_style=False, sort_keys=False)
