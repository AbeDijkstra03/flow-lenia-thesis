#!/usr/bin/env python3
"""
Flow-Lenia Master's Thesis Unified Experiment Runner & CLI Orchestrator.

Usage:
  # 1. Run Open IMGEP Curiosity Search vs. Random Baseline
  uv run python run_experiment.py --mode imgep --trials 50 --steps 2000

  # 2. Run Static Barrier Obstacle & Corridor Navigation Experiment
  uv run python run_experiment.py --mode wall_obstacle --trials 50 --steps 2000

  # 3. Run Novel Passage Constriction Sweep Experiment (Thesis Focus)
  uv run python run_experiment.py --mode barrier_constriction --widths 8 16 24 32 --steps 2000

  # 4. Run Scaled-Up High-Resolution FPS Reruns (512x512, 10k steps)
  uv run python run_experiment.py --mode scaleup --scale_grid_size 512 --scale_steps 10000

  # 5. Run Long Multi-Species Ecosystem Hero Simulation
  uv run python run_experiment.py --mode hero --grid_size 384 --steps 4000

  # 6. Run Physical Method Comparison Showcase
  uv run python run_experiment.py --mode showcase

  # 7. Run Orbium Glider Physics Verification
  uv run python run_experiment.py --mode orbium

  # 8. Load from YAML Configuration File
  uv run python run_experiment.py --config configs/baseline.yaml
"""

import os
import sys
import argparse
from typing import Optional

from core.config import load_config, SimulationConfig
from experiments.run_imgep_search import main as run_imgep_main
from experiments.run_scaleup import main as run_scaleup_main
from experiments.run_barrier_constriction import main as run_constriction_main
from experiments.run_resource_depletion import main as run_depletion_main
from scripts.run_hero_ecosystem import main as run_hero_main
from scripts.run_showcase_methods import main as run_showcase_main
from scripts.spawn_orbium import main as run_orbium_main

def main():
    parser = argparse.ArgumentParser(
        description="Flow-Lenia Master's Thesis Simulation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["imgep", "wall_obstacle", "barrier_constriction", "depletion", "scaleup", "hero", "showcase", "orbium"],
        default="imgep",
        help="Experiment execution mode (default: imgep)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. configs/baseline.yaml)")
    parser.add_argument("--trials", type=int, default=50, help="Trial budget for search")
    parser.add_argument("--bootstrap", type=int, default=10, help="Bootstrap phase random trials")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size resolution")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation horizon steps")
    parser.add_argument("--sample_interval", type=int, default=50, help="Frame sampling interval")
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16, 24, 32], help="Passage widths for constriction sweep")
    parser.add_argument("--patches", type=int, default=6, help="Number of species patches for hero run")
    parser.add_argument("--scale_grid_size", type=int, default=512, help="Grid size for scale-up reruns")
    parser.add_argument("--scale_steps", type=int, default=10000, help="Steps for scale-up reruns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    
    args, unknown = parser.parse_known_args()
    
    # Load config if specified
    if args.config is not None:
        cfg = load_config(args.config)
        args.grid_size = cfg.grid_size
        args.steps = cfg.steps
        args.sample_interval = cfg.sample_interval
        args.seed = cfg.seed
        args.trials = cfg.trials
        args.bootstrap = cfg.bootstrap
        if args.output_dir is None:
            args.output_dir = cfg.output_dir
            
    if args.mode == "imgep":
        out_dir = args.output_dir or "results/baseline_imgep"
        sys.argv = [
            sys.argv[0],
            "--trials", str(args.trials),
            "--bootstrap", str(args.bootstrap),
            "--grid_size", str(args.grid_size),
            "--steps", str(args.steps),
            "--sample_interval", str(args.sample_interval),
            "--env", "open",
            "--seed", str(args.seed),
            "--output_dir", out_dir
        ]
        run_imgep_main()
        
    elif args.mode == "wall_obstacle":
        out_dir = args.output_dir or "results/wall_obstacles"
        sys.argv = [
            sys.argv[0],
            "--trials", str(args.trials),
            "--bootstrap", str(args.bootstrap),
            "--grid_size", str(args.grid_size),
            "--steps", str(args.steps),
            "--sample_interval", str(args.sample_interval),
            "--env", "wall",
            "--seed", str(args.seed),
            "--output_dir", out_dir
        ]
        run_imgep_main()
        
    elif args.mode == "barrier_constriction":
        out_dir = args.output_dir or "results/barrier_constriction"
        width_strs = [str(w) for w in args.widths]
        sys.argv = [
            sys.argv[0],
            "--widths"
        ] + width_strs + [
            "--grid_size", str(args.grid_size),
            "--steps", str(args.steps),
            "--sample_interval", str(max(10, args.sample_interval // 2)),
            "--seed", str(args.seed),
            "--output_dir", out_dir
        ]
        run_constriction_main()
        
    elif args.mode == "depletion":
        out_dir = args.output_dir or "results/resource_depletion"
        sys.argv = [
            sys.argv[0],
            "--grid_size", str(args.grid_size),
            "--steps", str(args.steps),
            "--sample_interval", str(max(10, args.sample_interval // 2)),
            "--seed", str(args.seed),
            "--output_dir", out_dir
        ]
        run_depletion_main()
        
    elif args.mode == "scaleup":
        out_dir = args.output_dir or "results/scaleup"
        sys.argv = [
            sys.argv[0],
            "--k_reruns", "3",
            "--scale_grid_size", str(args.scale_grid_size),
            "--scale_steps", str(args.scale_steps),
            "--sample_interval", str(args.sample_interval * 5),
            "--seed", str(args.seed),
            "--output_dir", out_dir
        ]
        run_scaleup_main()
        
    elif args.mode == "hero":
        out_dir = args.output_dir or "results/hero_ecosystems"
        sys.argv = [
            sys.argv[0],
            "--patches", str(args.patches),
            "--grid_size", str(args.grid_size),
            "--steps", str(args.steps),
            "--sample_interval", str(max(10, args.sample_interval // 2)),
            "--seed", str(args.seed),
            "--output_dir", out_dir
        ]
        run_hero_main()
        
    elif args.mode == "showcase":
        sys.argv = [sys.argv[0]]
        run_showcase_main()
        
    elif args.mode == "orbium":
        sys.argv = [sys.argv[0]]
        run_orbium_main()

if __name__ == "__main__":
    main()
