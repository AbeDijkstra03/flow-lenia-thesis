#!/usr/bin/env python3
"""
Flow-Lenia Bachelor's Thesis Unified Experiment Runner & CLI Orchestrator.

Usage:
  # Chapter 1: Foundational Physics & Mass Conservation
  uv run python run_experiment.py --mode orbium

  # Chapter 2: Micro-Mechanisms & Collision Ablations
  uv run python run_experiment.py --mode gene_mutation --seeds 42 101 2024
  uv run python run_experiment.py --mode negotiation --seeds 42 101 2024 --beta 3.0

  # Chapter 3: Curiosity-Driven Open-Ended Evolution (IMGEP vs Random Search)
  uv run python run_experiment.py --mode imgep --trials 40 --seeds 42 101 2024
  uv run python run_experiment.py --mode agentic_loop --generations 5

  # Chapter 4: Environmental Heterogeneity & Niche Construction
  uv run python run_experiment.py --mode wall_obstacle --trials 40 --seeds 42 101 2024
  uv run python run_experiment.py --mode barrier_constriction --widths 8 16 24 32 48 64 --seeds 42 101 2024
  uv run python run_experiment.py --mode depletion --grid_size 256 --seeds 42 101 2024

  # Chapter 5: Macro-Scale Ecology & Scale-Up
  uv run python run_experiment.py --mode scaleup --scale_grid_size 512 --seeds 42 101 2024
  uv run python run_experiment.py --mode epic --grid_size 384 --steps 22500 --patches 8 --seeds 42 101 2024

  # Load from YAML Configuration File
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
from experiments.run_gene_mutation import main as run_gene_mutation_main
from experiments.run_negotiation_rule import main as run_negotiation_main
from experiments.run_epic_ecosystem import main as run_epic_main
from experiments.run_physics_verification import main as run_physics_main
from experiments.run_autonomous_agentic_loop import main as run_agentic_main
from experiments.run_chemotaxis_calibration import main as run_chemotaxis_calib_main

def main():
    parser = argparse.ArgumentParser(
        description="Flow-Lenia Bachelor's Thesis Simulation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode", type=str,
        choices=[
            "imgep", "wall_obstacle", "chemotaxis_calibration", "barrier_constriction", "depletion",
            "gene_mutation", "negotiation", "agentic_loop", "epic",
            "scaleup", "orbium", "showcase"
        ],
        default="imgep",
        help="Experiment execution mode (default: imgep)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--trials", type=int, default=40, help="Trial budget for search")
    parser.add_argument("--bootstrap", type=int, default=10, help="Bootstrap phase random trials")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size resolution")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation horizon steps")
    parser.add_argument("--sample_interval", type=int, default=50, help="Frame sampling interval")
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64], help="Passage widths for constriction sweep")
    parser.add_argument("--patches", type=int, default=6, help="Number of species patches")
    parser.add_argument("--beta", type=float, default=3.0, help="Softmax negotiation beta")
    parser.add_argument("--scale_grid_size", type=int, default=512, help="Grid size for scale-up reruns")
    parser.add_argument("--scale_steps", type=int, default=3600, help="Steps for scale-up reruns")
    parser.add_argument("--generations", type=int, default=5, help="Generations for agentic loop")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    
    args, unknown = parser.parse_known_args()
    
    # Load config if specified
    if args.config is not None:
        cfg = load_config(args.config)
        args.grid_size = cfg.grid_size
        args.steps = cfg.steps
        args.sample_interval = cfg.sample_interval
        args.seeds = [cfg.seed]
        args.trials = cfg.trials
        args.bootstrap = cfg.bootstrap
        if args.output_dir is None:
            args.output_dir = cfg.output_dir
            
    seed_strs = [str(s) for s in args.seeds]
    
    if args.mode == "imgep":
        out_dir = args.output_dir or "results/baseline_imgep"
        sys.argv = [
            sys.argv[0],
            "--trials", str(args.trials),
            "--bootstrap", str(args.bootstrap),
            "--grid_size", str(args.grid_size),
            "--steps", str(args.steps),
            "--sample_interval", str(args.sample_interval),
            "--elite_steps", "3600",
            "--elite_sample_interval", "3",
            "--env", "open",
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
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
            "--elite_steps", "3600",
            "--elite_sample_interval", "3",
            "--env", "wall",
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_imgep_main()
        
    elif args.mode == "chemotaxis_calibration":
        out_dir = args.output_dir or "results/chemotaxis_calibration"
        sys.argv = [
            sys.argv[0],
            "--grid_size", str(args.grid_size),
            "--steps", "3600",
            "--sample_interval", "3",
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_chemotaxis_calib_main()
        
    elif args.mode == "barrier_constriction":
        out_dir = args.output_dir or "results/barrier_constriction"
        width_strs = [str(w) for w in args.widths]
        sys.argv = [
            sys.argv[0],
            "--widths"
        ] + width_strs + [
            "--grid_size", str(args.grid_size),
            "--steps", "3600",
            "--sample_interval", "3",
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_constriction_main()
        
    elif args.mode == "depletion":
        out_dir = args.output_dir or "results/resource_depletion"
        sys.argv = [
            sys.argv[0],
            "--grid_size", str(args.grid_size),
            "--steps", "3600",
            "--sample_interval", "3",
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_depletion_main()
        
    elif args.mode == "gene_mutation":
        out_dir = args.output_dir or "results/gene_mutation"
        sys.argv = [
            sys.argv[0],
            "--grid_size", str(args.grid_size if args.grid_size >= 256 else 384),
            "--steps", "3600",
            "--sample_interval", "3",
            "--patches", str(args.patches),
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_gene_mutation_main()
        
    elif args.mode == "negotiation":
        out_dir = args.output_dir or "results/negotiation_rule"
        sys.argv = [
            sys.argv[0],
            "--grid_size", str(args.grid_size if args.grid_size >= 256 else 384),
            "--steps", "3600",
            "--sample_interval", "3",
            "--patches", str(args.patches),
            "--beta", str(args.beta),
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_negotiation_main()
        
    elif args.mode == "agentic_loop":
        out_dir = args.output_dir or "results/agentic_loop"
        sys.argv = [
            sys.argv[0],
            "--generations", str(args.generations),
            "--trials_per_gen", "25",
            "--grid_size", "384",
            "--steps", "3000",
            "--sample_interval", "3",
            "--output_dir", out_dir
        ]
        run_agentic_main()
        
    elif args.mode == "epic":
        out_dir = args.output_dir or "results/epic_ecosystem"
        sys.argv = [
            sys.argv[0],
            "--grid_size", str(args.grid_size if args.grid_size >= 256 else 384),
            "--steps", str(args.steps if args.steps >= 10000 else 22500),
            "--sample_interval", "3",
            "--patches", str(args.patches if args.patches >= 8 else 8),
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_epic_main()
        
    elif args.mode == "scaleup":
        out_dir = args.output_dir or "results/scaleup"
        sys.argv = [
            sys.argv[0],
            "--k_reruns", "3",
            "--scale_grid_size", str(args.scale_grid_size if args.scale_grid_size >= 384 else 512),
            "--scale_steps", "3600",
            "--sample_interval", "3",
            "--seeds"
        ] + seed_strs + ["--output_dir", out_dir]
        run_scaleup_main()
        
    elif args.mode == "orbium":
        out_dir = args.output_dir or "results/orbium"
        sys.argv = [
            sys.argv[0],
            "--grid_size", str(args.grid_size),
            "--steps", str(args.steps),
            "--output_dir", out_dir
        ]
        run_physics_main()
        
    elif args.mode == "showcase":
        print("Note: Mode 'showcase' is legacy. Executing both 'gene_mutation' and 'negotiation' sequentially.")
        sys.argv = [sys.argv[0], "--output_dir", "results/gene_mutation", "--seeds"] + seed_strs
        run_gene_mutation_main()
        sys.argv = [sys.argv[0], "--output_dir", "results/negotiation_rule", "--seeds"] + seed_strs
        run_negotiation_main()

if __name__ == "__main__":
    main()
