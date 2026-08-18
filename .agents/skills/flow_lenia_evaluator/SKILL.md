---
name: flow_lenia_evaluator
description: Instructions and workflow for running JAX Flow-Lenia physics, IMGEP search experiments, 6-frame trajectory analysis, and visual motion inspection.
---

# Flow-Lenia Evaluation & Inspection Skill

This skill provides step-by-step instructions for running, evaluating, and visually verifying JAX Flow-Lenia continuous cellular automata simulations.

---

## 1. Execution Commands

### A. Run Open-Environment IMGEP Search Benchmark
```bash
uv run python run_experiment.py --mode imgep --trials 50 --bootstrap 10 --grid_size 256 --steps 1000 --output_dir results/imgep_search
```

### B. Run Environment-Change Wall Obstacle Experiment
```bash
uv run python run_experiment.py --mode wall_obstacle --trials 50 --bootstrap 10 --grid_size 256 --steps 1000 --output_dir results/wall_obstacle
```

### C. Run Scaled-Up FPS Reruns (512x512, 10,000 steps)
```bash
uv run python run_experiment.py --mode scaleup --output_dir results/scaleup_reruns
```

### D. Run Self-Evaluating Hero Ecosystem Simulation
```bash
uv run python run_self_evaluating_hero.py --patches 6 --grid_size 384 --steps 4000 --output_file results/hero_ecosystem_self_evaluated.webp
```

---

## 2. Visual Trajectory Inspection Workflow

When evaluating whether a simulation rollout produced active locomotion or static still-lifes:

1. **Extract Trajectory PNGs**:
   Run `run_self_evaluating_hero.py` which automatically writes 6 trajectory PNG frames ($t=0\%, 20\%, 40\%, 60\%, 80\%, 100\%$) and `motion_heatmap.png` to `<output_file>_frames/`.

2. **Inspect via `view_file`**:
   Call `view_file` on `frame_step_1_pct0.png`, `frame_step_4_pct60.png`, `frame_step_6_pct100.png`, and `motion_heatmap.png`.

3. **Watertight Evaluation Criteria**:
   - **Mass Preservation Ratio ($R_{\text{mass}} \in [0.60, 1.80]$)**: Rejects rollouts that dissolve/evacuate or explode.
   - **Solid Core Density Ratio ($R_{\text{core}} \ge 0.50$)**: Rejects hollow outlines where mass interior drops below $0.15$.
   - **Net Motility ($v_{\text{CoM}} \ge 10.0\text{ px}$)**: Rejects frozen still-lifes.
   - **Absolute Physical Density Scale ($[0.0, 1.0]$)**: `frame_step_X_pctY.png` uses fixed $[0.0, 1.0]$ physical mapping so mass decay/fading is visually uncheatable.

