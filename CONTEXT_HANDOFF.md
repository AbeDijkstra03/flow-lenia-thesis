# Context Handoff & Memory Persistence: Flow-Lenia Discovery Engine

This document maintains state persistence across conversation turns and model sessions for the GPU-accelerated Flow-Lenia Master's Thesis research framework.

---

## 1. Executive Summary & Physics Canon

- **Canonical Literature**:
  - Full reading list preserved in [docs/LITERATURE_READING_LIST.md](file:///home/abedijkstra/Documents/Scriptie/docs/LITERATURE_READING_LIST.md).
  - Primary references: Michel et al. (2025/2026, arXiv:2505.15998), Plantec et al. (2025, arXiv:2506.08569).
- **Core Physics Formulation**:
  $$\mathbf{F} = v_{\text{scale}} \cdot \left((1 - \alpha) \nabla G(U) - \alpha \nabla A\right)$$
  - Mass conservation: Outgoing fluxes normalized by `1.0 / max(1.0, v_sum)`. Step-0 mass total conserved to $100.000\%$.
  - Kernels: Multi-shell concentric rings ($b = [1.0, 0.5, 0.33]$, $r = [0.5, 0.25, 0.75]$, $r_w = 0.12$).
  - Spatial Derivatives: Sobel gradient operator with positive coordinate indexing ($x+1 \to \text{roll}(-1, 1)$, $y+1 \to \text{roll}(-1, 0)$).
  - Mixing Rules: Stochastic Gene-Wise Sampling & Growth Negotiation Rule ($\text{softmax}(\beta G)$).

---

## 2. Modernized SOTA Visualization & Video Engine (`core/visualization.py`)

- **Video Format**: H.264 MP4 (`libx264`, `yuv420p`, CRF 18) via `imageio-ffmpeg`.
- **Dual-Panel Scientific Visualizer**:
  - *Left Panel*: Categorical multi-species colormap (Cyan, Magenta, Lime, Amber, Purple, Coral) or soft log Plasma colormap.
  - *Right Panel*: Absolute physical density $[0.0, 1.0]$ in grayscale with barrier wall overlays (DodgerBlue) and active center-of-mass trajectory tracking.
- **6-Frame Trajectory Filmstrips**: Publication-ready composite PNGs ($t=0\%, 20\%, 40\%, 60\%, 80\%, 100\%$) with timestamp headers for direct inclusion into LaTeX / thesis documents.
- **Motion Heatmaps**: Normalized spatial state transition heatmaps ($\sum_t |\Delta A_t|$).

---

## 3. Directory Layout & Key Executables

- `run_experiment.py`: Main unified root CLI runner.
- `core/`: Physics (`flow_lenia_jax.py`), Metrics (`metrics.py`), IMGEP (`imgep.py`), Config (`config.py`), Visualization (`visualization.py`), Environment (`environment.py`).
- `experiments/`:
  - `run_imgep_search.py`: Open IMGEP curiosity search vs. Random baseline.
  - `run_barrier_constriction.py`: Passage width sweep & transmission efficiency experiment.
  - `run_scaleup.py`: 512x512 scaled-up FPS reruns (10k steps).
  - `run_autonomous_agentic_loop.py`: AI Scientist autonomous discovery loop.
- `scripts/`: `spawn_orbium.py`, `run_hero_ecosystem.py`, `run_showcase_methods.py`.
- `configs/`: Typed YAML configurations (`baseline.yaml`, `wall_obstacle.yaml`, `barrier_constriction.yaml`, `resource_depletion.yaml`, `scaleup.yaml`).
- `docs/`: `LITERATURE_READING_LIST.md`, `THESIS_ARCHITECTURE.md`, `REFACTORING_HISTORY.md`.
- `results/`: `baseline_imgep/`, `wall_obstacles/`, `barrier_constriction/`, `hero_ecosystems/`, `showcase/`, `agentic_loop/`.

---

## 4. Watertight Quality Evaluation Loop (`core/metrics.py`)

All evaluations MUST pass `evaluate_watertight_quality_score()`:
- **Mass Preservation Ratio**: $R_{\text{mass}} \in [0.60, 5.00]$ (Step-0 target mass ratio preserved).
- **Solid Core Density Ratio**: $R_{\text{core}} \ge 0.50$ (Core density $A \ge 0.15$). Disqualifies hollow outlines.
- **Net Motility**: $v_{\text{CoM}} \ge 5.0\text{ px}$. Disqualifies frozen still-lifes.
- **Spatial Bounding**: $C_{\text{grid}} \le 0.25$. Disqualifies unconstrained grid chaos.

---

## 5. Autonomous Discovery Loop Harness

To run continuous multi-hour research campaigns:

```bash
uv run python experiments/run_autonomous_agentic_loop.py --generations 10 --trials_per_gen 25 --steps 3000 --output_dir results/agentic_loop
```

- State is persisted automatically to `results/agentic_loop/agentic_loop_state.json`.
- Trajectory frames are extracted as dual-panel PNGs (`gen_X_top1_frame_step_1_pct0.png`, `gen_X_top1_motion_heatmap.png`) alongside `gen_X_top1_rollout.mp4` to `results/agentic_loop/frames/`.
- AI models inspect trajectory PNGs via `view_file` to evaluate glider motility and collision dynamics.
