# GPU-Accelerated Flow-Lenia Open-Ended Evolution Framework (JAX)

A high-performance, GPU-accelerated simulation and discovery framework for **Flow-Lenia** continuous cellular automata in native JAX, built for NVIDIA Blackwell (RTX 5090) hardware.

Adheres strictly to the canonical literature:
- **Michel, Cvjetko, Hamon, Oudeyer, Moulin-Frier (2025/2026)**: *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* (arXiv:2505.15998).
- **Plantec et al. (2025)**: *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, Artificial Life 31(2) (arXiv:2506.08569).

---

## 1. Quickstart & Installation

Ensure you have Python 3.10+ and `uv` installed:

```bash
# Verify GPU availability
uv run python -c "import jax; print(jax.devices())"

# Run automated test suite
uv run python -m unittest discover tests
```

---

## 2. Directory Structure

```
.
├── core/                           # Physics engine, metrics, IMGEP, configs, visualization
│   ├── flow_lenia_jax.py           # JAX circular convolution & flux advection
│   ├── metrics.py                  # 3D metric suite (EA, Motility, Complexity, Solidity)
│   ├── imgep.py                    # Goal-directed IMGEP exploration & FPS archiving
│   ├── environment.py              # Geometric wall masks and passage corridors
│   ├── visualization.py            # H.264 MP4, trajectory filmstrip, & heatmap exporter
│   └── config.py                   # Typed dataclass YAML configuration engine
├── experiments/                    # Structured thesis experiments
│   ├── run_imgep_search.py         # Open IMGEP vs. Random Search benchmark
│   ├── run_barrier_constriction.py # Passage width sweep & transmission efficiency
│   ├── run_scaleup.py              # 512x512 scaled-up FPS reruns (10k steps)
│   └── run_autonomous_agentic_loop.py # Multi-generation AI Scientist discovery loop
├── scripts/                        # Standalone utilities & showcases
│   ├── spawn_orbium.py             # Classic Orbium glider physics verification
│   ├── run_hero_ecosystem.py       # Long multi-blob ecosystem simulation (4000 steps)
│   └── run_showcase_methods.py     # Comparison of 3 physical mechanisms
├── configs/                        # Standardized YAML configuration files
│   ├── baseline.yaml               # Open IMGEP curiosity search
│   ├── wall_obstacle.yaml          # Static barrier wall navigation
│   ├── barrier_constriction.yaml   # Corridor constriction parameter sweep
│   ├── resource_depletion.yaml     # Dynamic foraging & depletion wake
│   └── scaleup.yaml                # 512x512 FPS scale-up rerun
├── docs/                           # Academic thesis documentation
│   ├── LITERATURE_READING_LIST.md  # Core papers & literature review
│   ├── THESIS_ARCHITECTURE.md      # Detailed mathematical specification
│   └── REFACTORING_HISTORY.md      # PyTorch-to-JAX refactoring log
├── .agents/                        # AI Assistant rules, memory, and skills
│   ├── AGENTS.md                   # Core behavioral rules & physics constraints
│   ├── CONTEXT_HANDOFF.md          # Active cross-session AI agent state memory
│   └── skills/                     # Specialized agent workflow skills
├── results/                        # Categorized simulation outputs (MP4s, plots, NPZ)
├── tests/                          # Automated unit test suite
├── Dockerfile                      # Container environment definition
├── run_experiment.py               # Main unified CLI runner
└── pyproject.toml                  # Package configuration & dependencies
```

---

## 3. How to Run Experiments

All experiments can be launched from the main CLI entry point:

| Experiment / Objective | CLI Command | Key Outputs |
| :--- | :--- | :--- |
| **Open IMGEP vs. Random Baseline** | `uv run python run_experiment.py --mode imgep --trials 50 --steps 2000` | `summary.json`, elite MP4s, filmstrips in `results/baseline_imgep/` |
| **Wall Obstacle Navigation** | `uv run python run_experiment.py --mode wall_obstacle --trials 50 --steps 2000` | Navigation MP4s & metrics in `results/wall_obstacles/` |
| **Corridor Constriction Sweep** | `uv run python run_experiment.py --mode barrier_constriction --widths 8 16 24 32` | `transmission_curve.png`, MP4s in `results/barrier_constriction/` |
| **512x512 Scaled-Up Reruns** | `uv run python run_experiment.py --mode scaleup --scale_steps 10000` | `scaleup_report.json`, MP4s in `results/scaleup/` |
| **Long Multi-Patch Ecosystem** | `uv run python run_experiment.py --mode hero --grid_size 384 --steps 4000` | Broadcast MP4 & filmstrip in `results/hero_ecosystems/` |
| **Physical Mechanism Showcase** | `uv run python run_experiment.py --mode showcase` | 3 comparison MP4s in `results/showcase/` |
| **Orbium Glider Test** | `uv run python run_experiment.py --mode orbium` | Verification MP4 in `results/spawned_orbium.mp4` |
| **Run from YAML Config** | `uv run python run_experiment.py --config configs/baseline.yaml` | Reproducible run matching config |

---

## 4. Output Artifact Standards

Simulations automatically export SOTA research artifacts conforming to ALife publishing standards:
- **Broadcast MP4 Video**: Dual-panel side-by-side video (`libx264`, `yuv420p`, CRF 18):
  - *Left Panel*: Perceptually uniform Plasma colormap with soft log contrast $\log(1 + A)$.
  - *Right Panel*: Absolute physical mass scale $[0.0, 1.0]$ with obstacle boundaries and CoM trajectory.
- **Trajectory Filmstrips**: 6-frame horizontal composite PNGs ($t=0\%, 20\%, 40\%, 60\%, 80\%, 100\%$) formatted with timestamp labels for direct inclusion into LaTeX / thesis subfigures.
- **Motion Heatmaps**: Normalized cumulative spatial state transitions ($\sum_t |\Delta A_t|$).
- **NPZ Time Series**: Full numerical state arrays for exact replay and post-hoc analysis.
- **JSON Metadata**: Hyperparameters, seeds, and 3D metric vectors (`com_displacement`, `ea_raw`, `complexity_raw`, `watertight_score`).

---

## 5. Mathematical Canon

1. **Circular Convolutions**:
   $$ U_k = \mathcal{F}^{-1}\left(\mathcal{F}(A) \cdot \widehat{K}_k\right) $$
   using multi-shell concentric rings ($b_{\text{shells}} = [1.0, 0.5, 0.33]$, $r_{\text{peaks}} = [0.5, 0.25, 0.75]$).

2. **Continuous Growth Mapping**:
   $$ G_k(U) = 2 \exp\left(-\frac{(U - \mu_k)^2}{2\sigma_k^2}\right) - 1 $$

3. **Flux-Conserved Velocity Advection**:
   $$ \mathbf{v} = v_{\text{scale}} \cdot \left((1 - \alpha)\nabla G(U) - \alpha \nabla A\right) $$
   Normalized by $\min(1, 1 / \|\mathbf{v}\|_1)$ ensuring **100.000% machine-precision mass conservation**.
