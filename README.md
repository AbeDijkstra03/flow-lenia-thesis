# GPU-Accelerated Flow-Lenia Open-Ended Evolution Framework (JAX)

A high-performance, GPU-accelerated simulation and open-ended discovery framework for **Flow-Lenia** continuous cellular automata in native JAX, built for NVIDIA Blackwell (RTX 5090) hardware.

Adheres strictly to the canonical literature:
- **Michel, Cvjetko, Hamon, Oudeyer, Moulin-Frier (2025/2026)**: *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* (arXiv:2505.15998).
- **Plantec et al. (2025)**: *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, Artificial Life 31(2) (arXiv:2506.08569).

---

## 1. Quickstart & Installation

### Option A: Using `uv` (Recommended, Fast)
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/flow-lenia.git
cd flow-lenia

# 2. Run automated test suite (automatically creates virtualenv and installs dependencies)
uv run python -m unittest discover tests -v

# 3. Verify JAX GPU acceleration
uv run python -c "import jax; print('JAX Devices:', jax.devices())"
```

### Option B: Using Standard `pip`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m unittest discover tests -v
```

### Option C: Using Docker (Zero-Setup Container)
```bash
# 1. Build Docker image
docker build -t flow-lenia:latest .

# 2. Run unit tests inside container
docker run --rm flow-lenia:latest

# 3. Run any experiment with output mounted to host machine:
docker run --rm -v $(pwd)/results:/app/results flow-lenia:latest python run_experiment.py --mode barrier_constriction --widths 8 16 24 32
```

---

## 2. Directory & Documentation Structure

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
│   ├── run_imgep_search.py         # Open IMGEP vs. Random Search benchmark (Exp 1)
│   ├── run_barrier_constriction.py # Passage width sweep & transmission curves (Exp 2)
│   ├── run_resource_depletion.py   # Dynamic niche depletion & foraging study (Exp 3)
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
│   ├── THESIS_ARCHITECTURE.md      # Exhaustive mathematical specification & formulas
│   ├── LITERATURE_READING_LIST.md  # Core papers & literature review
│   └── REFACTORING_HISTORY.md      # PyTorch-to-JAX refactoring log
├── .agents/                        # AI Assistant rules, memory, and skills
│   ├── AGENTS.md                   # Core behavioral rules & physics constraints
│   ├── CONTEXT_HANDOFF.md          # Active cross-session AI agent state memory
│   └── skills/                     # Specialized agent workflow skills
├── results/                        # Categorized simulation outputs with subfolders per run
│   ├── baseline_imgep/             # IMGEP open search (elite_1/, elite_2/, elite_3/)
│   ├── wall_obstacles/             # IMGEP barrier search (elite_1/, elite_2/, elite_3/)
│   ├── barrier_constriction/       # Constriction sweep (width_08/, width_16/, width_24/, width_32/)
│   ├── epic_ecosystem/             # 5-minute Master Epic Ecosystem broadcast video & filmstrip
│   ├── orbium/                     # Classic Orbium unicaudatus glider verification
│   ├── resource_depletion/         # static_baseline/ vs dynamic_depletion/
│   ├── scaleup/                    # 512x512 scaled-up reruns (rerun_1/, rerun_2/)
│   ├── showcase/                   # method_1_gene_mutation/ & method_2_negotiation_rule/
│   └── agentic_loop/               # Multi-generation AI Scientist discovery state
├── tests/                          # Automated unit test suite
├── Dockerfile                      # Container environment definition
├── run_experiment.py               # Main unified CLI runner
└── pyproject.toml                  # Package configuration & dependencies
```

---

## 3. How to Run the Experiments

### Available Experiment Commands

| Mode | Command | Output Duration | What Happens? |
| :--- | :--- | :--- | :--- |
| **Baseline IMGEP** | `uv run python run_experiment.py --mode imgep --trials 40` | **3x 1 min** (60s each) | Runs 40 IMGEP goal exploration trials vs Random Search, saving top 3 elites into `elite_1/`, `elite_2/`, `elite_3/`. |
| **Wall Obstacles IMGEP** | `uv run python run_experiment.py --mode wall_obstacle --trials 40` | **3x 1 min** (60s each) | Runs IMGEP search in arena with DodgerBlue barriers and narrow corridor slits. |
| **Barrier Constriction** | `uv run python run_experiment.py --mode barrier_constriction` | **1 min per width** (4x 60s) | Sweeps corridor widths $W \in [8, 16, 24, 32]$ px (`width_08/` .. `width_32/`), measuring soliton deformation & transmission $T(W)$. |
| **Resource Depletion** | `uv run python run_experiment.py --mode depletion` | **1 min** (2x 60s) | Compares `static_baseline/` vs. `dynamic_depletion/` foraging trails. |
| **Showcase Methods** | `uv run python run_experiment.py --mode showcase` | **1 min each** (2x 60s) | Demonstrates `method_1_gene_mutation/` and `method_2_negotiation_rule/`. |
| **Scaled-Up Reruns** | `uv run python run_experiment.py --mode scaleup` | **1 min each** (2x 60s) | Re-evaluates top IMGEP candidates on massive $512 \times 512$ grids (`rerun_1/`, `rerun_2/`). |
| **Orbium Verification** | `uv run python run_experiment.py --mode orbium` | **1 min** (60s) | Verifies physics engine on classic *Orbium unicaudatus* glider in `results/orbium/`. |
| **Epic Ecosystem Master** | `uv run python run_experiment.py --mode epic --steps 22500` | **5.0 minutes** (300s) | Master broadcast simulation (22,500 steps, 7,500 frames, 56.6 MB) with 8 species interacting around 4 DodgerBlue arena obstacle pillars. |

---

## 4. How to Interpret Simulation Output Files

Each experiment exports standardized research artifacts into `results/<experiment_name>/`:

| Output File | Format | Scientific Meaning & How to Interpret |
| :--- | :--- | :--- |
| `*_rollout.mp4` | Video (H.264, 20 fps) | **Broadcast Dual-Panel Video**: Left panel displays multi-species categorical palette or soft-log Plasma contrast ($\log(1+A)$) revealing wave dynamics. Right panel displays absolute physical density $[0.0, 1.0]$ in grayscale with barrier walls (DodgerBlue) and center-of-mass trajectory tracking. |
| `*_trajectory_filmstrip.png` | Composite PNG | **Thesis Subfigure**: 6 evenly spaced time slices ($t=0\%, 20\%, 40\%, 60\%, 80\%, 100\%$) with timestamp headers, formatted for direct LaTeX inclusion (`\includegraphics{...}`). |
| `*_motion_heatmap.png` | Static Heatmap PNG | **Locomotion Trail**: Cumulative spatial displacement ($\sum_t \|\Delta A_t\|$) in Magma colormap, visualizing the full trajectory on printed/static pages. |
| `*_metadata.json` / `summary.json` | JSON | **Quantitative Metric Log**: Contains random seed, hyperparameters ($\mu, \sigma, \mathbf{w}, v_{\text{scale}}, \alpha$), and computed 3D metrics ($v_{\text{CoM}}$, $\text{EA}$, complexity, watertight scores). |
| `*_data.npz` | Compressed NumPy | **Raw Numerical State**: Binary arrays (`sampled_mass`, `sampled_gid`) allowing post-hoc analysis or Jupyter visualization without re-running physics. |

---

## 5. How to Run Autonomous AI Scientist Discovery Campaigns

To run continuous multi-generation exploration loops where an AI model autonomously generates candidates, inspects trajectories with computer vision, filters invalid mutants, and tunes parameter bounds:

```bash
uv run python experiments/run_autonomous_agentic_loop.py --generations 10 --trials_per_gen 25 --steps 3000 --output_dir results/agentic_loop
```

- Discovered lineages and metrics are logged to `results/agentic_loop/agentic_loop_state.json`.
- Dual-panel snapshot frames and motion heatmaps are saved in `results/agentic_loop/frames/`.
- Summary findings are automatically synchronized to [.agents/CONTEXT_HANDOFF.md](file:///home/abedijkstra/Documents/Scriptie/.agents/CONTEXT_HANDOFF.md).
