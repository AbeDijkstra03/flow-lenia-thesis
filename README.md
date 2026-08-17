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

## 2. Master's Thesis Experimental Architecture (The 5 Acts)

| Chapter | Experiment / Benchmark | Primary CLI Command | Key Result / Metric |
| :--- | :--- | :--- | :--- |
| **Act 1A** | Physics Verification (*Orbium*) | `uv run python run_experiment.py --mode orbium` | Exact mass conservation ($Q=0.00$) |
| **Act 1B** | Gumbel-Max Gene Mixing | `uv run python run_experiment.py --mode gene_mutation` | Prevents parameter blurring into gray averages |
| **Act 1C** | Softmax Growth Negotiation | `uv run python run_experiment.py --mode negotiation` | Sharp territorial competitive exclusion |
| **Act 2A** | IMGEP vs. Random Search | `uv run python run_experiment.py --mode imgep` | 3D Goal Exploration discovers $+280\%$ motility |
| **Act 2B** | Autonomous AI Scientist Loop | `uv run python run_experiment.py --mode agentic_loop` | 138 gens, 108 elites, +392% score gain |
| **Act 3** | Chemotaxis & Cohesion-Fission | `uv run python run_experiment.py --mode chemotaxis_calibration` | Unitary Droplet ($\Delta x = +120.7$) vs. Mitosis ($\Delta x = +144.4$) |
| **Act 4A** | Soft Barrier Constriction | `uv run python run_experiment.py --mode barrier_constriction` | Sigmoidal Transmission $T(W) \in [0.0\%, 77.0\%]$ |
| **Act 4B** | Wall Obstacle Navigation | `uv run python run_experiment.py --mode wall_obstacle` | IMGEP maze steering around geometric baffles |
| **Act 4C** | Dynamic Resource Depletion | `uv run python run_experiment.py --mode depletion` | Localized substrate exhaustion forces cyclic trails |
| **Act 5A** | $512 \times 512$ Scale-Up Invariance | `uv run python run_experiment.py --mode scaleup` | Scale-invariant macro dynamics |
| **Act 5B** | **Grand Synthesis Colosseum** | `uv run python run_experiment.py --mode epic` | 8 species, 4 chemotactic sanctuaries, 5-min HD videos |

---

## 3. Directory Structure

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
│   ├── run_chemotaxis_calibration.py # 3-way Cohesion vs Fission phase transition (Act 3)
│   ├── run_barrier_constriction.py # Passage width sweep & transmission curves (Act 4A)
│   ├── run_imgep_search.py         # Open IMGEP vs. Random Search benchmark (Act 2A)
│   ├── run_resource_depletion.py   # Dynamic niche depletion & foraging study (Act 4C)
│   ├── run_scaleup.py              # 512x512 scaled-up FPS reruns (Act 5A)
│   ├── run_epic_ecosystem.py       # Grand Synthesis Chemotactic Colosseum (Act 5B)
│   └── run_autonomous_agentic_loop.py # Multi-generation AI Scientist discovery loop
├── scripts/                        # Standalone utilities & showcases
│   ├── spawn_orbium.py             # Classic Orbium glider physics verification
│   └── run_hero_ecosystem.py       # Ecosystem simulation
├── docs/                           # Academic thesis documentation
│   ├── THESIS_ARCHITECTURE.md      # Exhaustive mathematical specification & formulas
│   ├── LITERATURE_READING_LIST.md  # Core papers & literature review
│   └── REFACTORING_HISTORY.md      # Engineering & scientific evolution log
├── .agents/                        # AI Assistant rules, memory, and skills
│   ├── AGENTS.md                   # Core behavioral rules & physics constraints
│   ├── CONTEXT_HANDOFF.md          # Active cross-session AI agent state memory
│   └── skills/                     # Specialized agent workflow skills
├── results/                        # Multi-seed simulation outputs (seed_42, seed_101, seed_2024)
│   ├── chemotaxis_calibration/     # 3-Way Cohesion vs Fission comparison
│   ├── barrier_constriction/       # Soft-bodied aperture transmission sweep
│   ├── epic_ecosystem/             # 5-minute Grand Synthesis broadcast MP4s & filmstrips
│   ├── baseline_imgep/             # IMGEP open search
│   ├── wall_obstacles/             # IMGEP barrier search
│   ├── resource_depletion/         # Static baseline vs dynamic foraging depletion
│   ├── gene_mutation/              # Multi-species Gumbel-Max mixing
│   ├── negotiation_rule/           # Softmax growth negotiation
│   ├── scaleup/                    # 512x512 scaled-up reruns
│   ├── orbium/                     # Classic Orbium glider verification
│   └── agentic_loop/               # Multi-generation AI Scientist discovery state
├── tests/                          # Automated unit test suite
├── Dockerfile                      # Container environment definition
├── run_experiment.py               # Main unified CLI runner
└── pyproject.toml                  # Package configuration & dependencies
```

---

## 4. Scientific Documentation

- **[THESIS_ARCHITECTURE.md](docs/THESIS_ARCHITECTURE.md)**: Exhaustive mathematical specification containing all continuous PDE equations, Fourier kernel formulations, 3D metric definitions, watertight filter rules, and parameter specification tables.
- **[REFACTORING_HISTORY.md](docs/REFACTORING_HISTORY.md)**: Complete changelog documenting the transition from legacy PyTorch to JAX, Blackwell RTX 5090 acceleration, Sobel gradient sign fixes, the Stochastic Seed Overfitting dilemma, and the discovery of the Chemotactic Cohesion-Fission Phase Transition.
