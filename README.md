# GPU-Accelerated Flow-Lenia Open-Ended Evolution Framework (JAX)

A high-performance, GPU-accelerated simulation and open-ended discovery framework for **Flow-Lenia** continuous cellular automata in native JAX.

Adheres strictly to the canonical literature:
- **Michel, Cvjetko, Hamon, Oudeyer, Moulin-Frier (2025/2026)**: *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* (arXiv:2505.15998).
- **Plantec et al. (2025)**: *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, Artificial Life 31(2) (arXiv:2506.08569).

---

## 1. Quickstart & Installation

### Option A: Using `uv` (Recommended, Fast)
```bash
# 1. Clone repository
git clone https://github.com/AbeDijkstra03/flow-lenia-thesis.git
cd flow-lenia-thesis

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

### Option C: Using Docker & GitHub Container Registry (GHCR)
```bash
# Option 1: Pull pre-built image directly from GitHub Packages (GHCR)
docker pull ghcr.io/abedijkstra03/flow-lenia-thesis:latest

# Option 2: Build Docker image locally
docker build -t flow-lenia:latest .

# Run test suite inside container:
docker run --rm ghcr.io/abedijkstra03/flow-lenia-thesis:latest

# Run experiment inside container (persisting artifacts to host):
docker run --rm -v $(pwd)/results:/app/results ghcr.io/abedijkstra03/flow-lenia-thesis:latest python run_experiment.py --mode orbium

# Interactive container shell:
docker run --rm -it -v $(pwd)/results:/app/results ghcr.io/abedijkstra03/flow-lenia-thesis:latest bash
```

---

## 2. Hardware Requirements & Configuration Guide

Because the framework is written in native **JAX**, all mathematical physics computations (2D Real FFT convolutions, finite-difference velocity updates, and `jax.lax.scan` time-steppers) are compiled via **XLA (Accelerated Linear Algebra)** into target-optimized machine code.

### Hardware Compatibility & Scaling

| Hardware Tier | Memory / VRAM | Status | Practical Scaling & Usage |
| :--- | :--- | :--- | :--- |
| **Workstation GPU ($\ge 24\text{ GB}$)** | $24–32\text{ GB}$ | **Supported** | Full throughput; supports large canvas ($512\times 512$) and wide parallel batches (`--trials 16-32`). Reference testing conducted on RTX 5090. |
| **Consumer GPU ($8–16\text{ GB}$)** | $8–16\text{ GB}$ | **Supported** | Runs out of the box with standard CUDA. For high-step rollouts, set `--trials` to 4–8 per batch to stay within available VRAM. |
| **HPC / Cloud Accelerators ($40–80\text{ GB}$)** | $40–80\text{ GB}$ | **Supported** | Ideal for cluster exploration (e.g. A100 / H100). High memory bandwidth accelerates parallel FFT rollouts across large batches (`--trials 64+`). |
| **CPU (Intel, AMD, Apple Silicon)** | Host RAM | **Supported** | Verified automatically in CI/CD. Ideal for testing, debugging, and smoke tests (slower than GPU due to CPU thread throughput). |
| **Non-CUDA Hardware (AMD ROCm / Apple Metal)** | VRAM / Unified | **Supported** | Supported via `pip install "jax[rocm]"` (AMD) or JAX Metal plugin on macOS without modifying any code. |

### Memory Management & Configuration

1. **VRAM Allocation Controls**:
   By default, JAX attempts to preallocate up to 90% of available GPU memory on startup. On shared machines or consumer GPUs with limited VRAM, configure dynamic allocation:
   ```bash
   # Prevent full preallocation and allocate memory dynamically as needed:
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   # Or limit JAX to a specific percentage of available VRAM:
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
   ```

2. **Cluster / SLURM Batch Execution**:
   When running on HPC clusters, submit experiments as batch jobs:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=flow-lenia-imgep
   #SBATCH --nodes=1
   #SBATCH --gpus=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=04:00:00

   module load cuda/12.3
   uv run python run_experiment.py --mode imgep --trials 100 --steps 3000 --output_dir results/cluster_imgep
   ```

3. **Numerical Consistency Across Devices**:
   All simulations use explicit PRNG keys (`jax.random.PRNGKey(seed)`). While physics dynamics remain qualitatively consistent across platforms, minor floating-point variations ($<10^{-6}$) may occur between GPU architectures due to compiler-level Fused Multiply-Add (FMA) optimizations.

---

## 3. Bachelor's Thesis Experimental Architecture

### Core Thesis Experiments (Authentic Flow-Lenia PDE & Open-Ended Evolution)

| Act | Experiment / Benchmark | Primary CLI Command | Key Result / Metric |
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

### Supplementary Exploratory Experiments (`experiments/supplementary/`)

These experiments explore practical bio-inspired applications and test the boundaries of Flow-Lenia by coupling Flow-Lenia PDE physics or visual profiles with external potential fields or discrete active matter mechanics. They are isolated in `experiments/supplementary/` and output to `results/supplementary/`:

> **Scientific Note on Problem-Solving Agency (The Navigation Fallacy)**:  
> If a local continuous cellular automaton ($R \approx 15\text{ px}$) could spontaneously solve global $256\times 256$ mazes or NP-hard TSP tours without memory or global lookahead, that would represent an unprecedented algorithmic breakthrough. In reality, the spatial pathfinding is computed by **precomputed external potential fields** (Dijkstra geodesic distance or Softmax gravity). Flow-Lenia provides the **continuous soft-matter physical substrate** (droplet cohesion, surface tension, mass conservation, elastomeric deformation), while the external field provides the **navigational guidance**. Maintaining this clear separation protects the thesis from exaggerated or misleading claims of "emergent cognitive intelligence".

| Module | Primary CLI Command | Mechanics / Design Rationale | Thesis Discussion Note |
| :--- | :--- | :--- | :--- |
| **Predator-Prey Trophic Ecology** | `uv run python run_experiment.py --mode predator_prey` | Discrete active matter + Flow-Lenia multi-shell visual morphology | *Why supplementary:* Isolated Flow-Lenia PDE solitons lose >97% mass in 2,000 steps without gene-mixing. True 2-species field coupling destabilizes the growth peak. Demonstrates that Flow-Lenia solitons are inherently collective population phenomena, not isolated individuals. |
| **Topological Transport & Bio-Routing** | `uv run python run_experiment.py --mode topological_transport` | Flow-Lenia PDE cohesion + external geodesic distance potential (Dijkstra) | *Why supplementary:* Soliton morphological integrity is PDE-driven, but spatial pathfinding is externally guided by Dijkstra potential fields rather than emergent PDE dynamics. |
| **Autonomous Multi-Body TSP Solver** | `uv run python run_experiment.py --mode tsp` | Flow-Lenia PDE cohesion + external Softmax gravitational city field | *Why supplementary:* Soliton cohesion is PDE-driven, but city selection is governed by an external Softmax potential. Demonstrates physical substrate coupling for NP-hard optimization problems. |
| **Collective Bridge Building** | `uv run python run_experiment.py --mode collective_bridge` | Flow-Lenia PDE droplets + directional guidance forces | *Why supplementary:* Demonstrates bio-inspired self-assembling living scaffolds across impassable chasms and conductive swarm transport using Flow-Lenia droplets with directional goal vectors. |

---

## 4. Directory Structure

```
.
├── core/                           # Physics engine, metrics, IMGEP, configs, visualization
│   ├── flow_lenia_jax.py           # JAX circular convolution & flux advection
│   ├── flow_lenia_trophic.py       # Two-species PDE trophic step (benchmarked prototype)
│   ├── metrics.py                  # 3D metric suite (EA, Motility, Complexity, Solidity)
│   ├── imgep.py                    # Goal-directed IMGEP exploration & FPS archiving
│   ├── environment.py              # Geometric wall masks and passage corridors
│   ├── visualization.py            # H.264 MP4, trajectory filmstrip, & heatmap exporter
│   └── config.py                   # Typed dataclass YAML configuration engine
├── experiments/                    # Core thesis experiments (Authentic Flow-Lenia Open-Ended Evolution)
│   ├── run_physics_verification.py # Classical Orbium glider verification (Act 1A)
│   ├── run_gene_mutation.py        # Multi-species Gumbel-Max mixing (Act 1B)
│   ├── run_negotiation_rule.py     # Softmax growth negotiation (Act 1C)
│   ├── run_imgep_search.py         # Open IMGEP vs. Random Search benchmark (Act 2A)
│   ├── run_autonomous_agentic_loop.py # Multi-generation AI Scientist discovery loop (Act 2B)
│   ├── run_chemotaxis_calibration.py # 3-way Cohesion vs Fission phase transition (Act 3)
│   ├── run_barrier_constriction.py # Passage width sweep & transmission curves (Act 4A)
│   ├── run_resource_depletion.py   # Dynamic niche depletion & foraging study (Act 4C)
│   ├── run_scaleup.py              # 512x512 scaled-up FPS reruns (Act 5A)
│   ├── run_epic_ecosystem.py       # Grand Synthesis Chemotactic Colosseum (Act 5B)
│   └── supplementary/              # Supplementary exploratory experiments
│       ├── run_predator_prey.py    # Multi-species predator-prey ecology (hybrid AM + FL)
│       ├── run_topological_transport.py # Graph maze navigation (FL + geodesic potential)
│       ├── run_traveling_salesperson.py # TSP solver (FL + softmax city field)
│       └── run_collective_bridge.py # Bio-inspired living bridge scaffold (FL + directional guidance)
├── configs/                        # Standardized YAML configuration files
│   ├── THESIS_ARCHITECTURE.md      # Exhaustive mathematical specification & formulas
│   ├── LITERATURE_READING_LIST.md  # Core papers & literature review
│   └── REFACTORING_HISTORY.md      # Engineering & scientific evolution log
├── .agents/                        # AI Assistant rules, memory, and skills
│   ├── AGENTS.md                   # Core behavioral rules & physics constraints
│   ├── CONTEXT_HANDOFF.md          # Active cross-session AI agent state memory
│   └── skills/                     # Specialized agent workflow skills
├── results/                        # Core simulation outputs
│   ├── baseline_imgep/             # IMGEP open search
│   ├── wall_obstacles/             # IMGEP barrier search
│   ├── resource_depletion/         # Static baseline vs dynamic foraging depletion
│   ├── gene_mutation/              # Multi-species Gumbel-Max mixing
│   ├── negotiation_rule/           # Softmax growth negotiation
│   ├── scaleup/                    # 512x512 scaled-up reruns
│   ├── orbium/                     # Classic Orbium glider verification
│   ├── agentic_loop/               # Multi-generation AI Scientist discovery state
│   └── supplementary/              # Supplementary experiment outputs
│       ├── predator_prey/          # Predator-prey Lotka-Volterra rollouts
│       ├── topological_transport/  # Maze, dynamic reroute, Tokyo rail, swarm channeling
│       ├── traveling_salesperson/  # TSP Hamiltonian tour optimization
│       └── collective_bridge/      # Living bridge scaffold & swarm transport
├── tests/                          # Automated unit test suite
├── Dockerfile                      # Container environment definition
├── run_experiment.py               # Main unified CLI runner
└── pyproject.toml                  # Package configuration & dependencies
```

---

## 5. Scientific Documentation

- **[THESIS_ARCHITECTURE.md](docs/THESIS_ARCHITECTURE.md)**: Exhaustive mathematical specification containing all continuous PDE equations, Fourier kernel formulations, 3D metric definitions, watertight filter rules, and parameter specification tables.
- **[LITERATURE_READING_LIST.md](docs/LITERATURE_READING_LIST.md)**: Curated list of seminal papers on continuous cellular automata, intrinsic motivation, and open-ended evolution.
- **[REFACTORING_HISTORY.md](docs/REFACTORING_HISTORY.md)**: Complete changelog documenting the transition from legacy PyTorch to JAX, Blackwell acceleration, Sobel gradient sign fixes, the Stochastic Seed Overfitting dilemma, and the discovery of the Chemotactic Cohesion-Fission Phase Transition.

---

## 6. Academic Attribution & Original Repositories

This Bachelor's Thesis codebase builds upon and extends the theoretical and computational foundation established by the following canonical literature and open-source repositories:

### Foundational Papers
1. **Michel et al. (2025/2026)**: *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* — [arXiv:2505.15998](https://arxiv.org/abs/2505.15998)
   - *Interactive Explorer*: [Flow-Lenia Universes Journal](https://developmentalsystems.org/Flow-Lenia-Universes-Journal/)
2. **Plantec et al. (2025)**: *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, *Artificial Life* 31(2) — [arXiv:2506.08569](https://arxiv.org/abs/2506.08569)
   - *Original Codebase*: [github.com/erwanplantec/FlowLenia](https://github.com/erwanplantec/FlowLenia)
   - *Project Website*: [sites.google.com/view/flowlenia](https://sites.google.com/view/flowlenia/)
3. **Chan (2019, 2020, 2023)**: *Lenia: Continuous Cellular Automata*, *Complex Systems* 28(3) — [arXiv:1812.05433](https://arxiv.org/abs/1812.05433)
   - *Original Lenia Codebase*: [github.com/Chakazul/Lenia](https://github.com/Chakazul/Lenia)
4. **Faldor & Cully (2024)**: *Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity ("Leniabreeder")*, *ALIFE 2024* — [arXiv:2406.04235](https://arxiv.org/abs/2406.04235)
   - *Leniabreeder Codebase*: [github.com/maxencefaldor/Leniabreeder](https://github.com/maxencefaldor/Leniabreeder)

### BibTeX Citation
```bibtex
@article{plantec2025flowlenia,
  title={Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata},
  author={Plantec, Erwan and Hamon, Erwan and Etcheverry, Mayalen and Chan, Bert Wang-Chak and Oudeyer, Pierre-Yves and Moulin-Frier, Cl{\'e}ment},
  journal={Artificial Life},
  volume={31},
  number={2},
  year={2025},
  publisher={MIT Press}
}

@article{michel2025exploring,
  title={Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist},
  author={Michel, Gautier and Cvjetko, Lana and Hamon, Erwan and Oudeyer, Pierre-Yves and Moulin-Frier, Cl{\'e}ment},
  journal={arXiv preprint arXiv:2505.15998},
  year={2025}
}

@article{chan2019lenia,
  title={Lenia: Continuous Cellular Automata},
  author={Chan, Bert Wang-Chak},
  journal={Complex Systems},
  volume={28},
  number={3},
  pages={275--323},
  year={2019}
}
```
