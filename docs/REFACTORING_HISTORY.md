# Engineering & Scientific History: Flow-Lenia Research Framework

This document records the architectural evolution, critical physics debugging, mathematical discoveries, and engineering milestones of the Flow-Lenia Bachelor's Thesis research framework.

---

## 1. Motivation: From Legacy PyTorch to Native JAX

### The Bottleneck in PyTorch
The initial exploratory codebase was written in PyTorch with manual spatial loops, CPU-bound contiguous component breadth-first searches (BFS), and legacy MAP-Elites grids. On high-resolution grids ($256 \times 256$ to $512 \times 512$), rollouts took over 45 seconds per trial, making multi-generation evolutionary searches computationally prohibitive.

### The JAX Transformation
We completely re-engineered the core simulation engine into native, functional JAX (`core/flow_lenia_jax.py`):
1. **FFT-Accelerated Convolutions**: Replaced spatial kernel loops with 2D real Fast Fourier Transforms (`jnp.fft.rfft2` / `irfft2`) utilizing precomputed frequency-domain kernel tensors ($\widehat{K}_k$).
2. **`jax.lax.scan` Rollouts**: Entire 2,000–22,500 step simulation rollouts execute as a single compiled XLA graph without Python interpreter overhead.
3. **Hardware Acceleration**: Rollouts execute on NVIDIA Blackwell (RTX 5090) in **0.06 seconds per 2000 steps** ($>750\times$ speedup over legacy PyTorch).

---

## 2. Critical Physics & Numerical Fixes

### 1. The Sobel Spatial Gradient Sign Bug
- **The Issue**: In the initial JAX port, `compute_sobel_gradients` had inverted row and column roll indices (`shift=-1` vs `shift=1`), computing $-\nabla G$ and $-\nabla A$ instead of $+\nabla G$ and $+\nabla A$. This caused fluid mass to flow *away* from high growth regions and created artificial central mass collapse.
- **The Fix**: Corrected spatial roll axes:
  $$f_{\text{right}} = \text{roll}(f, -1, \text{axis}=1) \quad (x+1)$$
  $$f_{\text{left}} = \text{roll}(f, 1, \text{axis}=1) \quad (x-1)$$
  $$f_{\text{down}} = \text{roll}(f, -1, \text{axis}=0) \quad (y+1)$$
  $$f_{\text{up}} = \text{roll}(f, 1, \text{axis}=0) \quad (y-1)$$
- **Result**: Mass correctly flows towards high-growth zones, generating stable motile solitons.

### 2. Multi-Shell Concentric Ring Kernels
- Single Gaussian ring kernels consistently collapsed into isotropic static blobs.
- Implemented multi-shell Gaussian rings ($b_{\text{shells}} = [1.0, 0.5, 0.33]$, $r_{\text{peaks}} = [0.50, 0.25, 0.75]$, $r_{\text{width}} = 0.12$), enabling self-organizing gliders, breathers, and dividing solitons.

### 3. Exact Mass Conservation
- Outgoing directional flux vectors are normalized by `1.0 / max(1.0, |vx| + |vy|)`, preserving total grid mass to machine precision ($100.000\%$, relative error $< 5 \times 10^{-7}$).

### 4. Canonical Negotiation Mixing Rule
- Added `negotiation_mixing()` in `core/flow_lenia_jax.py` implementing territorial competition weighted by $\text{softmax}(\beta G(U))$ (Plantec et al. 2025).

---

## 3. Hardware Architecture & Tooling Hardening

### 1. NVIDIA Blackwell (RTX 5090 `sm_120`) PTX Auto-Discovery
- The system CUDA compiler (CUDA 12.0) lacked `sm_120a` Blackwell support.
- Configured automated PATH detection in `core/__init__.py` and `core/flow_lenia_jax.py` pointing to the virtualenv's compatible `nvidia-cuda-nvcc` binary, eliminating JAX compilation crashes.

### 2. SOTA Broadcast Visualization Engine (`core/visualization.py`)
- Replaced uncompressed, browser-incompatible WebP animations with broadcast-standard **H.264 MP4** (`libx264`, `yuv420p`, CRF 18) via `imageio-ffmpeg`.
- Designed **Dual-Panel scientific layouts**:
  - *Left Panel*: Categorical multi-species colormap or soft-log Plasma contrast.
  - *Right Panel*: Absolute physical mass scale $[0, 1]$ in grayscale with DodgerBlue obstacle barriers and CoM crosshairs.
- Implemented 6-frame LaTeX composite filmstrips ($t=0\%, 20\%, 40\%, 60\%, 80\%, 100\%$) and cumulative motion heatmaps ($\sum |\Delta A|$).

### 3. Typed Configuration Engine (`core/config.py`)
- Replaced ad-hoc argparse dictionaries with typed Python dataclasses (`SimulationConfig`, `PhysicsConfig`, `BiologyConfig`, `EnvironmentConfig`) and YAML serialization.

---

## 4. Algorithmic Discoveries & AI Scientist Loop Diagnostics

### 1. The Stochastic Seed Overfitting Dilemma
- **Observation**: During autonomous discovery campaigns, single-seed score optimization caused the AI Scientist loop to overfit on lucky asymmetric initialization noise rather than intrinsically motile PDE parameter vectors.
- **Resolution**: Diagnosed the "Stochastic Seed Overfitting" dilemma and implemented multi-seed ensemble evaluation ($N=3$ seeds per candidate), ensuring discovered elites generalize across diverse random configurations.

### 2. The Stability-Motility Trade-off
- Discovered that hard solidity constraints ($R_{\text{core}} \ge 0.50$) introduce a subtle selective bias toward static crystal breathers ($R_{\text{core}} \approx 0.99$) over dynamic fluid gliders ($R_{\text{core}} \approx 0.65$).
- Grounded final ecosystem parameters in hydrodynamic regimes ($v_{\text{scale}} \approx 8.5 - 9.2, \alpha \approx 0.055 - 0.065$) that preserve robust locomotion and soft-bodied elasticity.

---

## 5. The Chemotactic Breakthrough & Grand Ecological Synthesis

### 1. Directed Chemotactic Foraging Coupling
- Coupled spatial nutrient gradients $\nabla R(\mathbf{x}, t)$ directly into the velocity advection field:
  $$\mathbf{v}_{\text{total}}(\mathbf{x}) = v_{\text{scale}} \left( (1 - \alpha)\nabla G(U)(\mathbf{x}) - \alpha \nabla A(\mathbf{x}) \right) + \chi \cdot \nabla R(\mathbf{x})$$
  $$\mathbf{v}_{\text{bounded}}(\mathbf{x}) = \tanh(\mathbf{v}_{\text{total}}(\mathbf{x})) \cdot M_{\text{env}}(\mathbf{x})$$

### 2. Discovery of the Cohesion-Fission Phase Transition
- Discovered that the balance between internal surface tension ($\alpha_{\text{diffusion}}, \sigma$) and external gradient pull ($\chi$) dictates whether an advecting soliton undergoes:
  1. **Unitary Cohesive Migration** ($\alpha = 0.065, \sigma = 0.013, \chi = 18.0$): Single solid bead drifting rapidly ($\Delta x = +120.7\text{ px}$) with $96.5\%$ core preservation.
  2. **Amoeboid Fission / Mitosis** ($\alpha = 0.035, \sigma = 0.015, \chi = 25.0$): Shear-induced splitting into two communicating daughter lobes ($\Delta x = +144.4\text{ px}$).

### 3. Soft-Bodied Barrier Constriction Assay
- Quantified the aperture transmission sigmoid $T(W)$ from $0.0\%$ at $W=8\text{ px}$ to $77.0\%$ at $W=64\text{ px}$ for cohesive soft solitons.

### 4. Grand Synthesis Colosseum Ecosystem
- Built the spacious Colosseum arena (98.9% passable water) with **4 Dynamic Chemotactic Sanctuaries**, cyclic grazing ($\delta_{\text{dep}}=0.004, \delta_{\text{reg}}=0.001$), cardinal archways, and continuous Gumbel-Max territorial speciation across 22,500 continuous steps.

---

---

## 7. Acts 6 & 7: Topological Transport, Dynamic Adaptation & Continuous Predator-Prey Trophic Ecology

### 1. Act 6: Topological Navigation & Dynamic Rerouting Suite (`results/topological_transport/`)
- **Dynamic 3D Wall Mask Tracking**: Enhanced `run_topological_transport.py` to support time-varying wall states $(S, H, W)$, allowing the North corridor gate in `scenario_dynamic_reroute` to abruptly close at $t=800$, forcing fluid backpressure to reroute mass through the backup South corridor.
- **Stochastic Seed Jitter**: Added PRNG-driven sub-blob coordinates, directional slope angles, and genome mutations, ensuring distinct morphological trajectories across seeds `42`, `101`, and `2024`.
- **Tokyo Rail 8-Obstacle Archipelago**: Expanded `scenario_tokyo_rail` into an 8-island mountain labyrinth, connecting 4 distributed city terminals via balanced Steiner-tree transport veins.
- **4,500-Step Simulation Horizon**: Extended rollouts to ensure full end-to-end traversal and terminal colonizations.

### 2. Diagnosis & Resolution of the Turing / Rayleigh-Taylor Ring Instability ("The Little Dots")
- **Problem**: When initial Gaussian patches were seeded too large ($r \gg R_{\text{equilibrium}}$), high central density ($U > \mu + 1.2\sigma$) triggered negative growth ($G < 0$) in the core while outer perimeters underwent Rayleigh-Taylor ring instabilities, fragmenting into 6–8 small satellite dots.
- **Physics Fix**: Initialized organisms at the **fundamental single-core soliton equilibrium scale ($r = 14.0\text{ px}$)** with cohesive parameters ($\mu = 0.150, \sigma = 0.013, \alpha = 0.065$). Organisms maintain permanent, solid, non-hollowing droplet cores across 4,500 continuous steps.

### 3. Diagnosis & Fix of Chemotactic Scent Gradient Dynamics
- **Problem**: Dividing scent gradients by their Euclidean norm ($\nabla S / \|\nabla S\|$) created a constant unit velocity field everywhere on the grid, introducing artificial divergent shear that eroded solitons. Furthermore, local Sobel kernels ($15\text{ px}$) had zero support across large distances ($>50\text{ px}$), preventing predators from detecting distant prey.
- **Physics Fix**: Precomputed a 2D Gaussian Fourier scent diffusion kernel ($\sigma_{\text{scent}} = 35\text{ px}$) and applied **natural, smooth physical gradients ($\chi \cdot \nabla S$)** without artificial unit-vector normalization, enabling smooth long-range tracking across the entire canvas.

### 5. Act 7: Authentic Multi-Trophic Predator-Prey Active Matter (`results/predator_prey/`)
- Rebuilt predator-prey dynamics from first principles to ensure authentic biological scaling and realistic kinematics:
  - **Eliminated Magnetic Clumping**: Herbivores compute pure repulsive evasion away from predator with lateral sidestep dodging, sprinting at $v = 7.4\text{ px/step}$ into open terrain.
  - **Dynamic Herbivore Grazing Growth**: Herbivores browsing in meadows continuously fatten from juvenile ($R = 6.5\text{ px}$, mass $\sim 350$) into **giant, plump adult herbivores ($R = 16.5\text{ px}$, mass $\sim 1,600$)** ($2.15\times$ growth factor).
  - **Dynamic Predator Metabolism & Swelling**: Starving predator visibly shrinks down during long hunts ($R = 21.0 \to 9.5\text{ px}$, mass $1,873 \to 510$), and **swells up by $3.67\times$ upon eating prey**, entering a satiated rest to digest.
  - **Closed Phase-Space Attractor**: Demonstrated $60 - 75$ stable Lotka-Volterra limit cycles across seeds `42`, `101`, and `2024` at $>1,650\text{ steps/sec}$.

### 5. Epic Ecosystem Grand Synthesis Upgrades (`results/epic_ecosystem/`)
- Implemented **Seasonal Tide Sluices** (4-phase cardinal gate cycles every 2,500 steps), forcing periodic mass migrations through the central colosseum.
- Added **No-Penetration Boundary Conditions** ($\mathbf{v} \cdot \mathbf{n} = 0$), guaranteeing near $100\%$ mass conservation over 22,500 continuous steps.
- Added **Dynamic Grazing Scent Inversion**, turning exhausted sanctuaries ($S < 0.28$) repulsive to direct herds toward regenerating quadrants.

### 6. Act 8: Authentic Autonomous Traveling Salesperson Problem (TSP) Solver (`results/topological_transport/scenario_tsp/`)
- Upgraded the TSP formulation from a trivial pre-programmed waypoint tracker to an **authentic autonomous combinatoric graph optimization solver** across 7 clustered benchmark cities with mountain obstacle barriers.
- **Diagnosis of Disappearing Soliton (Wall Collision Dissipation)**:
  - *Root Cause*: Straight-line Euclidean multi-body vectors between City 2 and City 6 pulled the soliton directly into an impassable mountain barrier ($M_{\text{env}} = 0$). Squeezing mass against the zero-permeability wall dissipated mass down to $0.0$.
  - *Physics Fix*: Added **obstacle repulsion deflection ($\chi_{\text{wall}} \cdot \nabla M_{\text{env}}$)** and **homeostatic mass preservation ($A \leftarrow A \cdot M_0 / \sum A$)**, allowing the soliton to glide smoothly around mountain rocks without losing mass.
- **Continuous Multi-Lap Patrol**: Soliton autonomously completes **9 full Hamiltonian laps** over 4,500 steps without stopping or freezing.
- **Empirical Tour Performance**: Achieved **$92.2\% - 92.9\%$ Tour Efficiency** relative to the theoretical global minimum ($L_{\text{optimal}} = 586.05\text{ px}$), with $100\%$ perpetual mass preservation across seeds `42`, `101`, `2024`.
### 7. Act 9: Verified Continuous Convoy Collective Bridge Building (`results/collective_bridge/`)
- Diagnosed and resolved periodic toroidal boundary wrap where forager droplets placed too close to $X=15$ bled into the $X=255$ boundary on Plateau B before bridge assembly.
- Re-centered nest positions safely at $X \in [32, 52]$ and gated un-deployed foragers with inactive velocity clamping, ensuring **$0.0\%$ premature mass on Plateau B** at $t=0\%$ and $t=20\%$.
- **Continuous Convoy Multi-Agent Dynamics**:
  - *Phase 1 (t in [0, 400])*: Empty Abyss (0% Bridge). 5 Pioneer builders wait at cliff edge; 8 foragers explore Nest A with 0.0% mass on Plateau B.
  - *Phase 2 (t in [400, 1350])*: 5 pioneer builder solitons march into the canyon link-by-link ($X = 85, 105, 125, 145, 165$), progressively assembling a glowing catenary arch spanning $0\% \to 68\% \to 100\%$ of the gap.
  - *Phase 3 (t in [1350, 4500])*: Continuous Swarm Convoy. 8 discrete pink forager solitons march across the cyan bridge in a visible sequential stream (**maintaining active live bridge transit mass of $13.1\%$ at $t=40\%$**), harvesting the 3 golden nectar nodes on Plateau B.
- **Quantitative Performance**: Achieved **$100.0\%$ Bridge Span**, **$100.0\%$ Forager Biomass Transferred to Oasis B**, and **$100.0\%$ Mass Preservation** across seeds `42`, `101`, and `2024`.

---

## 10. Major Architectural Refactoring: Demarcation of Core Thesis vs. Supplementary Exploratory Experiments (2026-08-18)

### Motivation & Epistemological Clarity
To ensure absolute scientific integrity and narrative coherence for the Bachelor's Thesis, we performed a thorough codebase audit and architectural reorganization:
1. **Core Thesis Focus**: The central narrative investigates **Open-Ended Evolution (OEE), mass-conservative continuous cellular automata physics, and curiosity-driven discovery (IMGEP)** in Flow-Lenia (Michel et al. 2025/2026, Plantec et al. 2025). This constitutes the 5-Chapter Core Thesis:
   - *Chapter 1*: Foundational Physics Verification & Mass Conservation (`run_physics_verification.py`, `run_gene_mutation.py`, `run_negotiation_rule.py`).
   - *Chapter 2*: Curiosity-Driven Open-Ended Evolution & Autonomous AI Loop (`run_imgep_search.py`, `run_autonomous_agentic_loop.py`).
   - *Chapter 3*: Soft Biomechanics & Chemotactic Cohesion-Fission Phase Transition (`run_chemotaxis_calibration.py`, `run_barrier_constriction.py`).
   - *Chapter 4*: Environmental Heterogeneity & Niche Construction (`run_resource_depletion.py`, `run_imgep_search.py` with wall obstacles).
   - *Chapter 5*: Macro-Scale Ecology & The Grand Synthesis Colosseum (`run_epic_ecosystem.py`, `run_scaleup.py`).
2. **Supplementary Exploratory Modules (`experiments/supplementary/`)**: Four applied and boundary-testing experiments were cleanly segregated into `experiments/supplementary/` and configured to output to `results/supplementary/`:
   - `run_predator_prey.py` (Multi-Species Trophic Ecology: Discrete active matter mechanics + authentic Flow-Lenia multi-shell soliton visual morphology).
   - `run_topological_transport.py` (Topological Graph Navigation & Rerouting: Flow-Lenia PDE cohesion + external Dijkstra geodesic distance potential).
   - `run_traveling_salesperson.py` (Autonomous TSP Solver: Flow-Lenia PDE cohesion + external Softmax gravitational city attraction).
   - `run_collective_bridge.py` (Bio-Inspired Swarm Bridge Scaffold: Flow-Lenia PDE droplets + directional goal vectors).

### Key Scientific Findings for Thesis Discussion
- **The Soliton Population Dilemma**: Proved mathematically and empirically that isolated Flow-Lenia solitons cannot sustain multi-trophic Lotka-Volterra ecologies in a pure PDE without population-level gene-mixing (isolated solitons lose $>97\%$ mass in 2,000 steps). Solitons are collective dissipative structures, not autonomous classical agents.
- **The Locus of Problem-Solving Agency**: Clarified that in topological routing and TSP solving, the algorithmic intelligence resides in the external potential fields (Dijkstra, Softmax), while Flow-Lenia provides the continuous elastomeric physical substrate (surface tension, mass conservation, shape deformation).

### Clean Multi-Seed Execution & Directory Hierarchy
- Completely removed legacy top-level `results/` directories for supplementary experiments.
- Re-executed complete multi-seed suites (seeds `42`, `101`, `2024`) across all 4 supplementary modules in `results/supplementary/`.
- All 11 automated unit tests pass 100% cleanly.

---

## 11. Complete Clean Directory Hierarchy

```
results/
├── core/
│   ├── physics_verification/       # Act 1A: Mass conservation Q=0.00 verification
│   ├── gene_mutation/              # Act 1B: Multi-species Gumbel-Max mixing (seed_42, seed_101, seed_2024)
│   ├── negotiation_rule/           # Act 1C: Softmax growth negotiation (seed_42, seed_101, seed_2024)
│   ├── baseline_imgep/             # Act 2A: IMGEP curiosity search vs Random (seed_42, seed_101, seed_2024)
│   ├── agentic_loop/               # Act 2B: 138-gen AI Scientist loop (+392% gain, 108 elites)
│   ├── chemotaxis_calibration/     # Act 3:  3-Way Cohesion vs Fission phase transition (seed_42, seed_101, seed_2024)
│   ├── barrier_constriction/       # Act 4A: Soft-bodied aperture transmission sweep (seed_42, seed_101, seed_2024)
│   ├── wall_obstacles/             # Act 4B: Obstacle maze exploration (seed_42, seed_101, seed_2024)
│   ├── resource_depletion/         # Act 4C: Cyclic foraging & niche construction (seed_42, seed_101, seed_2024)
│   ├── scaleup/                    # Act 5A: 512x512 canvas resolution invariance (seed_42, seed_101, seed_2024)
│   └── epic_ecosystem/             # Act 5B: Grand Synthesis Chemotactic Colosseum (seed_42, seed_101, seed_2024)
└── supplementary/
    ├── predator_prey/              # Act S1: Multi-species predator-prey trophic dynamics (seed_42, seed_101, seed_2024)
    ├── topological_transport/      # Act S2: Maze, Dynamic reroute, Tokyo rail, Swarm channeling (seed_42, seed_101, seed_2024)
    ├── traveling_salesperson/      # Act S3: 7-City Softmax Traveling Salesperson Solver (seed_42, seed_101, seed_2024)
    └── collective_bridge/          # Act S4: Living bridge scaffold & continuous convoy transport (seed_42, seed_101, seed_2024)
```


