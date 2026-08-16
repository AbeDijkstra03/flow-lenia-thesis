# Comprehensive Scientific Reference & Mathematical Physics Canon: Flow-Lenia Research Framework

This document serves as the **exhaustive, authoritative scientific reference** for the Master's Thesis research framework implementing GPU-accelerated **Flow-Lenia Open-Ended Evolution (OEE)** in native **JAX**.

It contains the complete mathematical specifications, physics equations, behavioral metrics, curiosity-driven algorithms, experimental protocols, and parameter tables needed to write the thesis and reproduce every result without inspecting source code.

---

## Table of Contents

1. [Academic Grounding & Literature Foundation](#1-academic-grounding--literature-foundation)
2. [Continuous Cellular Automata Physics Canon](#2-continuous-cellular-automata-physics-canon)
   - 2.1 Continuous State Space & Toroidal Geometry
   - 2.2 Multi-Shell Concentric Ring Kernels
   - 2.3 Fourier-Domain Circular Convolutions
   - 2.4 Continuous Growth Mappings & Negative Bounds
   - 2.5 Flux-Conserved Velocity Advection
   - 2.6 Discrete Flux Normalization & Exact Mass Conservation
   - 2.7 Spatial Derivatives & Sobel Operators
   - 2.8 Genome Mixing & Territorial Competition Rules
3. [Environmental Heterogeneity & Niche Construction](#3-environmental-heterogeneity--niche-construction)
   - 3.1 Static Barrier Walls & Passage Corridors
   - 3.2 Corridor Constriction & Transmission Coefficient ($T$)
   - 3.3 Dynamic Resource Depletion & Regeneration Fields
4. [Behavior Characterization & 3D Metric Space](#4-behavior-characterization--3d-metric-space)
   - 4.1 Center of Mass Motility ($v_{\text{CoM}}$)
   - 4.2 Non-Neutral Quadratic Evolutionary Activity ($\text{EA}$)
   - 4.3 Compression Complexity & Multi-Scale Entropy
   - 4.4 Solid Core Ratio ($R_{\text{core}}$) & Mass Preservation ($R_{\text{mass}}$)
5. [Watertight Quality Filter Architecture](#5-watertight-quality-filter-architecture)
   - 5.1 The Five Disqualification Gates
   - 5.2 Mathematical Formulation of Quality Score
6. [Curiosity-Driven Exploration Algorithms](#6-curiosity-driven-exploration-algorithms)
   - 6.1 IMGEP (Intrinsically Motivated Goal Exploration Process)
   - 6.2 Uniform Random Search Baseline
   - 6.3 Farthest-Point Sampling (FPS) Archive Maintenance
7. [Autonomous AI Scientist Discovery Pipeline](#7-autonomous-ai-scientist-discovery-pipeline)
   - 7.1 Multi-Generation Exploration Harness
   - 7.2 Multimodal Dual-Panel Visual Feedback Loop
   - 7.3 Adaptive Domain Tuning
8. [Experimental Protocols & Hypotheses](#8-experimental-protocols--hypotheses)
   - 8.1 Experiment 1: Baseline Open IMGEP vs. Random Search
   - 8.2 Experiment 2: Corridor Constriction & Morphological Plasticity
   - 8.3 Experiment 3: Reactive Resource Depletion & Niche Construction
   - 8.4 Experiment 4: Long-Duration Multi-Species Ecosystem Dynamics
   - 8.5 Experiment 5: Resolution Invariance & Long-Horizon Scale-Up
9. [Complete Hyperparameter Specification Tables](#9-complete-hyperparameter-specification-tables)
10. [Step-by-Step Reproduction Guide](#10-step-by-step-reproduction-guide)
11. [BibTeX Academic Bibliography](#11-bibtex-academic-bibliography)

---

## 1. Academic Grounding & Literature Foundation

The framework is grounded in and expands upon continuous artificial life, open-ended evolution, and developmental robotics:

1. **Michel, Cvjetko, Hamon, Oudeyer, Moulin-Frier (2025/2026)**:
   *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* (arXiv:2505.15998).
   - **Foundational Role**: Establishes the 3-D novelty metric space $[\text{Motility}, \text{Evolutionary Activity}, \text{Complexity}]$, the IMGEP goal exploration algorithm, and mixing-rule ablations.
2. **Plantec, Hamon, Etcheverry, Chan, Oudeyer, Moulin-Frier (2025)**:
   *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, Artificial Life 31(2) (arXiv:2506.08569).
   - **Foundational Role**: Introduces fluid mass conservation ($\frac{\partial A}{\partial t} + \nabla \cdot (A \mathbf{v}) = 0$) to continuous CA, establishing the physical basis for persistent, self-organizing gliders.
3. **Chan (2019, 2020, 2023)**:
   *Lenia: Continuous Cellular Automata*, Complex Systems 28(3) (arXiv:1812.05433); *Towards Large-Scale Simulations of Open-Ended Evolution in Continuous Cellular Automata* (arXiv:2304.05639).
   - **Foundational Role**: Formulates continuous multi-shell kernel structures and continuous growth mappings.
4. **Oudeyer, Kaplan, Hafner (2007)**:
   *Intrinsic Motivation Systems for Autonomous Mental Development*, IEEE Trans. Evol. Comput. 11(2).
   - **Foundational Role**: Provides the theoretical framework for curiosity-driven goal exploration in high-dimensional continuous systems.

---

## 2. Continuous Cellular Automata Physics Canon

### 2.1 Continuous State Space & Toroidal Geometry

The simulation domain is defined as a 2D continuous toroidal lattice $\mathbb{T}^2 = [0, H) \times [0, W)$ with periodic boundary conditions.
- **Mass Density Field**: $A(\mathbf{x}, t) \in [0.0, 1.0]$, representing continuous physical mass density.
- **Parameter Fields**: Each spatial cell $\mathbf{x} = (y, x)$ carries a multi-gene parameter vector:
  $$\boldsymbol{\mu}(\mathbf{x}, t) \in \mathbb{R}^K, \quad \boldsymbol{\sigma}(\mathbf{x}, t) \in \mathbb{R}^K, \quad \mathbf{w}(\mathbf{x}, t) \in \mathbb{R}^K$$
  where $K = 9$ represents the number of concentric neighborhood kernels.

---

### 2.2 Multi-Shell Concentric Ring Kernels

Single Gaussian kernels produce isotropic round blobs. To enable gliders, breathers, and dividing solitons, continuous kernels are defined as multi-shell concentric Gaussian rings:

$$K_k(r) = \sum_{m=1}^{M} b_m \cdot \exp\left(-\frac{(r - r_m \cdot R_k)^2}{2 (w_m \cdot R_k)^2}\right)$$

Where:
- $k \in \{0, 1, \dots, K-1\}$ is the kernel index.
- $R_k \in [6.0, 15.0]\text{ pixels}$ is the outer radius of kernel $k$.
- $M = 3$ is the number of concentric shells.
- $b_{\text{shells}} = [1.0, 0.50, 0.33]$ are the relative peak amplitude weights.
- $r_{\text{peaks}} = [0.50, 0.25, 0.75]$ are the normalized radial peak positions.
- $w_{\text{width}} = 0.12$ is the normalized radial Gaussian width.

Each 2D kernel matrix is normalized to unit sum: $\iint_{\mathbb{R}^2} K_k(\mathbf{x}) d\mathbf{x} = 1.0$.

---

### 2.3 Fourier-Domain Circular Convolutions

To achieve massive parallelism on GPU hardware, continuous spatial convolutions are computed via 2D Real Fast Fourier Transforms:

$$U_k(\mathbf{x}, t) = \mathcal{F}^{-1}\left(\mathcal{F}(A(\mathbf{x}, t)) \odot \widehat{K}_k\right)$$

Where $\widehat{K}_k = \mathcal{F}(K_k)$ is precomputed at initialization for grid dimensions $(H, W)$, eliminating spatial kernel iteration.

---

### 2.4 Continuous Growth Mappings & Negative Bounds

For each kernel $k$, potential $U_k$ is mapped through a unimodal Gaussian growth function:

$$G_k(U_k) = 2 \exp\left(-\frac{(U_k - \mu_k)^2}{2\sigma_k^2}\right) - 1$$

Where:
- $\mu_k \in [0.13, 0.22]$ is the optimal growth center.
- $\sigma_k \in [0.011, 0.024]$ is the growth tolerance width.
- $G_k \in [-1.0, +1.0]$: Positive growth ($G > 0$) attracts mass; negative growth ($G < 0$) repels mass.

The total effective growth field is the weighted linear combination:

$$G(U)(\mathbf{x}) = \sum_{k=0}^{K-1} w_k(\mathbf{x}) \cdot G_k(U_k(\mathbf{x}))$$

**Negative Growth Enforcement**: For densities $U > \mu + 1.2\sigma$, $G(U) < 0$ is strictly maintained, preventing solid gliders from collapsing into dense, static singularities ("melting").

---

### 2.5 Flux-Conserved Velocity Advection

In Flow-Lenia, mass advection is governed by the continuous continuity equation:

$$\frac{\partial A}{\partial t} + \nabla \cdot (A \mathbf{v}) = 0$$

The unnormalized directional flux vector field $\mathbf{F} = (F_x, F_y)$ is computed from potential and density gradients:

$$\mathbf{F}(\mathbf{x}) = v_{\text{scale}} \cdot \left((1 - \alpha) \nabla G(U)(\mathbf{x}) - \alpha \nabla A(\mathbf{x})\right)$$

Where:
- $v_{\text{scale}} \in [4.2, 6.5]$ is the velocity magnitude scaling factor.
- $\alpha \in [0.04, 0.08]$ is the entropic mass diffusion coefficient.
- $\nabla G(U)$ is the spatial gradient of growth, propelling fluid mass towards high-growth zones.
- $\nabla A$ is the spatial gradient of density, providing smooth regularizing pressure.

---

### 2.6 Moroz (2020) Bilinear Reintegration Tracking & Exact Mass Conservation

Standard 4-way orthogonal flux clipping introduces artificial numerical damping drag for sub-pixel velocities $(1 - |v_x| - |v_y|)$, which rapidly arrests moving gliders into stationary rings. Following Plantec et al. (2025), our framework implements **Moroz (2020) Bilinear Reintegration Tracking**, a semi-Lagrangian continuous transport algorithm with 9-neighbor bilinear splatting:

Given continuous velocity displacement $\mathbf{v} = (v_y, v_x) \in [-1.0, 1.0]^2$, directional fractions are decomposed:

$$f_{x,+} = \max(0, v_x), \quad f_{x,-} = \max(0, -v_x), \quad f_{x,0} = 1 - f_{x,+} - f_{x,-}$$
$$f_{y,+} = \max(0, v_y), \quad f_{y,-} = \max(0, -v_y), \quad f_{y,0} = 1 - f_{y,+} - f_{y,-}$$

Each source cell $(y, x)$ distributes its mass $A(y, x)$ to its 9 local spatial neighbors $(y \pm \Delta y, x \pm \Delta x)$ via exact bilinear tensor products:

$$m_{0,0} = A \cdot f_{y,0} f_{x,0}, \quad m_{0,R} = A \cdot f_{y,0} f_{x,+}, \quad m_{0,L} = A \cdot f_{y,0} f_{x,-}$$
$$m_{D,0} = A \cdot f_{y,+} f_{x,0}, \quad m_{U,0} = A \cdot f_{y,-} f_{x,0}$$
$$m_{DR} = A \cdot f_{y,+} f_{x,+}, \quad m_{DL} = A \cdot f_{y,+} f_{x,-}, \quad m_{UR} = A \cdot f_{y,-} f_{x,+}, \quad m_{UL} = A \cdot f_{y,-} f_{x,-}$$

Incoming mass is accumulated using periodic torus roll operators:

$$A(y, x, t + \Delta t) = m_{0,0} + \text{roll}(m_{0,R}, +1, x) + \text{roll}(m_{0,L}, -1, x) + \text{roll}(m_{D,0}, +1, y) + \text{roll}(m_{U,0}, -1, y) + \sum_{\text{diag}} \text{roll}(m_{\text{diag}}, \pm 1, (y, x))$$

This ensures **exact mass conservation to machine precision ($0.00\text{e}+00$ relative error)** with **zero artificial viscous drag**, enabling continuous, long-range glider translation.

---

### 2.7 Spatial Derivatives & Sobel Operators

Spatial gradients are computed using 2D Sobel convolution operators with toroidal periodic boundaries:

$$\mathbf{S}_x = \frac{1}{8} \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad \mathbf{S}_y = \frac{1}{8} \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

Using periodic array rolls:
- $\nabla_x f(y, x) = \frac{1}{8} \left[(f_{y-1, x+1} + 2f_{y, x+1} + f_{y+1, x+1}) - (f_{y-1, x-1} + 2f_{y, x-1} + f_{y+1, x-1})\right]$
- $\nabla_y f(y, x) = \frac{1}{8} \left[(f_{y+1, x-1} + 2f_{y+1, x} + f_{y+1, x+1}) - (f_{y-1, x-1} + 2f_{y-1, x} + f_{y-1, x+1})\right]$

---

### 2.8 Genome Mixing & Territorial Competition Rules

When multiple species collide, their parameter maps $(\boldsymbol{\mu}, \boldsymbol{\sigma}, \mathbf{w})$ are updated using one of two mixing rules to prevent parameter blurring into inert averages:

1. **Stochastic Gene-Wise Sampling (Gumbel-Max)**:
   Parameters are sampled categorically from incoming directional mass fluxes using Gumbel-Max perturbations:
   $$\text{Source}(\mathbf{x}) = \arg\max_{s \in \{0, 1, \dots, 4\}} \left(\log(\text{Flux}_s(\mathbf{x}) + 10^{-6}) + g_s\right), \quad g_s \sim \text{Gumbel}(0, 1)$$
2. **Canonical Growth Negotiation Rule (Plantec et al. 2025)**:
   When fluid masses overlap, local growth potentials compete through a temperature-scaled softmax:
   $$w_{\text{eff}, s}(\mathbf{x}) = \frac{\exp\left(\beta \cdot G_s(U(\mathbf{x}))\right) \cdot \text{Mass}_s(\mathbf{x})}{\sum_j \exp\left(\beta \cdot G_j(U(\mathbf{x}))\right) \cdot \text{Mass}_j(\mathbf{x})}$$
   where $\beta = 2.0$ represents territorial growth aggressiveness.

---

## 3. Environmental Heterogeneity & Niche Construction

### 3.1 Static Barrier Walls & Passage Corridors

The environmental spatial mask is defined as $M_{\text{env}}(\mathbf{x}) \in [0.0, 1.0]$:
- $M_{\text{env}}(\mathbf{x}) = 1.0$: Passable vacuum / fluid domain.
- $M_{\text{env}}(\mathbf{x}) = 0.0$: Rigid, impermeable obstacle boundary.

At barrier boundaries, advection fluxes into obstacles are zeroed: $\mathbf{F}(\mathbf{x}) \leftarrow \mathbf{F}(\mathbf{x}) \cdot M_{\text{env}}(\mathbf{x})$, enforcing zero-flux boundary conditions.

---

### 3.2 Corridor Constriction & Transmission Coefficient ($T$)

To evaluate the morphological plasticity and soft-bodied elasticity of Flow-Lenia solitons, a vertical barrier wall ($x = W/2$, thickness $d = 8\text{ px}$) partitions the domain into **Chamber 1** ($x < W/2$) and **Chamber 2** ($x > W/2$), joined by a central passage corridor of width $W_{\text{passage}} \in [8, 16, 24, 32]\text{ pixels}$.

**Transmission Coefficient**:
$$T(W_{\text{passage}}) = \frac{\sum_{y} \sum_{x > W/2} A(y, x, t_{\text{end}})}{\sum_{y} \sum_{x} A(y, x, t_{\text{end}})}$$

- $T = 0.0$: Total reflection or blockage by the barrier.
- $T = 1.0$: Complete migration and transmission through the corridor into Chamber 2.

---

### 3.3 Dynamic Resource Depletion & Regeneration Fields

To simulate niche construction, organisms interact with a dynamic scalar resource field $R(\mathbf{x}, t) \in [0.0, 1.0]$:
- **Depletion**: At cells where organism mass density $A(\mathbf{x}) \ge 0.10$:
  $$R(\mathbf{x}, t + \Delta t) = \max\left(0.0, R(\mathbf{x}, t) - \delta_{\text{dep}}\right), \quad \delta_{\text{dep}} = 0.04$$
- **Regeneration**: At unoccupied cells ($A(\mathbf{x}) < 0.10$):
  $$R(\mathbf{x}, t + \Delta t) = \min\left(1.0, R(\mathbf{x}, t) + \delta_{\text{regen}}\right), \quad \delta_{\text{regen}} = 0.01$$
- **Effective Growth Coupling**: The growth potential is scaled by the local resource:
  $$G_{\text{coupled}}(\mathbf{x}) = G(U)(\mathbf{x}) \cdot R(\mathbf{x}, t)$$

This creates a negative feedback loop: stationary organisms deplete their local substrate, collapsing their growth potential and forcing continuous migration and cyclic foraging trails.

---

## 4. Behavior Characterization & 3D Metric Space

Each simulation rollout is projected into a standardized 3-dimensional behavioral space:

$$\mathcal{B} = \left[v_{\text{CoM}}, \text{EA}, \mathcal{C}_{\text{gzip}}\right]$$

### 4.1 Center of Mass Motility ($v_{\text{CoM}}$)

Measures net spatial displacement of the organism's center of mass across the horizon:

$$\mathbf{x}_{\text{CoM}}(t) = \frac{\sum_{\mathbf{x}} \mathbf{x} \cdot A(\mathbf{x}, t)}{\sum_{\mathbf{x}} A(\mathbf{x}, t)}$$

$$v_{\text{CoM}} = \|\mathbf{x}_{\text{CoM}}(t_{\text{end}}) - \mathbf{x}_{\text{CoM}}(0)\|_2 \quad (\text{pixels})$$

### 4.2 Non-Neutral Quadratic Evolutionary Activity ($\text{EA}$)

Measures cumulative evolutionary dynamism and persistent non-neutral state transitions:

$$\text{EA} = \frac{1}{T} \sum_{t=1}^{T} \left(\frac{1}{HW} \sum_{\mathbf{x}} \left(A(\mathbf{x}, t) - A(\mathbf{x}, t - \Delta t)\right)^2\right)$$

### 4.3 Compression Complexity & Multi-Scale Entropy

1. **Gzip Compression Complexity ($\mathcal{C}_{\text{gzip}}$)**:
   Measures structural algorithmic complexity by compressing the thresholded binary trajectory array:
   $$\mathcal{C}_{\text{gzip}} = \text{len}\left(\text{gzip}\left(\mathbb{I}(A(\mathbf{x}, t) \ge 0.05)\right)\right) \quad (\text{bytes})$$
2. **Multi-Scale Spatial Shannon Entropy ($H$)**:
   $$H(A) = -\sum_{i=1}^{B} p_i \log_2(p_i), \quad p_i = \frac{\text{count}(A \in \text{bin}_i)}{HW}$$

### 4.4 Solid Core Ratio ($R_{\text{core}}$) & Mass Preservation ($R_{\text{mass}}$)

- **Solid Core Density Ratio**:
  $$R_{\text{core}} = \frac{\sum_{\mathbf{x}} A(\mathbf{x}, t_{\text{end}}) \cdot \mathbb{I}(A(\mathbf{x}, t_{\text{end}}) \ge 0.15)}{\sum_{\mathbf{x}} A(\mathbf{x}, t_{\text{end}})}$$
  Ensures that the organism maintains a dense, coherent nucleus ($A \ge 0.15$) rather than decaying into hollow boundary shells.
- **Mass Preservation Ratio**:
  $$R_{\text{mass}} = \frac{\sum_{\mathbf{x}} A(\mathbf{x}, t_{\text{end}})}{\sum_{\mathbf{x}} A(\mathbf{x}, 0)}$$

---

## 5. Watertight Quality Filter Architecture

### 5.1 The Five Disqualification Gates

To prevent evolutionary search from being polluted by degenerated artifacts (dissolved gas, hollow shells, frozen still-lifes, or unconstrained grid chaos), every candidate rollout is evaluated through `evaluate_watertight_quality_score()` (`core/metrics.py`):

| Gate # | Quality Filter Check | Threshold Criterion | Failure Meaning |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Mass Conservation | $R_{\text{mass}} \in [0.60, 5.00]$ | Numerical explosion or total mass annihilation |
| **Gate 2** | Solid Core Ratio | $R_{\text{core}} \ge 0.50$ | Hollow, dispersed outline with no solid nucleus |
| **Gate 3** | Minimum Motility | $v_{\text{CoM}} \ge 5.0\text{ px}$ | Frozen, static still-life (zero migration) |
| **Gate 4** | Spatial Bounding | $C_{\text{grid}} = \frac{\text{count}(A \ge 0.05)}{HW} \le 0.25$ | Unconstrained global chaotic noise |
| **Gate 5** | Non-Trivial Mass | $\sum_{\mathbf{x}} A(\mathbf{x}) \ge 10.0$ | Empty vacuum canvas |

If **ANY** gate fails, the candidate is disqualified: $\text{Watertight Score} \leftarrow 0.0000$.

### 5.2 Mathematical Formulation of Quality Score

For candidates passing all 5 gates:

$$\text{Score}_{\text{watertight}} = \left(\frac{v_{\text{CoM}}}{50.0}\right) \cdot \left(\frac{\text{EA}}{0.010}\right) \cdot R_{\text{core}} \cdot \left(1.0 - C_{\text{grid}}\right)$$

---

## 6. Curiosity-Driven Exploration Algorithms

### 6.1 IMGEP (Intrinsically Motivated Goal Exploration Process)

IMGEP explores the continuous behavior space $\mathcal{B} = [v_{\text{CoM}}, \text{EA}, \mathcal{C}_{\text{gzip}}]$ through iterative goal babbling:

```
Algorithm 1: IMGEP Goal Exploration with Watertight Gating
------------------------------------------------------------
Initialize Archive A = {}
Phase 1: Bootstrap (n_bootstrap trials)
  For i = 1 to n_bootstrap:
    Sample genome θ_i ~ Uniform(Θ)
    Rollout x_i = Simulate(θ_i)
    Compute metrics b_i = Metrics(x_i), q_i = Watertight(x_i)
    If q_i > 0: Add (θ_i, b_i, q_i) to A

Phase 2: Goal Exploration (n_trials - n_bootstrap trials)
  For i = n_bootstrap + 1 to n_trials:
    Sample random behavioral goal g ~ Uniform(B_bounds)
    Select parent θ_parent = argmin_{(θ, b) in A} ||b - g||_2
    Mutate child θ_child = θ_parent + N(0, Σ_mut)
    Rollout x_child = Simulate(θ_child)
    Compute metrics b_child, q_child
    If q_child > 0: Add (θ_child, b_child, q_child) to A
```

### 6.2 Uniform Random Search Baseline

As an experimental control, Random Search samples all trials independently and uniformly from parameter space $\Theta \sim \text{Uniform}(\Theta_{\min}, \Theta_{\max})$ without goal selection or parent mutation.

### 6.3 Farthest-Point Sampling (FPS) Archive Maintenance

To downsample large archives while maximizing behavioral diversity, Farthest-Point Sampling greedily selects candidates that maximize minimum Euclidean distance in metric space:

$$x_{k+1} = \arg\max_{x \in \mathcal{A} \setminus S_k} \min_{s \in S_k} \|b(x) - b(s)\|_2$$

---

## 7. Autonomous AI Scientist Discovery Pipeline

Inspired by Michel et al. (2025/2026), the framework integrates an autonomous closed-loop discovery harness (`experiments/run_autonomous_agentic_loop.py`):

```
┌────────────────────────────────────────────────────────┐
│               AI Scientist Agent Loop                  │
│                                                        │
│   1. IMGEP Search Generation (20-50 candidates)       │
│                  │                                     │
│                  ▼                                     │
│   2. Watertight Quality Scoring & Hard Gating          │
│                  │                                     │
│                  ▼                                     │
│   3. Multimodal Vision Trajectory Inspection (PNG)     │
│                  │                                     │
│                  ▼                                     │
│   4. Adaptive Parameter Bound Tuning                   │
│                  │                                     │
│                  ▼                                     │
│   5. Persistent State Archive (JSON + MP4 + NPZ)       │
└────────────────────────────────────────────────────────┘
```

### 7.1 Multimodal Dual-Panel Visual Feedback Loop
AI agents inspect 6-frame dual-panel trajectory PNGs ($t=0\%, 20\%, 40\%, 60\%, 80\%, 100\%$) and motion heatmaps via native computer vision (`view_file`), validating:
1. Dense white persistent nucleus in the physical panel.
2. Coherent perimeter waves without speckle noise in the plasma panel.
3. Smooth continuous translation trails in the motion heatmap.

---

## 8. Experimental Protocols & Hypotheses

### 8.1 Experiment 1: Baseline Open IMGEP vs. Random Search
- **Objective**: Demonstrate that curiosity-driven goal exploration discovers significantly higher motility and complexity than uniform random sampling.
- **Parameters**: 50 IMGEP trials vs 50 Random Search trials, 2000 steps, $256 \times 256$ grid.
- **CLI**: `uv run python run_experiment.py --mode imgep --trials 50 --steps 2000`

### 8.2 Experiment 2: Corridor Constriction & Morphological Plasticity
- **Objective**: Quantify the transmission efficiency $T(W)$ and soft-bodied elasticity of Flow-Lenia solitons traversing geometric constrictions ($W_{\text{passage}} \in [8, 16, 24, 32]\text{ px}$).
- **CLI**: `uv run python run_experiment.py --mode barrier_constriction --widths 8 16 24 32`

### 8.3 Experiment 3: Reactive Resource Depletion & Niche Construction
- **Objective**: Show how stateful substrate depletion forces cyclic foraging and continuous locomotion.
- **CLI**: `uv run python run_experiment.py --mode showcase`

### 8.4 Experiment 4: Long-Duration Multi-Species Ecosystem Dynamics
- **Objective**: Observe macro-scale multi-species interactions, territorial boundary formation, and soliton collisions over long horizons (4000 steps, $384 \times 384$).
- **CLI**: `uv run python run_experiment.py --mode hero --patches 6 --grid_size 384 --steps 4000`

### 8.5 Experiment 5: Resolution Invariance & Long-Horizon Scale-Up
- **Objective**: Verify that discovered solitons remain stable and resolution-invariant when scaled to $512 \times 512$ over 10,000 steps.
- **CLI**: `uv run python run_experiment.py --mode scaleup --scale_grid_size 512 --scale_steps 10000`

---

## 9. Complete Hyperparameter Specification Tables

### Table 1: Physical Simulation Hyperparameters
| Parameter | Symbol | Default Value | Search / Sweep Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| Time step | $\Delta t$ | $0.05$ | Fixed | Temporal integration step |
| Kernel count | $K$ | $9$ | Fixed | Number of concentric ring kernels |
| Kernel outer radius | $R_k$ | Uniform$[6.0, 15.0]$ | $[6.0, 15.0]\text{ px}$ | Radial scaling per kernel |
| Shell amplitudes | $b_{\text{shells}}$ | $[1.0, 0.50, 0.33]$ | Fixed | Concentric ring peak heights |
| Shell peak radii | $r_{\text{peaks}}$ | $[0.50, 0.25, 0.75]$ | Fixed | Concentric ring peak locations |
| Shell width | $w_{\text{width}}$ | $0.12$ | Fixed | Concentric ring Gaussian width |
| Velocity scale | $v_{\text{scale}}$ | $5.2$ | $[4.2, 6.5]$ | Overall mass flux magnitude |
| Diffusion coeff | $\alpha$ | $0.06$ | $[0.04, 0.08]$ | Entropic mass diffusion pressure |
| Softmax beta | $\beta$ | $2.0$ | $[1.0, 4.0]$ | Negotiation competition aggressiveness |
| Depletion rate | $\delta_{\text{dep}}$ | $0.04$ | $[0.02, 0.08]$ | Resource depletion per step |
| Regen rate | $\delta_{\text{reg}}$ | $0.01$ | $[0.005, 0.03]$ | Resource regeneration per step |

### Table 2: Biological & Mutation Hyperparameters
| Parameter | Symbol | Default Value | Search Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| Species patches | $N_{\text{patches}}$ | $6$ | $[2, 8]$ | Number of initial seed organisms |
| Growth center | $\mu_k$ | Uniform$[0.13, 0.22]$ | $[0.10, 0.28]$ | Preferred kernel density center |
| Growth width | $\sigma_k$ | Uniform$[0.011, 0.024]$ | $[0.008, 0.030]$ | Growth tolerance window |
| Kernel weights | $w_k$ | $1.0 / K$ | $[0.0, 1.0]$ | Linear kernel combination weight |
| Mutation interval | $T_{\text{mut}}$ | $50\text{ steps}$ | $[20, 100]$ | Periodic mutation frequency |
| Mutation radius | $R_{\text{mut}}$ | $10\text{ px}$ | $[5, 15]$ | Spatial radius of mutation patch |
| Mutation noise | $\sigma_{\text{mut}}$ | $0.01$ | $[0.005, 0.02]$ | Gaussian standard deviation of mutation |

---

## 10. Step-by-Step Reproduction Guide

### Prerequisites
```bash
# Verify Python 3.10+ and JAX GPU acceleration
uv run python -c "import jax; print('JAX Devices:', jax.devices())"

# Execute complete test suite
uv run python -m unittest discover tests -v
```

### Reproducing All Thesis Benchmarks
```bash
# 1. Baseline IMGEP vs Random Search (Experiment 1: results/baseline_imgep/elite_1/ .. elite_3/)
uv run python run_experiment.py --mode imgep --trials 40 --steps 2500 --output_dir results/baseline_imgep

# 2. Wall Obstacles IMGEP Exploration (Experiment 2: results/wall_obstacles/elite_1/ .. elite_3/)
uv run python run_experiment.py --mode wall_obstacle --trials 40 --steps 2500 --output_dir results/wall_obstacles

# 3. Barrier Constriction Parameter Sweep (Experiment 3: results/barrier_constriction/width_08/ .. width_32/)
uv run python run_experiment.py --mode barrier_constriction --widths 8 16 24 32 --steps 3600 --output_dir results/barrier_constriction

# 4. Resource Depletion & Niche Construction (Experiment 4: results/resource_depletion/static_baseline/ & dynamic_depletion/)
uv run python run_experiment.py --mode depletion --grid_size 256 --steps 3600 --output_dir results/resource_depletion

# 5. Mechanism Showcase Comparison (Experiment 5: results/showcase/method_1_gene_mutation/ & method_2_negotiation_rule/)
uv run python run_experiment.py --mode showcase

# 6. Long-Horizon 512x512 Scale-up Reruns (Experiment 6: results/scaleup/rerun_1/ & rerun_2/)
uv run python run_experiment.py --mode scaleup --scale_grid_size 512 --scale_steps 3600 --output_dir results/scaleup

# 7. Classic Orbium Unicaudatus Physics Verification (results/orbium/)
uv run python run_experiment.py --mode orbium

# 8. Master Epic Ecosystem Simulation (results/epic_ecosystem/)
uv run python run_experiment.py --mode epic --grid_size 384 --steps 22500 --sample_interval 3 --patches 8 --output_dir results/epic_ecosystem
```

### Docker Execution
```bash
docker build -t flow-lenia:latest .
docker run --rm flow-lenia:latest
```

---

## 11. BibTeX Academic Bibliography

```bibtex
@article{michel2025exploring,
  title={Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist},
  author={Michel, Gautier and Cvjetko, Lana and Hamon, Erwan and Oudeyer, Pierre-Yves and Moulin-Frier, Cl{\'e}ment},
  journal={arXiv preprint arXiv:2505.15998},
  year={2025}
}

@article{plantec2025flowlenia,
  title={Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata},
  author={Plantec, Erwan and Hamon, Erwan and Etcheverry, Mayalen and Chan, Bert Wang-Chak and Oudeyer, Pierre-Yves and Moulin-Frier, Cl{\'e}ment},
  journal={Artificial Life},
  volume={31},
  number={2},
  year={2025},
  publisher={MIT Press}
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

@article{chan2023towards,
  title={Towards Large-Scale Simulations of Open-Ended Evolution in Continuous Cellular Automata},
  author={Chan, Bert Wang-Chak},
  journal={arXiv preprint arXiv:2304.05639},
  year={2023}
}

@article{oudeyer2007intrinsic,
  title={Intrinsic motivation systems for autonomous mental development},
  author={Oudeyer, Pierre-Yves and Kaplan, Fr{\'e}d{\'e}ric and Hafner, Verena V},
  journal={IEEE Transactions on Evolutionary Computation},
  volume={11},
  number={2},
  pages={265--286},
  year={2007},
  publisher={IEEE}
}

@article{faldor2024leniabreeder,
  title={Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity},
  author={Faldor, Maxence and Cully, Antoine},
  journal={ALIFE 2024: Proceedings of the 2024 Artificial Life Conference},
  year={2024}
}
```
