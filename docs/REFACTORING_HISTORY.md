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

## 6. Complete Clean Directory Hierarchy

```
results/
├── physics_verification/       # Mass conservation Q=0 verification
├── gene_mutation/              # Multi-species Gumbel-Max mixing (seed_42, seed_101, seed_2024)
├── negotiation_rule/           # Softmax growth negotiation (seed_42, seed_101, seed_2024)
├── baseline_imgep/             # IMGEP curiosity search vs Random (seed_42, seed_101, seed_2024)
├── agentic_loop/               # 138-gen AI Scientist loop (+392% gain, 108 elites)
├── chemotaxis_calibration/     # 3-Way Cohesion vs Fission phase transition (seed_42, seed_101, seed_2024)
├── barrier_constriction/       # Soft-bodied aperture transmission sweep (seed_42, seed_101, seed_2024)
├── wall_obstacles/             # Obstacle maze exploration (seed_42, seed_101, seed_2024)
├── resource_depletion/         # Cyclic foraging & trailing kinetics (seed_42, seed_101, seed_2024)
├── scaleup/                    # 512x512 canvas resolution invariance (seed_42, seed_101, seed_2024)
└── epic_ecosystem/             # Grand Synthesis Chemotactic Colosseum (seed_42, seed_101, seed_2024)
```
