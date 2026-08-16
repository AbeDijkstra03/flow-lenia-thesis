# Engineering History: PyTorch to JAX Flow-Lenia Modernization

This document records the architectural evolution, critical physics debugging, and engineering milestones of the Flow-Lenia Master's Thesis research framework.

---

## 1. Motivation: From Legacy PyTorch to Native JAX

### The Bottleneck in PyTorch
The initial exploratory codebase was written in PyTorch with manual spatial loops, CPU-bound contiguous component breadth-first searches (BFS), and legacy MAP-Elites grids. On high-resolution grids ($256 \times 256$ to $512 \times 512$), rollouts took over 45 seconds per trial, making multi-generation evolutionary searches computationally prohibitive.

### The JAX Transformation
We completely re-engineered the core simulation engine into native, functional JAX (`core/flow_lenia_jax.py`):
1. **FFT-Accelerated Convolutions**: Replaced spatial kernel loops with 2D real Fast Fourier Transforms (`jnp.fft.rfft2` / `irfft2`) utilizing precomputed frequency-domain kernel tensors ($\widehat{K}_k$).
2. **`jax.lax.scan` Rollouts**: Entire 2,000–10,000 step simulation rollouts execute as a single compiled XLA graph without Python interpreter overhead.
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

## 4. Summary of Modern Repository Layout

| Component | Modern File Location | Purpose |
| :--- | :--- | :--- |
| **Physics Engine** | `core/flow_lenia_jax.py` | JAX circular convolutions, advection fluxes, mixing rules |
| **3D Metrics Suite** | `core/metrics.py` | Motility, EA, Complexity, Solidity, Watertight Quality Filter |
| **Curiosity Search** | `core/imgep.py` | IMGEP archive, goal sampling, mutation, FPS selection |
| **Environments** | `core/environment.py` | Static barriers, corridor passages, resource fields |
| **Visualization** | `core/visualization.py` | H.264 MP4, 6-frame filmstrips, motion heatmaps |
| **Configuration** | `core/config.py` | Typed dataclass YAML config loader |
| **Constriction Experiment** | `experiments/run_barrier_constriction.py` | Corridor passage sweep & transmission curves |
| **Scale-up Reruns** | `experiments/run_scaleup.py` | 512x512 long-horizon FPS scaleup reruns |
| **Agentic Loop** | `experiments/run_autonomous_agentic_loop.py` | Multi-generation AI Scientist discovery loop |
| **Unified CLI** | `run_experiment.py` | Top-level CLI orchestrator |
