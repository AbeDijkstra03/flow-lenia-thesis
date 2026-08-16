# Workspace Rules: Flow-Lenia Master's Thesis Framework

## Project Overview & Canon Alignment
This codebase implements a GPU-accelerated Flow-Lenia Open-Ended Evolution (OEE) simulation framework in native JAX, adhering strictly to the canonical literature:
- **Michel, Cvjetko, Hamon, Oudeyer, Moulin-Frier (2025, updated 2026)**: *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* (arXiv:2505.15998).
- **Plantec et al. (2025)**: *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, Artificial Life 31(2) (arXiv:2506.08569).

---

## Core Engineering Decisions (Non-Negotiable)

1. **JAX Hardware Acceleration**:
   - All physics convolutions must use FFT frequency-domain convolutions (`jnp.fft.rfft2` / `irfft2`) with precomputed kernel FFTs.
   - All simulation rollouts must be `jax.jit`-compiled and batched via `jax.vmap` across trials for RTX 5090 acceleration.

2. **Multi-Shell Concentric Ring Kernels**:
   - Continuous Gaussian ring kernels must use multi-shell concentric rings ($b_{\text{shells}} = [1.0, 0.5, 0.33]$, $r_{\text{peaks}} = [0.5, 0.25, 0.75]$, $r_{\text{width}} = 0.12$).
   - Multi-shell kernels generate self-organizing gliders, breathers, and dividing solitons.

3. **Spatially Localized Seeding (No Full-Grid Noise)**:
   - Simulations start with 2 to 6 spatially localized density patches in vacuum.
   - Each patch is initialized as a mixture of asymmetric Gaussian sub-blobs with directional density slope gradients ($1 + 0.40 \cdot \mathbf{k} \cdot \mathbf{x}$) to force an immediate spatial phase shift and non-zero velocity gradient ($\mathbf{v} = \nabla G(U) \neq 0$).

4. **Negative Growth Bounds ($G(U) < 0$)**:
   - Growth function $G(U)$ enforces negative growth ($G < 0$) for high densities $U > \mu + 1.2\sigma$, preventing central mass contraction ("melting into blobs").

5. **Stochastic Gene-Wise Sampling & Negotiation Rules**:
   - Parameter maps $(\boldsymbol{\mu}, \boldsymbol{\sigma}, \mathbf{w})$ are updated via Stochastic Gene-Wise Sampling (Gumbel-Max categorical sampling over incoming directional mass fluxes) or Negotiation Rule growth competition to prevent parameter blurring into inert averages.

6. **3-D Metric Space & IMGEP Curiosity Search**:
   - Primary metric space: `[Center of Mass Displacement (Motility), Non-Neutral Evolutionary Activity (Quadratic EA), Compression Complexity (gzip)]`.
   - IMGEP goal exploration is benchmarked against a uniform Random Search baseline with identical trial budgets.

7. **Multimodal Agent Feedback Loop & Dual-Panel Rendering**:
   - To visually inspect simulation rollouts, scripts extract 6 key trajectory frames ($t=0\%, 20\%, 40\%, 60\%, 80\%, 100\%$) plus a Motion Heatmap PNG (`motion_heatmap.png`).
   - Trajectory PNGs render as dual-panel images: Left = Plasma Colorized (wave structure), Right = Absolute Physical Scale ($[0.0, 1.0]$ density mapping).
   - AI agents use native vision capabilities via `view_file` on PNG files alongside exact numerical scores ($v_{\text{CoM}}$, $R_{\text{core}}$, $R_{\text{mass}}$) to evaluate rollout motility and collision dynamics.

8. **Watertight Quality Filter Enforcement (`core/metrics.py`)**:
   - All candidate evaluations MUST pass `evaluate_watertight_quality_score()`:
     - Mass Preservation Ratio ($R_{\text{mass}} \in [0.60, 5.00]$).
     - Solid Core Ratio ($R_{\text{core}} \ge 0.50$, density $A \ge 0.15$).
     - Motility ($v_{\text{CoM}} \ge 5.0\text{ px}$).
     - Grid Coverage ($C_{\text{grid}} \le 0.25$).
   - Disqualified candidates MUST receive score `0.0000`.

9. **Long-Running Autonomous Campaign Harness**:
   - AI models run multi-hour discovery campaigns via `uv run python experiments/run_autonomous_agentic_loop.py`.
   - State is logged to `results/agentic_loop_state.json` and summarized in `CONTEXT_HANDOFF.md`.
   - Recommend the `/goal` slash command when the user requests thorough overnight exploration.

10. **Visual-First Selection Protocol (Mandatory)**:
    - Numerical gates are necessary but NOT sufficient. Agents must prioritize dual-panel trajectory images and motion heatmaps for final selection.
    - Why:
       - Entropy/complexity can increase from noise and does not guarantee coherent morphology.
       - Mass preservation can hold for static or uninteresting attractors.
       - Velocity can be non-zero while dynamics remain trivial (wobble, jitter, short local drift).
    - Required visual checks per elite:
       - Persistent dense white core in right panel from 0% to 100% frame.
       - Coherent plasma perimeter in left panel (no collapse into speckle or hollow-only shells).
       - Motion heatmap shows structured translation trails, not only local flicker.
       - Prefer morphological diversity across accepted elites (avoid near-duplicates).

11. **Adaptive Tuning Lessons (Observed 2026-08-11)**:
    - Effective regime for stable, expressive visuals:
       - `v_scale` around 4.2 to 6.5
       - `alpha_diffusion` around 0.04 to 0.08
       - `n_patches` around 3 to 7
       - localized, moderately jittered angular seeding
    - Failure pattern:
       - Overly aggressive directional coupling (e.g., forced global tangential swirl hubs) can drive 5-generation runs to zero valid elites.
    - Recovery pattern:
       - Roll back to simpler randomized directional slopes and moderate proximity seeding, then iterate with small range adjustments.
