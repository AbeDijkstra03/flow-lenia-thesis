# AI Assistant & Copilot Instructions: Flow-Lenia Research Framework

This repository implements a GPU-accelerated Flow-Lenia Open-Ended Evolution (OEE) research framework in native JAX, adhering strictly to the canonical literature:
- **Michel et al. (2025/2026)**: *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* (arXiv:2505.15998).
- **Plantec et al. (2025)**: *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, Artificial Life 31(2) (arXiv:2506.08569).

---

## 1. Core Engineering & Physics Principles

1. **JAX Hardware Acceleration**:
   - Convolutions use FFT frequency-domain operations (`jnp.fft.rfft2` / `irfft2`) with precomputed kernel FFTs.
   - All physics convolutions and rollouts are `jax.jit`-compiled.
2. **Multi-Shell Concentric Ring Kernels**:
   - Gaussian ring kernels use multi-shell concentric rings ($b_{\text{shells}} = [1.0, 0.5, 0.33]$, $r_{\text{peaks}} = [0.5, 0.25, 0.75]$, $r_{\text{width}} = 0.12$).
3. **Mass Conservation**:
   - Flux advection is normalized by `scale = 1.0 / max(1.0, v_sum)`, ensuring exact machine-precision mass preservation ($100.000\%$).
4. **Mixing Rules**:
   - Stochastic Gene-Wise Sampling (Gumbel-Max) or Continuous Negotiation Rule ($\text{softmax}(\beta G)$) are used to prevent parameter blurring into inert averages.
5. **Watertight Quality Filter (`core/metrics.py`)**:
   - Mass preservation ($R_{\text{mass}} \in [0.60, 5.00]$).
   - Solid core ratio ($R_{\text{core}} \ge 0.50$ for density $A \ge 0.15$).
   - Net motility ($v_{\text{CoM}} \ge 5.0\text{ px}$).
   - Disqualified candidates MUST receive score `0.0000`.

---

## 2. Directory Layout

- `core/`: Physics (`flow_lenia_jax.py`), Metrics (`metrics.py`), IMGEP (`imgep.py`), Config (`config.py`), Visualization (`visualization.py`), Environment (`environment.py`).
- `experiments/`:
  - `run_imgep_search.py`: Baseline IMGEP curiosity search vs. Random search.
  - `run_barrier_constriction.py`: Corridor width sweep & transmission efficiency experiment.
  - `run_scaleup.py`: 512x512 scaled-up FPS reruns (10,000 steps).
  - `run_autonomous_agentic_loop.py`: AI Scientist autonomous discovery loop.
- `scripts/`: Standalone utilities (`spawn_orbium.py`, `run_hero_ecosystem.py`, `run_showcase_methods.py`).
- `configs/`: Typed YAML configurations (`baseline.yaml`, `wall_obstacle.yaml`, `barrier_constriction.yaml`, `resource_depletion.yaml`, `scaleup.yaml`).
- `docs/`: Literature reading list, architecture specifications, refactoring logs.
- `results/`: Categorized simulation outputs (`baseline_imgep/`, `wall_obstacles/`, `barrier_constriction/`, `hero_ecosystems/`, `showcase/`, `agentic_loop/`).
- `run_experiment.py`: Main unified root CLI runner.

---

## 3. Standard Commands

```bash
# Run unit tests
uv run python -m unittest discover tests

# Launch baseline search
uv run python run_experiment.py --mode imgep --trials 50 --steps 2000

# Launch barrier constriction sweep
uv run python run_experiment.py --mode barrier_constriction --widths 8 16 24 32

# Launch hero ecosystem
uv run python run_experiment.py --mode hero --patches 6 --grid_size 384 --steps 4000
```
