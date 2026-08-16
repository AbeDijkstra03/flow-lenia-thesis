# AI Assistant & Copilot Instructions: Flow-Lenia Framework

This repository implements a GPU-accelerated Flow-Lenia continuous cellular automata research framework in native JAX.

---

## Canonical Rules & Documentation
To ensure alignment with the thesis architecture and literature (Michel et al. 2025/2026, Plantec et al. 2025):
- **Core Rules & Constraints**: See [.agents/AGENTS.md](file:///home/abedijkstra/Documents/Scriptie/.agents/AGENTS.md) for non-negotiable physics rules (FFT convolutions, mass conservation, watertight filters).
- **Active Agent State & Memory**: See [.agents/CONTEXT_HANDOFF.md](file:///home/abedijkstra/Documents/Scriptie/.agents/CONTEXT_HANDOFF.md) for cross-session state persistence.
- **Mathematical Specification**: See [docs/THESIS_ARCHITECTURE.md](file:///home/abedijkstra/Documents/Scriptie/docs/THESIS_ARCHITECTURE.md).
- **Literature Review & Canon**: See [docs/LITERATURE_READING_LIST.md](file:///home/abedijkstra/Documents/Scriptie/docs/LITERATURE_READING_LIST.md).

---

## Quick Reference CLI
```bash
# Run automated test suite
uv run python -m unittest discover tests

# IMGEP vs Random baseline
uv run python run_experiment.py --mode imgep --trials 40 --steps 2500

# IMGEP with wall obstacles
uv run python run_experiment.py --mode wall_obstacle --trials 40 --steps 2500

# Corridor constriction parameter sweep
uv run python run_experiment.py --mode barrier_constriction --widths 8 16 24 32

# Resource depletion study
uv run python run_experiment.py --mode depletion

# Master 5-minute Epic Ecosystem with obstacles
uv run python run_experiment.py --mode epic --steps 22500
```
