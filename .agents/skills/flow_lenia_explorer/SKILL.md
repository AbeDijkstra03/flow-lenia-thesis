---
name: flow_lenia_explorer
description: Instructions and workflow for AI agents (Antigravity & Copilot) to run autonomous, multi-hour IMGEP goal exploration, watertight feedback filtering, visual trajectory inspection, and context persistence.
---

# Flow-Lenia Autonomous Explorer Skill

This skill guides AI agents through orchestrating autonomous, long-running Flow-Lenia research campaigns using JAX hardware acceleration, watertight quality filtering, visual trajectory inspection, and persistent state handoffs.

Core principle: treat metrics as guard rails and images as the final scientific evidence.

---

## 1. Execution Workflow for Autonomous AI Campaigns

When instructed to perform continuous exploration, find new species, or optimize Flow-Lenia ecosystems:

### Phase 1: Context & Memory Loading
1. Read [CONTEXT_HANDOFF.md](file:///home/abedijkstra/Documents/Scriptie/CONTEXT_HANDOFF.md) to load current baseline state and elite history.
2. Check if `results/agentic_loop/agentic_loop_state.json` exists. If so, read elite history and recent generation outcomes.

### Phase 2: Launch Autonomous Agentic Harness
Run the multi-generation search engine:
```bash
uv run python experiments/run_autonomous_agentic_loop.py --generations 5 --trials_per_gen 25 --steps 3000 --output_dir results/agentic_loop
```

### Phase 3: Watertight Visual & Numerical Inspection
1. Read `results/agentic_loop/agentic_loop_state.json` and identify newest elite generations.
2. Locate extracted dual-panel PNG trajectory frames in `results/agentic_loop/frames/`:
   - `gen_X_top1_frame_step_1_pct0.png`
   - `gen_X_top1_frame_step_6_pct100.png`
   - `gen_X_top1_motion_heatmap.png`
3. If a generation has no valid elite, these files can be absent. Do not treat missing files as an error.
4. Call `view_file` on PNG files to visually inspect:
   - **Left Panel**: Wave perimeter structure and multi-shell concentric ring dynamics.
   - **Right Panel**: Absolute physical density distribution ($[0.0, 1.0]$) verifying a solid white core.
   - **Heatmap**: Translation trails showing continuous glider motion.
5. Apply visual-first ranking before trusting score order:
   - Reject pure flicker/noise trails even if metrics pass.
   - Reject near-duplicate elites with nearly identical morphology and heatmap shape.
   - Prefer elites with coherent core retention plus non-trivial trajectory geometry.

### Phase 4: Parameter Domain Adaptation
Based on watertight status scores:
- If candidates trigger `MASS_DISSIPATED`, increase mass concentration diffusion coupling $\alpha_{\text{diff}}$ in `core/imgep.py` towards $0.06 - 0.08$.
- If candidates trigger `FROZEN_STILL_LIFE`, increase velocity scaling $v_{\text{scale}}$ towards $4.5 - 6.5$ or increase asymmetric directional density slope gradient ($1.0 + 1.0 \cdot \mathbf{k} \cdot \mathbf{x}$).
- If candidates trigger `HOLLOW_OUTLINE_DEGENERATION`, adjust Gaussian affinity parameters $(\mu \in [0.14, 0.18], \sigma \in [0.012, 0.018])$.

Also adapt from visual evidence:
- If heatmaps are too compact/local despite valid metrics, widen seeding radius span moderately.
- If shape diversity collapses into repeated templates, increase angular jitter and patch-count variation.
- If five full generations produce zero valid elites after a tuning change, roll back the last aggressive change.

### Phase 5: Handoff & Memory Update
1. Update `results/agentic_loop/agentic_loop_state.json` with persisted elites.
2. Log summary in `CONTEXT_HANDOFF.md` with:
   - campaign command and generation range
   - visual findings per top elite set
   - what tuning was changed, what failed, and what recovered
   - explicit reason to continue or stop campaigns

### Stop Condition for "Top" Imagery

Continue campaigns until most recent generation block yields all of:
- at least 2 valid elites with clearly distinct morphology,
- persistent solid cores at 100% frame,
- motion heatmaps with structured non-trivial trails (not only local jitter),
- no obvious visual regression versus previous best set.

---

## 2. Recommended Slash Commands

- Use `/goal` when initiating an overnight or multi-hour exploration task so the agent executes continuous search iterations without stopping.
- Use `/learn` if specific hyperparameter bounds prove exceptionally motile to persist the discovery for future conversations.
