# Continuation Prompt for Future Agent

You are continuing a Flow-Lenia autonomous research campaign in this workspace.

## Mission

Continue searching for visually compelling, motile, solid gliders and interactive ecosystems in the JAX Flow-Lenia framework. Your primary objective is not to maximize any single scalar metric. Your objective is to discover configurations whose images and motion heatmaps are genuinely interesting, stable, and thesis-worthy.

## Hard Rules

1. Always read these files first:
   - `CONTEXT_HANDOFF.md`
   - `.agents/AGENTS.md`
   - `.agents/skills/flow_lenia_explorer/SKILL.md`

2. Treat metrics as gates, not as the final truth.
   - A candidate can pass the watertight filter and still be boring.
   - A candidate with high entropy or high velocity can still be uninteresting noise.
   - The final decision must be driven by the dual-panel frames and motion heatmap.

3. Use the current visual shortlist in `results/agentic_loop/visual_shortlist.md` as the baseline reference set.

4. Preserve only valid elites in `results/agentic_loop/agentic_loop_state.json` and keep `CONTEXT_HANDOFF.md` updated.

## What To Look At

For each promising generation, inspect:

- `gen_X_top1_frame_step_1_pct0.png`
- `gen_X_top1_frame_step_6_pct100.png`
- `gen_X_top1_motion_heatmap.png`

Use these visuals to judge:

- whether the right panel keeps a dense white core,
- whether the left panel preserves a coherent plasma boundary,
- whether the heatmap shows structured translation trails instead of only local flicker,
- whether the shape is distinct from prior elites.

## Working Hypotheses Worth Testing

The following are not guaranteed by the literature and should be treated as experimental hypotheses:

1. Slightly asymmetric multi-patch initialization may produce richer interaction geometry than symmetric ring layouts.
2. Radially biased or tangentially biased initial density slopes may create more expressive trails if the bias is moderate, but aggressive swirl can destroy valid yield.
3. A small increase in patch-count diversity can improve ecosystem-like interactions without collapsing solidity.
4. Visual novelty may improve when the sampling space changes gradually rather than through large jumps.
5. Novel improvements may come from seed geometry, initialization asymmetry, and selection pressure, not only from the physics parameters themselves.

## Known Good Ranges From Prior Runs

- `v_scale`: roughly `4.2` to `6.5`
- `alpha_diffusion`: roughly `0.04` to `0.08`
- `n_patches`: roughly `3` to `7`
- Avoid large coordinated tangent-hub forcing; it caused multiple 5-generation runs with zero valid elites.

## Known Failure Modes

- Pure metric chasing can converge to repetitive, visually dull gliders.
- Excessive directional coupling or over-engineered tangential seeding can eliminate valid elites entirely.
- Some valid elites are nearly duplicates; do not keep them all just because they pass filters.

## Research Direction Beyond the Existing Literature

You are explicitly encouraged to search for ideas that are not already spelled out in the canonical papers or the repository notes. Examples of useful directions:

- new seeding geometries,
- localized interaction rules,
- hybrid selection heuristics that prioritize image diversity,
- new ways to create non-trivial motion trails without losing solidity,
- controlled heterogeneity in patch sizes, slopes, or placement.

Do not claim novelty unless you actually test or observe it. Keep hypotheses separate from conclusions.

## Campaign Loop

1. Run a 5-generation campaign.
2. Inspect the latest valid elites visually.
3. Decide whether the images improved.
4. If yes, keep small changes and continue.
5. If no, roll back the last aggressive change and try a narrower variant.
6. Update `CONTEXT_HANDOFF.md` with what improved, what regressed, and why.

## Stop Criterion

Stop only when the newest campaign block produces a set of visually distinct, solid, motile elites whose frames look clearly better than the current shortlist in `results/agentic_loop/visual_shortlist.md`.
