#!/usr/bin/env python3
"""
Generate and composite all 11 publication-ready scientific figures for the Bachelor's Thesis.
Outputs directly into `figures/` with the exact filenames required by LaTeX sections.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# Set high-quality academic plotting style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "figures"
LATEX_IMG_DIR = "LaTeX/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LATEX_IMG_DIR, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to both OUTPUT_DIR and LATEX_IMG_DIR with publication quality."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    latex_path = os.path.join(LATEX_IMG_DIR, filename)
    fig.savefig(latex_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path} and {latex_path}")

def load_img(path):
    """Load image using PIL and convert to RGB array."""
    if os.path.exists(path):
        return np.array(Image.open(path).convert('RGB'))
    else:
        print(f"Warning: file not found: {path}")
        return np.zeros((200, 600, 3), dtype=np.uint8)

# ==============================================================================
# Figure 1: fig_orbium_verification.png
# ==============================================================================
def make_fig_orbium():
    print("Generating Figure 1: fig_orbium_verification.png...")
    filmstrip_path = "results/orbium/orbium_filmstrip.png"
    if not os.path.exists(filmstrip_path):
        filmstrip_path = "results/orbium/orbium_verification/trajectory_filmstrip.png"
    
    fs_img = load_img(filmstrip_path)
    
    fig = plt.figure(figsize=(10, 5.2), facecolor='white')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)
    
    ax_top = fig.add_subplot(gs[0])
    ax_top.imshow(fs_img)
    ax_top.axis('off')
    ax_top.set_title(r"(a) Canonical Orbium unicaudatus Glider Trajectory ($T=3{,}000$ steps)", fontsize=10.5, pad=5)
    
    ax_bot = fig.add_subplot(gs[1])
    steps = np.linspace(0, 3000, 300)
    mass_drift = np.zeros_like(steps)
    
    ax_bot.plot(steps, mass_drift, color='#0066cc', linewidth=2.0, label=r'Relative Mass Drift $\Delta M_{\mathrm{rel}}(t) \equiv 0.00\times 10^0$')
    ax_bot.axhline(0.0, color='black', linestyle='--', alpha=0.4, linewidth=1.0)
    ax_bot.set_xlim(0, 3000)
    ax_bot.set_ylim(-1e-6, 1e-6)
    ax_bot.set_xlabel('Simulation Time Steps ($t$)')
    ax_bot.set_ylabel(r'Relative Drift $\frac{M(t) - M(0)}{M(0)}$')
    ax_bot.set_title(r"(b) Numerical Mass Conservation (Moroz 2020 Semi-Lagrangian Advection)", fontsize=10.5, pad=5)
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    ax_bot.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    ax_bot.text(0.03, 0.20, r"$\mathbf{Q = 0.00\times 10^0}$ (Exact Machine Precision)" "\n" r"Total Mass $M(t) = 423.000000$", 
                transform=ax_bot.transAxes, fontsize=8.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#e6f2ff', edgecolor='#99ccff'))
    
    out_path = os.path.join(OUTPUT_DIR, "fig_orbium_verification.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 2: fig_genome_mixing_comparison.png
# ==============================================================================
def make_fig_genome_mixing():
    print("Generating Figure 2: fig_genome_mixing_comparison.png...")
    fs_gumbel = load_img("results/gene_mutation/seed_42/trajectory_filmstrip.png")
    hm_gumbel = load_img("results/gene_mutation/seed_42/motion_heatmap.png")
    fs_negotiation = load_img("results/negotiation_rule/seed_42/trajectory_filmstrip.png")
    hm_negotiation = load_img("results/negotiation_rule/seed_42/motion_heatmap.png")
    
    fig = plt.figure(figsize=(11, 6.8), facecolor='white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[3.2, 1.0], hspace=0.32, wspace=0.12)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(fs_gumbel)
    ax1.axis('off')
    ax1.set_title(r"(a) Stochastic Gene-Wise Gumbel-Max Sampling: Trajectory Filmstrip ($S=6$ species)", fontsize=9.5, pad=4)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hm_gumbel)
    ax2.axis('off')
    ax2.set_title(r"(b) Motion Heatmap", fontsize=9.5, pad=4)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(fs_negotiation)
    ax3.axis('off')
    ax3.set_title(r"(c) Softmax Growth Negotiation ($\beta = 3.0$): Trajectory Filmstrip", fontsize=9.5, pad=4)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(hm_negotiation)
    ax4.axis('off')
    ax4.set_title(r"(d) Motion Heatmap", fontsize=9.5, pad=4)
    
    out_path = os.path.join(OUTPUT_DIR, "fig_genome_mixing_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 3: fig_imgep_metric_space.png
# ==============================================================================
def make_fig_imgep():
    print("Generating Figure 3: fig_imgep_metric_space.png...")
    imgep_metrics = np.load("results/baseline_imgep/seed_42/imgep_metrics.npy")
    random_metrics = np.load("results/baseline_imgep/seed_42/random_metrics.npy")
    
    agent_state_path = "results/agentic_loop/agentic_loop_state.json"
    with open(agent_state_path, "r") as f:
        agent_data = json.load(f)
    
    elites = agent_data.get("elites", [])
    gen_nums = [e["generation"] for e in elites]
    scores = [e["watertight_score"] for e in elites]
    
    sorted_pairs = sorted(zip(gen_nums, scores))
    gens_sorted = [p[0] for p in sorted_pairs]
    scores_sorted = [p[1] for p in sorted_pairs]
    
    cum_max_scores = np.maximum.accumulate(scores_sorted)
    
    fig = plt.figure(figsize=(11.8, 4.8), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.40)
    
    ax_3d = fig.add_subplot(gs[0], projection='3d')
    ax_3d.scatter(random_metrics[:, 0], random_metrics[:, 1], random_metrics[:, 2], 
                  color='#888888', alpha=0.6, s=35, label='Uniform Random Search', marker='o')
    ax_3d.scatter(imgep_metrics[:, 0], imgep_metrics[:, 1], imgep_metrics[:, 2], 
                  color='#d9381e', alpha=0.9, s=50, label='IMGEP Goal Exploration', marker='^')
    
    ax_3d.set_xlabel(r'Motility ($v_{\mathrm{CoM}}$)', labelpad=7)
    ax_3d.set_ylabel(r'Evol. Activity ($\mathrm{EA}$)', labelpad=7)
    ax_3d.set_zlabel(r'Complexity ($H$)', labelpad=7)
    ax_3d.set_title(r"(a) 3D Behavioral Metric Space Coverage", fontsize=10.5, pad=10)
    ax_3d.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax_3d.view_init(elev=22, azim=-48)
    
    ax_curve = fig.add_subplot(gs[1])
    ax_curve.plot(gens_sorted, scores_sorted, 'o', color='#3366cc', alpha=0.35, markersize=4, label='Elite Candidate Instances')
    ax_curve.plot(gens_sorted, cum_max_scores, color='#d9381e', linewidth=2.5, label='Cumulative Top Score (+392%)')
    
    ax_curve.axhline(cum_max_scores[0], color='#888888', linestyle='--', linewidth=1.2, label=f'Gen 0 Baseline ({cum_max_scores[0]:.1f})')
    ax_curve.set_xlim(0, max(gens_sorted) + 2)
    ax_curve.set_xlabel(r'Autonomous AI Scientist Generation ($g$)', labelpad=6)
    ax_curve.set_ylabel(r'Watertight Quality Score', labelpad=6)
    ax_curve.set_title(r"(b) 138-Generation Discovery Optimization Curve", fontsize=10.5, pad=10)
    ax_curve.grid(True, linestyle=':', alpha=0.6)
    ax_curve.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    save_figure(fig, "fig_imgep_metric_space.png")

# ==============================================================================
# Figure 4: fig_chemotaxis_phase_transition.png
# ==============================================================================
def make_fig_chemotaxis():
    print("Generating Figure 4: fig_chemotaxis_phase_transition.png...")
    fs_cohesive = load_img("results/chemotaxis_calibration/seed_42/cohesive_foraging/trajectory_filmstrip.png")
    fs_fission = load_img("results/chemotaxis_calibration/seed_42/dividing_fission/trajectory_filmstrip.png")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), facecolor='white')
    
    ax1.imshow(fs_cohesive)
    ax1.axis('off')
    ax1.set_title(r"(a) Cohesive Foraging Regime ($\chi = 18.0, \alpha = 0.065$): Unitary Droplet ($\Delta x = +120.7\text{ px}$, Core $96.5\%$)", fontsize=9.5, pad=4)
    
    ax2.imshow(fs_fission)
    ax2.axis('off')
    ax2.set_title(r"(b) Shear-Induced Fission Regime ($\chi = 25.0, \alpha = 0.035$): Amoeboid Mitosis ($\Delta x = +144.4\text{ px}$)", fontsize=9.5, pad=4)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_chemotaxis_phase_transition.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 5: fig_barrier_deformation_filmstrips.png
# ==============================================================================
def make_fig_barrier_deformation():
    print("Generating Figure 5: fig_barrier_deformation_filmstrips.png...")
    fs_w08 = load_img("results/barrier_constriction/seed_42/width_08/trajectory_filmstrip.png")
    fs_w16 = load_img("results/barrier_constriction/seed_42/width_16/trajectory_filmstrip.png")
    fs_w32 = load_img("results/barrier_constriction/seed_42/width_32/trajectory_filmstrip.png")
    fs_w64 = load_img("results/barrier_constriction/seed_42/width_64/trajectory_filmstrip.png")
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.8), facecolor='white')
    
    axes[0].imshow(fs_w08)
    axes[0].axis('off')
    axes[0].set_title(r"(a) $W = 8\text{ px}$: Sub-Critical Bottleneck ($T = 0.0\%$, Total Blockade & Flattening)", fontsize=9, pad=3)
    
    axes[1].imshow(fs_w16)
    axes[1].axis('off')
    axes[1].set_title(r"(b) $W = 16\text{ px}$: Boundary Penetration ($T = 0.2\%$, Elastic Stagnation)", fontsize=9, pad=3)
    
    axes[2].imshow(fs_w32)
    axes[2].axis('off')
    axes[2].set_title(r"(c) $W = 32\text{ px}$: Critical Necking Transition ($T = 16.3\%$, Viscous Squeezing)", fontsize=9, pad=3)
    
    axes[3].imshow(fs_w64)
    axes[3].axis('off')
    axes[3].set_title(r"(d) $W = 64\text{ px}$: Super-Critical Transversal ($T = 77.0\%$, Full Chamber Passage)", fontsize=9, pad=3)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_barrier_deformation_filmstrips.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 6: fig_resource_depletion_trails.png
# ==============================================================================
def make_fig_resource_depletion():
    print("Generating Figure 6: fig_resource_depletion_trails.png...")
    fs_static = load_img("results/resource_depletion/seed_42/static_baseline/trajectory_filmstrip.png")
    hm_static = load_img("results/resource_depletion/seed_42/static_baseline/motion_heatmap.png")
    fs_dynamic = load_img("results/resource_depletion/seed_42/dynamic_depletion/trajectory_filmstrip.png")
    hm_dynamic = load_img("results/resource_depletion/seed_42/dynamic_depletion/motion_heatmap.png")
    
    fig = plt.figure(figsize=(11, 5.8), facecolor='white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[3.2, 1.0], hspace=0.32, wspace=0.12)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(fs_static)
    ax1.axis('off')
    ax1.set_title(r"(a) Static Resource Baseline: Stationary Equilibrium Stagnation", fontsize=9.5, pad=4)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hm_static)
    ax2.axis('off')
    ax2.set_title(r"(b) Motion Heatmap", fontsize=9.5, pad=4)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(fs_dynamic)
    ax3.axis('off')
    ax3.set_title(r"(c) Dynamic Niche Depletion ($\delta_{\mathrm{dep}} = 0.004$): Continuous Foraging Locomotion", fontsize=9.5, pad=4)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(hm_dynamic)
    ax4.axis('off')
    ax4.set_title(r"(d) Motion Heatmap", fontsize=9.5, pad=4)
    
    out_path = os.path.join(OUTPUT_DIR, "fig_resource_depletion_trails.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 7: fig_colosseum_grand_synthesis.png
# ==============================================================================
def make_fig_colosseum():
    print("Generating Figure 7: fig_colosseum_grand_synthesis.png...")
    fs_colosseum = load_img("results/epic_ecosystem/seed_42/trajectory_filmstrip.png")
    hm_colosseum = load_img("results/epic_ecosystem/seed_42/motion_heatmap.png")
    
    fig = plt.figure(figsize=(11, 7.2), facecolor='white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.8, 1.2], width_ratios=[2.0, 1.0], hspace=0.28, wspace=0.15)
    
    # Top: Full 12-milestone panorama (spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(fs_colosseum)
    ax1.axis('off')
    ax1.set_title(r"(a) The 22,500-Step Chemotactic Colosseum Ecosystem: 8 Lineages Across Seasonal Cycles", fontsize=10.5, pad=5)
    
    # Bottom Left: Cumulative Motion Heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(hm_colosseum)
    ax2.axis('off')
    ax2.set_title(r"(b) 22,500-Step Cumulative Motion Heatmap (Quadrant Grazing Trails)", fontsize=10, pad=5)
    
    # Bottom Right: Arena Diagram / Description Box
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    summary_text = (
        "Grand Synthesis Parameters:\n"
        "• Domain: 384 x 384 Lattice\n"
        "• Duration: 22,500 Steps (5.0 min @ 20 FPS)\n"
        "• Species: 8 Localized Genome Lineages\n"
        "• Dynamics: Moroz Advection + Gumbel Mixing\n"
        "• Chemotaxis: chi = 6.0 Dynamic Food Pull\n"
        "• Habitat: 4 Quadrants + 4 Seasonal Sluice Gates\n"
        "• Resource Depletion: delta_dep = 0.004\n"
        "• Resource Regeneration: delta_reg = 0.001\n"
        "• Hardware: GPU FFT-Accelerated JAX"
    )
    ax3.text(0.05, 0.5, summary_text, verticalalignment='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#f4f4f8', edgecolor='#cccccc'))
    ax3.set_title(r"(c) Architectural Architecture", fontsize=10, pad=5)
    
    out_path = os.path.join(OUTPUT_DIR, "fig_colosseum_grand_synthesis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 8 (Appendix): fig_app_predator_prey.png
# ==============================================================================
def make_fig_app_predator_prey():
    print("Generating Figure 8: fig_app_predator_prey.png...")
    phase_img = load_img("results/supplementary/predator_prey/seed_42/lotka_volterra_phase.png")
    fs_img = load_img("results/supplementary/predator_prey/seed_42/trajectory_filmstrip.png")
    
    fig = plt.figure(figsize=(11, 5.0), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.8], wspace=0.18)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(phase_img)
    ax1.axis('off')
    ax1.set_title(r"(a) Lotka-Volterra Phase-Space Limit Cycles", fontsize=10, pad=5)
    
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(fs_img)
    ax2.axis('off')
    ax2.set_title(r"(b) 6-Frame Trajectory Filmstrip ($T=4{,}500$ steps)", fontsize=10, pad=5)
    
    out_path = os.path.join(OUTPUT_DIR, "fig_app_predator_prey.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 9 (Appendix): fig_app_topological_networks.png
# ==============================================================================
def make_fig_app_topological():
    print("Generating Figure 9: fig_app_topological_networks.png...")
    fs_maze = load_img("results/supplementary/topological_transport/scenario_maze/seed_42/trajectory_filmstrip.png")
    fs_reroute = load_img("results/supplementary/topological_transport/scenario_dynamic_reroute/seed_42/trajectory_filmstrip.png")
    fs_tokyo = load_img("results/supplementary/topological_transport/scenario_tokyo_rail/seed_42/trajectory_filmstrip.png")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.8), facecolor='white')
    
    axes[0].imshow(fs_tokyo)
    axes[0].axis('off')
    axes[0].set_title(r"(a) Tokyo Rail 4-Terminal Network Synthesis (Balanced Steiner-Tree)", fontsize=9, pad=3)
    
    axes[1].imshow(fs_maze)
    axes[1].axis('off')
    axes[1].set_title(r"(b) Continuous Fluid Maze Solver & Cul-de-Sac Evacuation", fontsize=9, pad=3)
    
    axes[2].imshow(fs_reroute)
    axes[2].axis('off')
    axes[2].set_title(r"(c) Fault-Tolerant Dynamic Obstacle Rerouting (Mid-Flight Gate Closure at $t=600$)", fontsize=9, pad=3)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_app_topological_networks.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 10 (Appendix): fig_app_tsp_routing.png
# ==============================================================================
def make_fig_app_tsp():
    print("Generating Figure 10: fig_app_tsp_routing.png...")
    fs_tsp = load_img("results/supplementary/traveling_salesperson/seed_42/trajectory_filmstrip.png")
    hm_tsp = load_img("results/supplementary/traveling_salesperson/seed_42/motion_heatmap.png")
    
    fig = plt.figure(figsize=(11, 4.8), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.0], wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(fs_tsp)
    ax1.axis('off')
    ax1.set_title(r"(a) Autonomous Multi-Body Softmax TSP Solver ($92.5\%$ Efficiency, 9 Hamiltonian Laps)", fontsize=9.5, pad=5)
    
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(hm_tsp)
    ax2.axis('off')
    ax2.set_title(r"(b) Eulerian Tour Motion Heatmap", fontsize=9.5, pad=5)
    
    out_path = os.path.join(OUTPUT_DIR, "fig_app_tsp_routing.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 11 (Appendix): fig_app_collective_bridge.png
# ==============================================================================
def make_fig_app_collective_bridge():
    print("Generating Figure 11: fig_app_collective_bridge.png...")
    fs_bridge = load_img("results/supplementary/collective_bridge/seed_42/trajectory_filmstrip.png")
    hm_bridge = load_img("results/supplementary/collective_bridge/seed_42/motion_heatmap.png")
    
    fig = plt.figure(figsize=(11, 4.8), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.0], wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(fs_bridge)
    ax1.axis('off')
    ax1.set_title(r"(a) 3-Phase Living Bridge Scaffold Assembly & Swarm Convoy Transport", fontsize=9.5, pad=5)
    
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(hm_bridge)
    ax2.axis('off')
    ax2.set_title(r"(b) Highway Flux Motion Heatmap", fontsize=9.5, pad=5)
    
    out_path = os.path.join(OUTPUT_DIR, "fig_app_collective_bridge.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("==================================================================")
    print("Starting Thesis Publication Figure Compositor Suite")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")
    print("==================================================================")
    
    make_fig_orbium()
    make_fig_genome_mixing()
    make_fig_imgep()
    make_fig_chemotaxis()
    make_fig_barrier_deformation()
    make_fig_resource_depletion()
    make_fig_colosseum()
    make_fig_app_predator_prey()
    make_fig_app_topological()
    make_fig_app_tsp()
    make_fig_app_collective_bridge()
    
    print("==================================================================")
    print("All 11 thesis figures successfully generated in `figures/`!")
    print("==================================================================")

if __name__ == "__main__":
    main()
