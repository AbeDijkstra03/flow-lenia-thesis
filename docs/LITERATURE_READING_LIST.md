# Flow-Lenia Literature Reading List & Essential Context

This document preserves the canonical literature reading list and theoretical context for all AI agents working on the Flow-Lenia Open-Ended Evolution (OEE) framework.

---

## Literature Reading List (Ordered by Priority)

1. **Michel, Cvjetko, Hamon, Oudeyer, Moulin-Frier (2025, updated 2026)**
   - *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist* — arXiv:2505.15998.
   - **Key Focus**: Highest priority match. Covers mixing-rule ablation, IMGEP curiosity exploration algorithm, 3-D metrics (CoM motility, Evolutionary Activity, compression complexity), obstacle experiments, and scaling studies.
   - **Companion Visualizer**: [Flow-Lenia Universes Journal](https://developmentalsystems.org/Flow-Lenia-Universes-Journal/)

2. **Plantec, Hamon, Etcheverry, Chan, Oudeyer, Moulin-Frier (2025)**
   - *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata*, Artificial Life 31(2) — arXiv:2506.08569. (Original ALIFE 2023: arXiv:2212.07906).
   - **Key Focus**: The canonical Flow-Lenia reference introducing mass conservation $\nabla \cdot (A \mathbf{v})$ to Lenia continuous CA.
   - **Code/Videos**: [Flow-Lenia Site](https://sites.google.com/view/flowlenia/) | [Official GitHub](https://github.com/erwanplantec/FlowLenia)

3. **Faldor & Cully (2024)**
   - *Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity ("Leniabreeder")*, ALIFE 2024 — arXiv:2406.04235.
   - **Key Focus**: Restricted-genotype + iso+lineDD variation operator approach for generating highly polished individual gliders and creatures.
   - **Code**: [Leniabreeder GitHub](https://github.com/maxencefaldor/Leniabreeder)

4. **Papadopoulos & Guichard (2025)**
   - *MaCE: General Mass Conserving Dynamics for Cellular Automata*, ISAL 2025 — arXiv:2507.12306.
   - **Key Focus**: Alternative/simpler mass-conservation scheme with empirical comparisons to Flow-Lenia.

5. **Kumar, Lu, Kirsch, Tang, Stanley, Isola, Ha (2024/2025)**
   - *Automating the Search for Artificial Life with Foundation Models ("ASAL")* — arXiv:2412.17799.
   - **Key Focus**: VLM-guided search on Lenia continuous CA.
   - **Project Site**: [ASAL Sakana AI](https://asal.sakana.ai/)

6. **Faust et al. (2025)**
   - *Expedition & Expansion: Leveraging Semantic Representations for Goal-Directed Exploration in Continuous Cellular Automata* — arXiv:2509.03863.
   - **Key Focus**: Semantic representation & VLM goal exploration specifically on Flow-Lenia.

7. **Cisneros et al. (2024)**
   - *Flow-Lenia.png: Evolving Multi-Scale Complexity by Means of Compression* — arXiv:2408.06374.
   - **Key Focus**: Plain GA with compression-based fitness as an efficient fallback recipe.

8. **Chan (2023)**
   - *Towards Large-Scale Simulations of Open-Ended Evolution in Continuous Cellular Automata* — arXiv:2304.05639.
   - **Key Focus**: Warning on large-scale runs: without parameter constraint/suppression, simulations pass through transient diversity before being dominated by single fast-expanding patterns.
