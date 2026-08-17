#!/usr/bin/env python3
"""
Legacy Wrapper: Runs both Gene-Wise Mutation and Softmax Negotiation Rule ablations.
Directs output to dedicated results directories:
  - results/gene_mutation/
  - results/negotiation_rule/
"""
import sys
from experiments.run_gene_mutation import run_gene_mutation_experiment
from experiments.run_negotiation_rule import run_negotiation_rule_experiment

def main():
    print("=== Running Thesis Chapter 2 Collision & Mixing Ablations ===")
    print("\n[1/2] Running Ablation 2A: Stochastic Gene-Wise Sampling (Gumbel-Max)...")
    run_gene_mutation_experiment(output_dir="results/gene_mutation")
    
    print("\n[2/2] Running Ablation 2B: Softmax Growth Negotiation Competition...")
    run_negotiation_rule_experiment(output_dir="results/negotiation_rule")
    
    print("\nAll Chapter 2 ablation artifacts successfully generated in results/gene_mutation/ and results/negotiation_rule/!")

if __name__ == "__main__":
    main()
