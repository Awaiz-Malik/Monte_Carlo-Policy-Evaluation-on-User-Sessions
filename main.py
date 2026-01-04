"""
Main execution script for Monte Carlo and Dynamic Programming experiments.

This script orchestrates the complete workflow:
1. Environment setup and episode generation
2. First-Visit Monte Carlo prediction
3. Dynamic Programming (bonus)
4. Results visualization and saving
"""

import numpy as np
from env import SessionEnv, generate_episodes, print_episode_summary
from mc import first_visit_mc_prediction, print_mc_results
from dp import run_dp_demo
from utils import (
    set_seed, plot_convergence, plot_episode_length_histogram,
    save_results_csv, save_episodes_npy, print_value_table,
    create_output_dirs
)


# Configuration
CONFIG = {
    'N_EPISODES': 2000,
    'GAMMA': 0.9,
    'SEED': 42,
    'SAVE_EPISODES': True,
    'SAVE_PLOTS': True
}


def main():
    """
    Main execution function.
    """
    print("="*70)
    print(" Reinforcement Learning Project: Monte Carlo & Dynamic Programming")
    print("="*70)
    
    # Create output directories
    create_output_dirs()
    
    # Set random seed
    set_seed(CONFIG['SEED'])
    print(f"\nConfiguration:")
    print(f"  Episodes: {CONFIG['N_EPISODES']}")
    print(f"  Discount Factor (γ): {CONFIG['GAMMA']}")
    print(f"  Random Seed: {CONFIG['SEED']}")
    
    # ====================================================================
    # STEP 1: Environment Setup and Episode Generation
    # ====================================================================
    print(f"\n{'='*70}")
    print("STEP 1: Generating Episodes")
    print(f"{'='*70}")
    
    env = SessionEnv(seed=CONFIG['SEED'])
    print("\nEnvironment created with transition probabilities:")
    transition_probs = env.get_transition_probs()
    for state, probs in transition_probs.items():
        print(f"  State {state}: {probs}")
    
    print(f"\nGenerating {CONFIG['N_EPISODES']} episodes...")
    episodes = generate_episodes(env, CONFIG['N_EPISODES'], seed=CONFIG['SEED'])
    
    # Print summary
    print_episode_summary(episodes, n_samples=3)
    
    # Save episodes
    if CONFIG['SAVE_EPISODES']:
        save_episodes_npy(episodes, 'data/episodes.npy')
    
    # Plot episode length distribution
    if CONFIG['SAVE_PLOTS']:
        plot_episode_length_histogram(episodes, 'outputs/episode_lengths.png')
    
    # ====================================================================
    # STEP 2: First-Visit Monte Carlo Prediction
    # ====================================================================
    print(f"\n{'='*70}")
    print("STEP 2: First-Visit Monte Carlo Prediction")
    print(f"{'='*70}")
    
    print(f"\nTraining Monte Carlo on {len(episodes)} episodes...")
    V_mc, V_history = first_visit_mc_prediction(episodes, gamma=CONFIG['GAMMA'])
    
    # Print results
    print_mc_results(V_mc, episodes, CONFIG['GAMMA'])
    
    # Save results
    if CONFIG['SAVE_PLOTS']:
        plot_convergence(V_history, 'outputs/convergence.png', CONFIG['GAMMA'])
        save_results_csv(V_mc, 'outputs/mc_values.csv')
    
    # ====================================================================
    # STEP 3: Dynamic Programming (Bonus)
    # ====================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Dynamic Programming on Toy MDP (Bonus)")
    print(f"{'='*70}")
    
    dp_results = run_dp_demo(gamma=CONFIG['GAMMA'])
    
    # ====================================================================
    # STEP 4: Comparison and Summary
    # ====================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Summary and Comparison")
    print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("Monte Carlo vs Dynamic Programming")
    print("="*70)
    
    print("\nMonte Carlo (Session Environment):")
    print_value_table(V_mc, "MC State Values")
    
    print("Dynamic Programming (Toy MDP):")
    print_value_table(dp_results['V_improved'], "DP State Values (Improved Policy)")
    
    print("\n" + "="*70)
    print("Key Differences:")
    print("="*70)
    print("""
Monte Carlo (MC):
  - Model-free: learns from sampled episodes
  - No need to know transition probabilities
  - Higher variance, needs more samples
  - Only updates after episode completes (suitable for episodic tasks)
  - Used for session environment with unknown/complex dynamics

Dynamic Programming (DP):
  - Model-based: requires known P(s'|s,a) and R(s,a,s')
  - Low variance, can be more sample-efficient
  - Bootstrap from current estimates (Bellman updates)
  - Requires full model specification
  - Used for toy MDP with known, simple dynamics
    """)
    
    print("="*70)
    print("All experiments completed successfully!")
    print("="*70)
    print("\nOutputs:")
    print("  - outputs/convergence.png      : MC convergence plot")
    print("  - outputs/episode_lengths.png  : Episode length histogram")
    print("  - outputs/mc_values.csv        : Final MC state values")
    print("  - data/episodes.npy            : Generated episodes")
    print("="*70)


if __name__ == "__main__":
    main()
