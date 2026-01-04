import numpy as np
import matplotlib.pyplot as plt
import json
import os

def set_seed(seed):
    np.random.seed(seed)


def plot_convergence(V_history, save_path='outputs/convergence.png', gamma=0.9):
    plt.figure(figsize=(10, 6))
    
    state_names = {
        0: 'State 0 (Passive Browsing)',
        1: 'State 1 (Selective Reading)',
        2: 'State 2 (Deep Engagement)'
    }
    
    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    
    for state in sorted(V_history.keys()):
        episodes = range(1, len(V_history[state]) + 1)
        plt.plot(episodes, V_history[state], 
                label=state_names[state], 
                color=colors[state],
                linewidth=2,
                alpha=0.8)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('State Value V(s)', fontsize=12)
    plt.title(f'First-Visit Monte Carlo Convergence (γ={gamma})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {save_path}")
    
    plt.close()


def plot_episode_length_histogram(episodes, save_path='outputs/episode_lengths.png'):
    lengths = [ep['length'] for ep in episodes]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=range(1, max(lengths) + 2), 
             color='#45B7D1', alpha=0.7, edgecolor='black')
    
    # Add vertical lines for reward boundaries
    plt.axvline(x=3.5, color='red', linestyle='--', linewidth=2, 
                label='Length ≤3 (reward=-8)', alpha=0.7)
    plt.axvline(x=7.5, color='orange', linestyle='--', linewidth=2, 
                label='Length 4-7 (reward=+4)', alpha=0.7)
    plt.axvline(x=8, color='green', linestyle='--', linewidth=2, 
                label='Length ≥8 (reward=+10)', alpha=0.7)
    
    plt.xlabel('Episode Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Episode Length Distribution (n={len(episodes)})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Episode length histogram saved to: {save_path}")
    
    plt.close()


def save_results_csv(V, filename='outputs/mc_values.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write('State,Value\n')
        for state in sorted(V.keys()):
            f.write(f'{state},{V[state]:.6f}\n')
    
    print(f"Values saved to: {filename}")


def save_episodes_json(episodes, filename='data/episodes.json'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(episodes, f, indent=2)
    
    print(f"Episodes saved to: {filename}")


def save_episodes_npy(episodes, filename='data/episodes.npy'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    np.save(filename, episodes, allow_pickle=True)
    print(f"Episodes saved to: {filename}")


def load_episodes_npy(filename='data/episodes.npy'):
    return np.load(filename, allow_pickle=True).tolist()


def print_value_table(V, title="State Values"):
    state_names = {
        0: 'Passive Browsing',
        1: 'Selective Reading',
        2: 'Deep Engagement'
    }
    
    print(f"\n{title}")
    print(f"{'='*50}")
    print(f"{'State':<5} {'Description':<25} {'Value':>15}")
    print(f"{'-'*50}")
    
    if isinstance(V, dict):
        for state in sorted(V.keys()):
            print(f"{state:<5} {state_names.get(state, 'Unknown'):<25} {V[state]:>15.6f}")
    else:
        for state in range(len(V)):
            print(f"{state:<5} {state_names.get(state, 'Unknown'):<25} {V[state]:>15.6f}")
    
    print(f"{'='*50}\n")


def create_output_dirs():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    print("Output directories created: outputs/, data/")


if __name__ == "__main__":
    print("Testing utility functions...")
    
    create_output_dirs()
    
    # Test plotting
    V_history = {
        0: [0, -2, -4, -5, -5.5],
        1: [0, -1, -2.5, -3, -3.2],
        2: [0, 1, 2.5, 3.5, 4]
    }
    
    plot_convergence(V_history, save_path='outputs/test_convergence.png')
    
    # Test value table
    V = {0: -5.5, 1: -3.2, 2: 4.0}
    print_value_table(V, "Test Values")
    
    print("✓ Utility tests passed!")
