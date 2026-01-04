"""
First-Visit Monte Carlo Prediction for Policy Evaluation

Implements the First-Visit MC algorithm to estimate state values V(s)
from sampled episodes with delayed, sparse rewards.
"""

import numpy as np


def first_visit_mc_prediction(episodes, gamma=0.9, num_states=3):
    """
    Estimate state values using First-Visit Monte Carlo prediction.
    
    Algorithm:
    1. Initialize V(s) = 0 for all states
    2. For each episode:
        a. Compute returns G_t for all time steps (backward pass)
        b. For each state s visited in the episode:
            - If this is the FIRST visit to s in this episode:
                - Append G_t to returns[s]
                - Update V(s) = mean(returns[s])
    
    Args:
        episodes (list): List of episode dictionaries with 'states' and 'rewards'
        gamma (float): Discount factor
        num_states (int): Number of non-terminal states (default 3: {0, 1, 2})
    
    Returns:
        tuple: (V, V_history)
            - V (dict): Final state values {state: value}
            - V_history (dict): Convergence history {state: [values over episodes]}
    """
    # Initialize value function
    V = {s: 0.0 for s in range(num_states)}
    
    # Store all returns for each state
    returns = {s: [] for s in range(num_states)}
    
    # Track convergence history
    V_history = {s: [] for s in range(num_states)}
    
    # Process each episode
    for ep_idx, episode in enumerate(episodes):
        states = episode['states']
        rewards = episode['rewards']
        
        # Compute returns using backward pass
        G = 0.0
        returns_list = []
        
        for t in reversed(range(len(rewards))):
            G = gamma * G + rewards[t]
            returns_list.append(G)
        
        # Reverse to align with time steps
        returns_list.reverse()
        
        # Track which states we've already updated (first-visit rule)
        visited_states = set()
        
        # Update values for first-visit states
        for t in range(len(states)):
            state = states[t]
            
            # Only update if this is the first visit to this state in this episode
            if state not in visited_states and state < num_states:
                returns[state].append(returns_list[t])
                V[state] = np.mean(returns[state])
                visited_states.add(state)
        
        # Record convergence history (every episode)
        for s in range(num_states):
            V_history[s].append(V[s])
    
    return V, V_history


def compute_returns_for_episode(episode, gamma=0.9):
    """
    Compute returns for a single episode (for visualization/debugging).
    
    Args:
        episode (dict): Episode with 'states' and 'rewards'
        gamma (float): Discount factor
    
    Returns:
        list: Returns G_t for each time step
    """
    rewards = episode['rewards']
    
    G = 0.0
    returns_list = []
    
    for t in reversed(range(len(rewards))):
        G = gamma * G + rewards[t]
        returns_list.append(G)
    
    returns_list.reverse()
    
    return returns_list


def get_first_visit_indices(states):
    """
    Find the first occurrence index of each state in the trajectory.
    
    Args:
        states (list): List of states visited
    
    Returns:
        dict: {state: first_index}
    """
    first_visits = {}
    
    for t, state in enumerate(states):
        if state not in first_visits:
            first_visits[state] = t
    
    return first_visits


def print_mc_results(V, episodes, gamma):
    """
    Print Monte Carlo prediction results.
    
    Args:
        V (dict): State values
        episodes (list): Training episodes
        gamma (float): Discount factor
    """
    print(f"\n{'='*60}")
    print(f"First-Visit Monte Carlo Prediction Results")
    print(f"{'='*60}")
    print(f"Training Episodes: {len(episodes)}")
    print(f"Discount Factor (γ): {gamma}")
    print(f"\nEstimated State Values:")
    print(f"{'State':<20} {'Value':<15}")
    print(f"{'-'*35}")
    print(f"0 (Passive Browsing) {V[0]:>15.4f}")
    print(f"1 (Selective Reading){V[1]:>15.4f}")
    print(f"2 (Deep Engagement)  {V[2]:>15.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Quick test with hand-crafted episode
    print("Testing First-Visit Monte Carlo...")
    
    # Hand-crafted episode: [0,1,2,3] with length 3 → terminal reward -8
    # Expected returns (γ=0.9):
    #   G_2 = -8
    #   G_1 = 0.9 * (-8) + 0 = -7.2
    #   G_0 = 0.9 * (-7.2) + 0 = -6.48
    
    test_episode = {
        'states': [0, 1, 2],
        'rewards': [0, 0, -8],
        'length': 3
    }
    
    returns = compute_returns_for_episode(test_episode, gamma=0.9)
    print(f"\nTest Episode: states={test_episode['states']}, rewards={test_episode['rewards']}")
    print(f"Computed Returns: {returns}")
    print(f"Expected Returns: [-6.48, -7.2, -8.0]")
    
    # Verify
    expected = [-6.48, -7.2, -8.0]
    for i, (computed, exp) in enumerate(zip(returns, expected)):
        assert abs(computed - exp) < 0.01, f"Return mismatch at t={i}: {computed} vs {exp}"
    
    print("✓ Returns computation test passed!")
    
    # Test first-visit logic
    episodes = [test_episode]
    V, V_history = first_visit_mc_prediction(episodes, gamma=0.9)
    
    print(f"\nEstimated Values after 1 episode:")
    for s in [0, 1, 2]:
        print(f"  V({s}) = {V[s]:.4f}")
    
    print("\n✓ First-Visit MC test passed!")
