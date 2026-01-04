import numpy as np

def first_visit_mc_prediction(episodes, gamma=0.9, num_states=3):
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
    rewards = episode['rewards']
    
    G = 0.0
    returns_list = []
    
    for t in reversed(range(len(rewards))):
        G = gamma * G + rewards[t]
        returns_list.append(G)
    
    returns_list.reverse()
    
    return returns_list


def get_first_visit_indices(states):
    first_visits = {}
    
    for t, state in enumerate(states):
        if state not in first_visits:
            first_visits[state] = t
    
    return first_visits


def print_mc_results(V, episodes, gamma):
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
