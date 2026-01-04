"""
Session Environment Simulator for Reinforcement Learning

States:
    0 = Passive Browsing
    1 = Selective Reading
    2 = Deep Engagement
    3 = Exit (terminal)

Rewards:
    - All intermediate rewards = 0
    - Terminal reward based on session length:
        +10 if length >= 8
        +4 if length 4-7
        -8 if length <= 3
"""

import numpy as np


class SessionEnv:
    """
    Session environment with discrete states and stochastic transitions.
    
    The environment simulates user browsing sessions with transitions between
    different engagement states until exit.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the session environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        
        # State space: 0, 1, 2 (non-terminal), 3 (terminal)
        self.num_states = 4
        
        # Transition probabilities: P[current_state] = [prob_0, prob_1, prob_2, prob_3]
        # Designed to create varying session lengths
        self.transition_probs = {
            0: [0.40, 0.30, 0.20, 0.10],  # Passive: higher exit prob
            1: [0.30, 0.40, 0.20, 0.10],  # Selective: moderate
            2: [0.10, 0.20, 0.50, 0.20],  # Deep: sticky, encourages longer sessions
        }
        
        # Maximum steps safety cap
        self.max_steps = 50
        
        # Episode state
        self.current_state = None
        self.step_count = 0
        self.done = False
    
    def reset(self):
        """
        Reset environment to a random initial state.
        
        Returns:
            int: Initial state (0, 1, or 2)
        """
        # Start from one of the non-terminal states
        self.current_state = self.rng.choice([0, 1, 2])
        self.step_count = 0
        self.done = False
        return self.current_state
    
    def step(self, action=None):
        """
        Execute one time step in the environment.
        
        Args:
            action: Not used (environment is stochastic, no actions)
        
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state (int): Next state
                - reward (float): Reward for this transition (0 unless terminal)
                - done (bool): True if episode has ended
                - info (dict): Additional information
        """
        if self.done:
            raise RuntimeError("Episode has already ended. Call reset() to start a new episode.")
        
        # Transition to next state based on probabilities
        next_state = self.rng.choice(
            [0, 1, 2, 3],
            p=self.transition_probs[self.current_state]
        )
        
        self.step_count += 1
        
        # Check if episode should end
        if next_state == 3 or self.step_count >= self.max_steps:
            self.done = True
            # Compute terminal reward based on session length
            reward = self._compute_terminal_reward(self.step_count)
            next_state = 3
        else:
            reward = 0.0
        
        info = {
            't': self.step_count,
            'episode_length': self.step_count if self.done else None
        }
        
        self.current_state = next_state
        
        return next_state, reward, self.done, info
    
    def _compute_terminal_reward(self, length):
        """
        Compute terminal reward based on session length.
        
        Args:
            length (int): Number of steps in the session
        
        Returns:
            float: Terminal reward
        """
        if length >= 8:
            return 10.0
        elif length >= 4:
            return 4.0
        else:
            return -8.0
    
    def get_transition_probs(self):
        """Get transition probability matrix for visualization."""
        return self.transition_probs.copy()


def generate_episodes(env, n_episodes, seed=None):
    """
    Generate a batch of episodes using the given environment.
    
    Each episode runs until termination (state 3) and stores the complete
    trajectory including states and rewards.
    
    Args:
        env: SessionEnv instance
        n_episodes (int): Number of episodes to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        list[dict]: List of episodes, each with:
            - 'states': List of visited states (excluding terminal state 3)
            - 'rewards': List of rewards (0 until terminal, then terminal reward)
            - 'length': Episode length
    """
    if seed is not None:
        env.rng = np.random.RandomState(seed)
    
    episodes = []
    
    for ep_idx in range(n_episodes):
        # Reset environment
        state = env.reset()
        
        states = [state]
        rewards = []
        
        # Run episode until termination
        while True:
            next_state, reward, done, info = env.step()
            
            # Store reward for this transition
            rewards.append(reward)
            
            if done:
                # Episode ended
                break
            
            # Store non-terminal state
            states.append(next_state)
        
        # Verify episode format
        assert len(states) == len(rewards), \
            f"States and rewards must align: {len(states)} states, {len(rewards)} rewards"
        
        episode = {
            'states': states,
            'rewards': rewards,
            'length': len(rewards)
        }
        
        episodes.append(episode)
    
    return episodes


def print_episode_summary(episodes, n_samples=5):
    """
    Print summary statistics and sample episodes.
    
    Args:
        episodes (list): List of generated episodes
        n_samples (int): Number of sample episodes to display
    """
    print(f"\n{'='*60}")
    print(f"Generated {len(episodes)} episodes")
    print(f"{'='*60}")
    
    # Length distribution
    lengths = [ep['length'] for ep in episodes]
    print(f"\nLength Statistics:")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.2f}")
    
    # Reward distribution
    rewards = [ep['rewards'][-1] for ep in episodes]  # Terminal rewards
    reward_counts = {
        -8: sum(1 for r in rewards if r == -8),
        4: sum(1 for r in rewards if r == 4),
        10: sum(1 for r in rewards if r == 10)
    }
    print(f"\nReward Distribution:")
    print(f"  -8 (length ≤3): {reward_counts[-8]} episodes ({reward_counts[-8]/len(episodes)*100:.1f}%)")
    print(f"  +4 (length 4-7): {reward_counts[4]} episodes ({reward_counts[4]/len(episodes)*100:.1f}%)")
    print(f"  +10 (length ≥8): {reward_counts[10]} episodes ({reward_counts[10]/len(episodes)*100:.1f}%)")
    
    # Sample episodes
    print(f"\nSample Episodes (first {n_samples}):")
    for i in range(min(n_samples, len(episodes))):
        ep = episodes[i]
        print(f"\n  Episode {i+1}:")
        print(f"    States:  {ep['states']}")
        print(f"    Rewards: {ep['rewards']}")
        print(f"    Length:  {ep['length']} → Terminal Reward: {ep['rewards'][-1]}")


if __name__ == "__main__":
    # Quick test
    print("Testing SessionEnv...")
    
    env = SessionEnv(seed=42)
    test_episodes = generate_episodes(env, n_episodes=100, seed=42)
    print_episode_summary(test_episodes, n_samples=5)
    
    # Verify all reward buckets are represented
    terminal_rewards = [ep['rewards'][-1] for ep in test_episodes]
    assert -8 in terminal_rewards, "Missing -8 reward bucket"
    assert 4 in terminal_rewards, "Missing +4 reward bucket"
    assert 10 in terminal_rewards, "Missing +10 reward bucket"
    
    print("\n✓ All reward buckets represented!")
    print("✓ Environment test passed!")
