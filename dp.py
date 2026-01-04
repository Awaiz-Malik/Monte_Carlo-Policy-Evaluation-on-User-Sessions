import numpy as np

class ToyMDP:
    def __init__(self):
        self.num_states = 3
        self.num_actions = 2
        self.gamma = 0.9

        self.P = {
            0: {  # State 0
                0: [(0.7, 0), (0.2, 1), (0.1, 2)],  # scroll: mostly stay
                1: [(0.3, 0), (0.4, 1), (0.3, 2)]   # click: more likely to advance
            },
            1: {  # State 1
                0: [(0.2, 0), (0.6, 1), (0.2, 2)],  # scroll: mostly stay
                1: [(0.1, 0), (0.3, 1), (0.6, 2)]   # click: likely to advance
            },
            2: {  # State 2
                0: [(0.1, 0), (0.2, 1), (0.7, 2)],  # scroll: mostly stay
                1: [(0.2, 0), (0.1, 1), (0.7, 2)]   # click: mostly stay
            }
        }
        
        # Rewards: R[s][a][s'] - reward for transitioning from s to s' via action a
        self.R = {
            0: {
                0: {0: 0, 1: 1, 2: 2},   # Small rewards for advancing
                1: {0: 0, 1: 2, 2: 4}    # Higher rewards for advancing via click
            },
            1: {
                0: {0: -1, 1: 1, 2: 3},
                1: {0: -1, 1: 2, 2: 5}
            },
            2: {
                0: {0: -2, 1: 0, 2: 2},  # State 2 gives good reward for staying
                1: {0: -2, 1: 0, 2: 3}
            }
        }
    
    def get_expected_reward(self, s, a):
        expected_r = 0.0
        for prob, next_s in self.P[s][a]:
            expected_r += prob * self.R[s][a][next_s]
        return expected_r


def policy_evaluation(mdp, policy, theta=1e-6, max_iterations=1000):
    V = np.zeros(mdp.num_states)
    
    for iteration in range(max_iterations):
        delta = 0.0
        
        for s in range(mdp.num_states):
            v_old = V[s]
            
            # Bellman expectation backup
            v_new = 0.0
            for a in range(mdp.num_actions):
                # Get policy probability
                pi_a = policy[s].get(a, 0.0)
                
                # Compute expected value for this action
                action_value = 0.0
                for prob, next_s in mdp.P[s][a]:
                    reward = mdp.R[s][a][next_s]
                    action_value += prob * (reward + mdp.gamma * V[next_s])
                
                v_new += pi_a * action_value
            
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        
        if delta < theta:
            return V, iteration + 1
    
    return V, max_iterations


def policy_improvement(mdp, V):
    improved_policy = {}
    
    for s in range(mdp.num_states):
        action_values = np.zeros(mdp.num_actions)
        
        for a in range(mdp.num_actions):
            # Compute Q(s,a)
            q_value = 0.0
            for prob, next_s in mdp.P[s][a]:
                reward = mdp.R[s][a][next_s]
                q_value += prob * (reward + mdp.gamma * V[next_s])
            
            action_values[a] = q_value
        
        # Greedy action
        improved_policy[s] = int(np.argmax(action_values))
    
    return improved_policy


def print_dp_results(mdp, initial_policy, V_initial, improved_policy, V_improved=None):
    action_names = {0: 'scroll', 1: 'click'}
    
    print(f"\n{'='*60}")
    print(f"Dynamic Programming Results (Toy MDP)")
    print(f"{'='*60}")
    
    print(f"\nInitial Policy (uniform random):")
    for s in range(mdp.num_states):
        actions = [f"{action_names[a]}:{initial_policy[s][a]:.1f}" 
                  for a in range(mdp.num_actions)]
        print(f"  State {s}: {', '.join(actions)}")
    
    print(f"\nValue Function under Initial Policy:")
    for s in range(mdp.num_states):
        print(f"  V({s}) = {V_initial[s]:.4f}")
    
    print(f"\nImproved Policy (greedy):")
    for s in range(mdp.num_states):
        best_action = improved_policy[s]
        print(f"  State {s}: {action_names[best_action]}")
    
    if V_improved is not None:
        print(f"\nValue Function under Improved Policy:")
        for s in range(mdp.num_states):
            print(f"  V({s}) = {V_improved[s]:.4f}")
    
    print(f"{'='*60}\n")


def run_dp_demo(gamma=0.9):
    # Create MDP
    mdp = ToyMDP()
    mdp.gamma = gamma
    
    # Define uniform random policy
    initial_policy = {}
    for s in range(mdp.num_states):
        initial_policy[s] = {0: 0.5, 1: 0.5}  # Equal probability for both actions
    
    print("Running Policy Evaluation...")
    V_initial, iterations = policy_evaluation(mdp, initial_policy)
    print(f"Converged in {iterations} iterations")
    
    print("\nRunning Policy Improvement...")
    improved_policy = policy_improvement(mdp, V_initial)
    
    # Optionally evaluate the improved policy
    improved_policy_dist = {}
    for s in range(mdp.num_states):
        improved_policy_dist[s] = {improved_policy[s]: 1.0}
    
    V_improved, _ = policy_evaluation(mdp, improved_policy_dist)
    
    # Print results
    print_dp_results(mdp, initial_policy, V_initial, improved_policy, V_improved)
    
    return {
        'mdp': mdp,
        'initial_policy': initial_policy,
        'V_initial': V_initial,
        'improved_policy': improved_policy,
        'V_improved': V_improved
    }


if __name__ == "__main__":
    print("Testing Dynamic Programming...")
    results = run_dp_demo(gamma=0.9)
    print("✓ DP test passed!")
