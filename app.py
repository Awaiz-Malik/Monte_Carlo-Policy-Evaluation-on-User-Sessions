"""
Streamlit Web Application for RL Project

Interactive frontend for:
- Episode generation and exploration
- Monte Carlo training and visualization
- Dynamic Programming evaluation and improvement
- Results comparison and analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

from env import SessionEnv, generate_episodes
from mc import first_visit_mc_prediction, compute_returns_for_episode, get_first_visit_indices
from dp import ToyMDP, policy_evaluation, policy_improvement
from utils import set_seed


# Page configuration
st.set_page_config(
    page_title="RL Project: MC & DP",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'episodes' not in st.session_state:
    st.session_state.episodes = None
if 'V_mc' not in st.session_state:
    st.session_state.V_mc = None
if 'V_history' not in st.session_state:
    st.session_state.V_history = None
if 'dp_results' not in st.session_state:
    st.session_state.dp_results = None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🎓 Reinforcement Learning Project</div>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666;">Monte Carlo Prediction & Dynamic Programming</p>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        n_episodes = st.slider("Number of Episodes", 
                               min_value=100, max_value=5000, value=2000, step=100)
        gamma = st.number_input("Discount Factor (γ)", 
                                min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        seed = st.number_input("Random Seed", 
                               min_value=0, max_value=9999, value=42, step=1)
        
        st.markdown("---")
        st.header("📖 Navigation")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Environment & Episodes",
        "📊 Monte Carlo Training",
        "🧠 Dynamic Programming",
        "📈 Analysis & Comparison"
    ])
    
    # ========================================================================
    # TAB 1: Environment & Episode Generation
    # ========================================================================
    with tab1:
        st.markdown('<div class="sub-header">Environment Setup</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### State Space")
            st.info("""
            **States:**
            - **0**: Passive Browsing
            - **1**: Selective Reading
            - **2**: Deep Engagement
            - **3**: Exit (Terminal)
            """)
        
        with col2:
            st.markdown("#### Reward Structure")
            st.success("""
            **Terminal Rewards:**
            - `+10` if session length ≥ 8 steps
            - `+4` if session length 4-7 steps
            - `-8` if session length ≤ 3 steps
            
            All intermediate rewards = 0
            """)
        
        # Transition probabilities
        st.markdown('<div class="sub-header">Transition Probabilities</div>', unsafe_allow_html=True)
        
        env = SessionEnv(seed=seed)
        transition_probs = env.get_transition_probs()
        
        # Create transition matrix dataframe
        trans_df = pd.DataFrame({
            'From State': ['0 (Passive)', '1 (Selective)', '2 (Deep)'],
            'To 0': [transition_probs[0][0], transition_probs[1][0], transition_probs[2][0]],
            'To 1': [transition_probs[0][1], transition_probs[1][1], transition_probs[2][1]],
            'To 2': [transition_probs[0][2], transition_probs[1][2], transition_probs[2][2]],
            'To 3 (Exit)': [transition_probs[0][3], transition_probs[1][3], transition_probs[2][3]]
        })
        
        st.dataframe(trans_df, use_container_width=True)
        
        # Episode generation
        st.markdown('<div class="sub-header">Generate Episodes</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Generate Episodes", type="primary", use_container_width=True):
            with st.spinner(f"Generating {n_episodes} episodes..."):
                set_seed(seed)
                env = SessionEnv(seed=seed)
                st.session_state.episodes = generate_episodes(env, n_episodes, seed=seed)
                st.success(f"✅ Generated {n_episodes} episodes!")
        
        # Display episode statistics
        if st.session_state.episodes is not None:
            episodes = st.session_state.episodes
            
            st.markdown('<div class="sub-header">Episode Statistics</div>', unsafe_allow_html=True)
            
            lengths = [ep['length'] for ep in episodes]
            rewards = [ep['rewards'][-1] for ep in episodes]
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Episodes", len(episodes))
            with col2:
                st.metric("Avg Length", f"{np.mean(lengths):.2f}")
            with col3:
                st.metric("Min Length", min(lengths))
            with col4:
                st.metric("Max Length", max(lengths))
            
            # Reward distribution
            col1, col2, col3 = st.columns(3)
            reward_counts = {
                -8: sum(1 for r in rewards if r == -8),
                4: sum(1 for r in rewards if r == 4),
                10: sum(1 for r in rewards if r == 10)
            }
            
            with col1:
                st.metric("Reward -8 (≤3 steps)", 
                         f"{reward_counts[-8]} ({reward_counts[-8]/len(episodes)*100:.1f}%)")
            with col2:
                st.metric("Reward +4 (4-7 steps)", 
                         f"{reward_counts[4]} ({reward_counts[4]/len(episodes)*100:.1f}%)")
            with col3:
                st.metric("Reward +10 (≥8 steps)", 
                         f"{reward_counts[10]} ({reward_counts[10]/len(episodes)*100:.1f}%)")
            
            # Episode length histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=lengths,
                nbinsx=max(lengths),
                marker_color='#45B7D1',
                opacity=0.7,
                name='Episode Lengths'
            ))
            
            # Add reward boundary lines
            fig.add_vline(x=3.5, line_dash="dash", line_color="red", 
                         annotation_text="Reward -8", annotation_position="top")
            fig.add_vline(x=7.5, line_dash="dash", line_color="orange", 
                         annotation_text="Reward +4", annotation_position="top")
            
            fig.update_layout(
                title="Episode Length Distribution",
                xaxis_title="Episode Length",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample episodes
            st.markdown('<div class="sub-header">Sample Episodes</div>', unsafe_allow_html=True)
            
            n_samples = st.slider("Number of samples to display", 1, 10, 5)
            
            for i in range(min(n_samples, len(episodes))):
                with st.expander(f"Episode {i+1}"):
                    ep = episodes[i]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**States:** {ep['states']}")
                        st.write(f"**Length:** {ep['length']}")
                    with col2:
                        st.write(f"**Rewards:** {ep['rewards']}")
                        st.write(f"**Terminal Reward:** {ep['rewards'][-1]}")
            
            # Download episodes
            if st.button("💾 Download Episodes (JSON)", use_container_width=True):
                json_str = json.dumps(episodes, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="episodes.json",
                    mime="application/json"
                )
    
    # ========================================================================
    # TAB 2: Monte Carlo Training
    # ========================================================================
    with tab2:
        st.markdown('<div class="sub-header">First-Visit Monte Carlo Prediction</div>', 
                   unsafe_allow_html=True)
        
        if st.session_state.episodes is None:
            st.warning("⚠️ Please generate episodes first (Environment & Episodes tab)")
        else:
            episodes = st.session_state.episodes
            
            st.info(f"""
            **Training Configuration:**
            - Episodes: {len(episodes)}
            - Discount Factor (γ): {gamma}
            - Algorithm: First-Visit Monte Carlo
            """)
            
            if st.button("🎯 Train Monte Carlo", type="primary", use_container_width=True):
                with st.spinner("Training Monte Carlo..."):
                    V_mc, V_history = first_visit_mc_prediction(episodes, gamma=gamma)
                    st.session_state.V_mc = V_mc
                    st.session_state.V_history = V_history
                    st.success("✅ Monte Carlo training complete!")
            
            # Display results
            if st.session_state.V_mc is not None:
                V_mc = st.session_state.V_mc
                V_history = st.session_state.V_history
                
                st.markdown('<div class="sub-header">Final State Values</div>', 
                           unsafe_allow_html=True)
                
                # Value table
                col1, col2, col3 = st.columns(3)
                
                state_names = {
                    0: "Passive Browsing",
                    1: "Selective Reading",
                    2: "Deep Engagement"
                }
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                for idx, (state, name) in enumerate(state_names.items()):
                    col = [col1, col2, col3][idx]
                    with col:
                        st.markdown(f"""
                        <div style="background: {colors[idx]}; color: white; padding: 1.5rem; 
                                    border-radius: 0.5rem; text-align: center;">
                            <h3 style="margin: 0; color: white;">State {state}</h3>
                            <p style="margin: 0.5rem 0 0 0; color: white;">{name}</p>
                            <h2 style="margin: 0.5rem 0 0 0; color: white;">{V_mc[state]:.4f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Convergence plot
                st.markdown('<div class="sub-header">Convergence Over Episodes</div>', 
                           unsafe_allow_html=True)
                
                fig = go.Figure()
                
                for state in [0, 1, 2]:
                    episodes_range = list(range(1, len(V_history[state]) + 1))
                    fig.add_trace(go.Scatter(
                        x=episodes_range,
                        y=V_history[state],
                        mode='lines',
                        name=f'State {state} ({state_names[state]})',
                        line=dict(color=colors[state], width=2)
                    ))
                
                fig.update_layout(
                    title=f"First-Visit Monte Carlo Convergence (γ={gamma})",
                    xaxis_title="Episode",
                    yaxis_title="State Value V(s)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Episode explorer
                st.markdown('<div class="sub-header">Episode Explorer</div>', 
                           unsafe_allow_html=True)
                
                episode_idx = st.selectbox("Select episode to inspect", 
                                          range(len(episodes)), 
                                          format_func=lambda x: f"Episode {x+1}")
                
                selected_ep = episodes[episode_idx]
                returns = compute_returns_for_episode(selected_ep, gamma=gamma)
                first_visits = get_first_visit_indices(selected_ep['states'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Episode Details:**")
                    st.write(f"- Length: {selected_ep['length']}")
                    st.write(f"- Terminal Reward: {selected_ep['rewards'][-1]}")
                    st.write(f"- States: {selected_ep['states']}")
                    st.write(f"- Rewards: {selected_ep['rewards']}")
                
                with col2:
                    st.write("**Returns (G_t):**")
                    for t, (state, G) in enumerate(zip(selected_ep['states'], returns)):
                        is_first = first_visits.get(state, -1) == t
                        marker = "🟢" if is_first else "⚪"
                        st.write(f"{marker} t={t}, s={state}, G={G:.4f}" + 
                                (" (first visit)" if is_first else ""))
    
    # ========================================================================
    # TAB 3: Dynamic Programming
    # ========================================================================
    with tab3:
        st.markdown('<div class="sub-header">Dynamic Programming on Toy MDP</div>', 
                   unsafe_allow_html=True)
        
        st.info("""
        This is a separate toy MDP with **known** transition dynamics, demonstrating 
        model-based reinforcement learning.
        """)
        
        # Display MDP specification
        mdp = ToyMDP()
        mdp.gamma = gamma
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### MDP Specification")
            st.write("**States:** {0, 1, 2}")
            st.write("**Actions:** {0: 'scroll', 1: 'click'}")
            st.write(f"**Discount Factor (γ):** {gamma}")
        
        with col2:
            st.markdown("#### Initial Policy")
            st.write("Uniform random policy:")
            st.write("- Each action has probability 0.5")
        
        if st.button("🧠 Run Policy Evaluation & Improvement", type="primary", 
                    use_container_width=True):
            with st.spinner("Running Dynamic Programming..."):
                # Initial policy (uniform random)
                initial_policy = {}
                for s in range(mdp.num_states):
                    initial_policy[s] = {0: 0.5, 1: 0.5}
                
                # Policy evaluation
                V_initial, iterations = policy_evaluation(mdp, initial_policy)
                
                # Policy improvement
                improved_policy = policy_improvement(mdp, V_initial)
                
                # Evaluate improved policy
                improved_policy_dist = {}
                for s in range(mdp.num_states):
                    improved_policy_dist[s] = {improved_policy[s]: 1.0}
                
                V_improved, _ = policy_evaluation(mdp, improved_policy_dist)
                
                st.session_state.dp_results = {
                    'mdp': mdp,
                    'initial_policy': initial_policy,
                    'V_initial': V_initial,
                    'improved_policy': improved_policy,
                    'V_improved': V_improved,
                    'iterations': iterations
                }
                
                st.success(f"✅ Policy evaluation converged in {iterations} iterations!")
        
        # Display results
        if st.session_state.dp_results is not None:
            results = st.session_state.dp_results
            
            st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Initial Policy Values")
                for s in range(3):
                    st.metric(f"V({s})", f"{results['V_initial'][s]:.4f}")
            
            with col2:
                st.markdown("#### Improved Policy")
                action_names = {0: 'scroll', 1: 'click'}
                for s in range(3):
                    action = results['improved_policy'][s]
                    st.metric(f"π'(s={s})", action_names[action])
            
            st.markdown("#### Improved Policy Values")
            col1, col2, col3 = st.columns(3)
            
            for idx, s in enumerate([0, 1, 2]):
                col = [col1, col2, col3][idx]
                with col:
                    st.metric(f"V({s})", f"{results['V_improved'][s]:.4f}")
    
    # ========================================================================
    # TAB 4: Analysis & Comparison
    # ========================================================================
    with tab4:
        st.markdown('<div class="sub-header">Monte Carlo vs Dynamic Programming</div>', 
                   unsafe_allow_html=True)
        
        if st.session_state.V_mc is None or st.session_state.dp_results is None:
            st.warning("⚠️ Please complete both MC training and DP evaluation first!")
        else:
            V_mc = st.session_state.V_mc
            V_dp = st.session_state.dp_results['V_improved']
            
            # Comparison table
            st.markdown("#### Value Function Comparison")
            
            comparison_df = pd.DataFrame({
                'State': [0, 1, 2],
                'MC Value': [V_mc[0], V_mc[1], V_mc[2]],
                'DP Value': [V_dp[0], V_dp[1], V_dp[2]],
                'Difference': [V_mc[0]-V_dp[0], V_mc[1]-V_dp[1], V_mc[2]-V_dp[2]]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['State 0', 'State 1', 'State 2'],
                y=[V_mc[0], V_mc[1], V_mc[2]],
                name='Monte Carlo',
                marker_color='#45B7D1'
            ))
            
            fig.add_trace(go.Bar(
                x=['State 0', 'State 1', 'State 2'],
                y=[V_dp[0], V_dp[1], V_dp[2]],
                name='Dynamic Programming',
                marker_color='#FF6B6B'
            ))
            
            fig.update_layout(
                title="Value Functions: MC vs DP",
                xaxis_title="State",
                yaxis_title="Value",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Conceptual comparison
            st.markdown('<div class="sub-header">Conceptual Comparison</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Monte Carlo")
                st.success("""
                **Advantages:**
                - Model-free (no P, R required)
                - Can learn from experience
                - Works with unknown dynamics
                - Episode-based learning
                
                **Disadvantages:**
                - Higher variance
                - Needs complete episodes
                - More samples needed
                - Only for episodic tasks
                """)
            
            with col2:
                st.markdown("#### 🧠 Dynamic Programming")
                st.info("""
                **Advantages:**
                - Model-based (uses P, R)
                - Low variance
                - Sample efficient
                - Bootstrap from estimates
                
                **Disadvantages:**
                - Requires full model
                - Computationally expensive
                - Model must be accurate
                - Not suitable when model unknown
                """)
            
            # Insights
            st.markdown('<div class="sub-header">Key Insights</div>', 
                       unsafe_allow_html=True)
            
            st.markdown("""
            #### Monte Carlo Results Interpretation
            - Values reflect the **expected returns** from each state under the stochastic policy
            - States with higher exit probability (State 0) tend to have **lower values** 
              due to shorter episodes
            - State 2 (Deep Engagement) is "sticky" and encourages longer sessions, 
              leading to **higher expected returns**
            - Sparse rewards make convergence slower, requiring more episodes
            
            #### Why Different MDPs?
            - **MC** learns from the session environment (complex, unknown transitions)
            - **DP** demonstrates model-based learning on a toy MDP (known, simple)
            - The values are **not directly comparable** as they come from different environments
            - This illustrates the fundamental difference: **model-free vs model-based** RL
            """)


if __name__ == "__main__":
    main()
