# Reinforcement Learning Project

University project demonstrating **First-Visit Monte Carlo Prediction** and **Dynamic Programming** for policy evaluation.

## 📋 Project Overview

This project implements:

1. **Session Environment Simulator** - Custom discrete environment with stochastic transitions and sparse, delayed rewards
2. **First-Visit Monte Carlo Prediction** - Model-free policy evaluation from sampled episodes
3. **Dynamic Programming** - Model-based policy evaluation and improvement on a toy MDP
4. **Interactive Streamlit Frontend** - Comprehensive web UI for all functionality
5. **CLI Interface** - Command-line execution with automatic visualization

## 🎯 Key Features

- **States**: {0: Passive Browsing, 1: Selective Reading, 2: Deep Engagement, 3: Exit (terminal)}
- **Sparse Rewards**: 
  - +10 if session length ≥ 8
  - +4 if session length 4-7  
  - -8 if session length ≤ 3
  - All intermediate rewards = 0
- **Discount Factor (γ)**: 0.9
- **First-Visit Rule**: Each state updated only once per episode
- **Proper Returns Calculation**: Backward pass with discounting

## 🛠️ Setup

### Prerequisites

- Python 3.8+ (using existing `.venv` virtual environment)
- Required packages: `numpy`, `matplotlib`, `streamlit`, `plotly`, `pandas`

### Installation

1. Activate your virtual environment:
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install numpy matplotlib streamlit plotly pandas
```

## 🚀 Usage

### Option 1: Streamlit Web Interface (Recommended)

Run the interactive web application:

```bash
streamlit run app.py
```

This opens a browser with 4 interactive pages:

1. **🌍 Environment & Episodes** - Configure and generate episodes
2. **📊 Monte Carlo Training** - Train MC and visualize convergence
3. **🧠 Dynamic Programming** - Run policy evaluation/improvement on toy MDP
4. **📈 Analysis & Comparison** - Compare MC vs DP results

### Option 2: Command Line Interface

Run the complete workflow:

```bash
python main.py
```

This executes:
- Episode generation (2000 episodes by default)
- First-Visit MC prediction
- Dynamic Programming evaluation
- Automatic plotting and result saving

## 📁 Project Structure

```
RL_Project/
├── env.py              # Session environment + episode generation
├── mc.py               # First-Visit Monte Carlo prediction
├── dp.py               # Dynamic Programming (policy eval/improvement)
├── utils.py            # Plotting, saving, and helper functions
├── main.py             # CLI execution script
├── app.py              # Streamlit web application
├── README.md           # This file
├── outputs/            # Generated plots and results
│   ├── convergence.png
│   ├── episode_lengths.png
│   └── mc_values.csv
└── data/               # Saved episodes
    └── episodes.npy
```

## 📊 Outputs

### Streamlit App
- Interactive visualizations
- Real-time convergence plots
- Episode explorer
- Download results as JSON/CSV

### CLI
- **outputs/convergence.png** - MC convergence plot for all states
- **outputs/episode_lengths.png** - Episode length histogram with reward boundaries
- **outputs/mc_values.csv** - Final state values
- **data/episodes.npy** - All generated episodes (for reproducibility)

## 🧪 Testing

Each module includes self-tests. Run individually:

```bash
python env.py      # Test environment
python mc.py       # Test MC with hand-crafted episode
python dp.py       # Test DP policy evaluation
python utils.py    # Test utility functions
```

## 🔍 Implementation Details

### Environment Design

**Transition Probabilities** (designed to create varied session lengths):
- **State 0**: 40% stay, 30% → 1, 20% → 2, 10% → Exit
- **State 1**: 30% → 0, 40% stay, 20% → 2, 10% → Exit
- **State 2**: 10% → 0, 20% → 1, 50% stay, 20% → Exit

State 2 is "stickier" to encourage longer sessions.

### First-Visit Monte Carlo

```
Algorithm:
1. Initialize V(s) = 0 for all states
2. For each episode:
   a. Compute returns G_t using backward pass:
      G = 0
      for t from last to first:
          G = γ * G + reward[t]
   b. For each state s in episode:
      if first visit to s:
          append G_t to returns[s]
          V(s) = mean(returns[s])
```

### Dynamic Programming

Separate toy MDP with known transitions:
- **Policy Evaluation**: Bellman expectation iteration
- **Policy Improvement**: Greedy policy w.r.t. computed values

## 📝 Key Differences: MC vs DP

| Aspect | Monte Carlo | Dynamic Programming |
|--------|-------------|---------------------|
| **Model** | Model-free | Model-based (requires P, R) |
| **Learning** | From episodes | From model knowledge |
| **Variance** | Higher | Lower |
| **Efficiency** | Needs more samples | More sample-efficient |
| **Use Case** | Unknown dynamics | Known dynamics |
| **Bootstrap** | No | Yes (uses V estimates) |
| **Task Type** | Episodic only | Both episodic & continuing |

## 🎓 Course Requirements

This implementation satisfies all project requirements:

✅ Discrete states: 0, 1, 2 (non-terminal), 3 (terminal)  
✅ Sparse, delayed rewards (+10, +4, -8)  
✅ Discount factor γ = 0.9  
✅ First-Visit MC with proper returns computation  
✅ Episode generation with all reward buckets  
✅ Convergence visualization  
✅ DP bonus (policy evaluation + improvement)  
✅ Complete documentation

## 📄 Academic Report

The academic report (`report.pdf`) will be created after implementation verification, covering:
- Environment design and methodology
- MC convergence analysis
- DP results interpretation
- MC vs DP comparison
- Effect of sparse/delayed rewards on learning

## 🤝 Contributing

This is a university project. Modifications should maintain:
- Exact state/reward specifications
- First-visit MC rule
- Proper return computation with γ=0.9
- Episode termination at state 3

## 📧 Contact

For questions or issues, refer to the course materials or instructor.

---

**Version**: 1.0  
**Last Updated**: January 2026
