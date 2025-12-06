# Portfolio Rebalancing using Reinforcement Learning (CS221 Final Project)

This project compares simple portfolio baselines with two reinforcement-learning approaches (Value Iteration and Q-Learning) using daily returns for five tech stocks (AAPL, MSFT, AMZN, GOOGL, META) starting from 2015-01-01.

All methods are evaluated on:

- A strict 20% chronological test window, which includes the 2022 tech-sector drawdown, an unusually difficult out-of-sample period for concentrated tech portfolios
- The full time period, which helps visualize long-run cumulative performance

## Policies
Each notebook loads the processed returns, computes strategy returns, and produces performance metrics and plots.

### baselines.ipynb  
Implements three benchmark strategies:

- Buy-and-Hold (Equal Weights)  
- Random Rebalance (Dirichlet-sampled weights)  
- Monthly Rebalance (every 21 trading days)

### mdp.ipynb
Implements regime-based Markov Decision Process Value Iteration.

### qlearning.ipynb  
Implements regime-based Q-Learning.

## Methodology Summary

### Regimes  
Market regimes are computed using the equal-weighted return of all assets, lagged one day to eliminate lookahead. The three regimes are:

- Bear (return < –1%)  
- Sideways (between –1% and +1%)  
- Bull (return > +1%)

### Actions (Exposure Levels)  
The policy chooses among three exposure levels:

- Risk-Off (0.5×)  
- Neutral (1.0×)  
- Risk-On (1.5×)

### Transaction Costs  
Transaction costs are applied identically across baselines, Value Iteration, and Q-Learning. A cost is charged whenever the exposure changes. The project uses a transaction cost of 0.001.

### Evaluation Metrics  
All strategies are evaluated using:

- Net daily returns  
- Sharpe ratio  
- Final cumulative value  
- Cumulative return plots

Each strategy is evaluated on both the test window and the full period.


## Running the Project

Install dependencies from requirements.txt

Run the notebooks in this order:

1. data.ipynb
2. baselines.ipynb  
3. mdp.ipynb  
4. qlearning.ipynb  


## Interpretation Notes

- The test window covers late 2021–2023, dominated by the 2022 tech-stock crash, where mega-cap tech declined sharply due to rising interest rates and macro tightening. This makes the test set intentionally adversarial for risk-on strategies.
- Buy-and-Hold often performs well on full-period plots because tech experienced strong long-term appreciation (2015–2021 and 2023 rebound).
- Regime-only RL policies (Value Iteration and Q-Learning) are simple, interpretable, but lack predictive features and cannot anticipate sharp regime breaks like 2022.
- Full-period plots are included for intuition only; all final evaluations use the strict test window.

## Future Extensions

Potential improvements include:

- Adding predictive features such as volatility, momentum, macro indicators  
- Using richer regime models (HMMs, clustering)  
- Switching from tabular RL to deep RL (DQN, PPO, actor–critic)
- Performing walk-forward validation instead of a single split
- Refining transaction-cost modeling or action sets
