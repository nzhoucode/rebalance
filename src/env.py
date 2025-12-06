import numpy as np
import pandas as pd

class PortfolioEnv:
    """ Simple portfolio environment for model-free methods """
    def __init__(self, returns_df: pd.DataFrame, transaction_cost=0.001, initial_value=1.0):
        self.returns = returns_df
        self.tc = transaction_cost
        self.initial_value = initial_value
        self.n_assets = returns_df.shape[1]
        self.reset()

    def reset(self, initial_weights=None):
        self.t = 0
        if initial_weights is None:
            # Start uniform across assets
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            w = np.array(initial_weights, dtype=float)
            w = np.clip(w, 0, None)
            self.weights = w / w.sum()

        self.value = float(self.initial_value)
        return self._get_state()

    def _get_state(self):
        return {
            "t": self.t,
            "weights": self.weights.copy(),
            "returns_today": self.returns.iloc[self.t].values.copy()
        }

    def step(self, new_weights):
        # Ensure valid weights
        new_weights = np.array(new_weights, dtype=float)
        new_weights = np.clip(new_weights, 0, None)
        if new_weights.sum() == 0:
            new_weights = self.weights.copy()
        else:
            new_weights = new_weights / new_weights.sum()

        # Turnover and transaction cost
        turnover = np.sum(np.abs(new_weights - self.weights))
        cost = turnover * self.tc

        # Realized portfolio return for this step
        asset_returns = self.returns.iloc[self.t].values
        gross_portfolio_ret = np.dot(new_weights, asset_returns)
        net_portfolio_ret = gross_portfolio_ret - cost

        # Update portfolio
        self.value *= (1 + net_portfolio_ret)
        self.weights = new_weights
        self.t += 1

        done = (self.t >= len(self.returns) - 1)
        next_state = self._get_state() if not done else None
        reward = net_portfolio_ret

        info = {
            "portfolio_value": self.value,
            "gross_ret": gross_portfolio_ret,
            "net_ret": net_portfolio_ret,
            "turnover": turnover,
        }

        return next_state, reward, done, info