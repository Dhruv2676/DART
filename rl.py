import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import time
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from collections import deque

from config import get_args
from dataset import StockDataset
from model import GNNModel, Trainer


class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for stock trading.
    It simulates a portfolio management task where an agent allocates capital 
    among various stocks and cash based on GNN embeddings and market data.
    """
    metadata = {'rended.modes': ['human']}
    def __init__(self, dataset, precomputed_states, precomputed_preds, dates, config, phase_idx, split='train'):
        """
        Initializes the trading environment.

        Parameters:
        - dataset: StockDataset instance containing market data.
        - precomputed_states: GNN embeddings for the current phase.
        - precomputed_preds: GNN predictions for the current phase.
        - dates: list of dates corresponding to the data.
        - config: configuration object.
        - phase_idx: current phase index.
        - split: 'train', 'dev', or 'test'.
        """
        super(TradingEnvironment, self).__init__()
        
        self.dataset = dataset
        self.config = config
        self.phase_idx = phase_idx
        self.split = split
        
        self.dataset.set_phase(phase_idx)
        
        if split == 'train':
            self.windows = dataset.train_windows
            self.labels = dataset.train_labels
            self.neighbors = dataset.train_sampled_neighbors
            self.global_start_index = self.dataset.phases[phase_idx]['global_indices']['train']
        elif split == 'dev':
            self.windows = dataset.dev_windows
            self.labels = dataset.dev_labels
            self.neighbors = dataset.dev_sampled_neighbors
            self.global_start_index = self.dataset.phases[phase_idx]['global_indices']['dev']
        else: 
            self.windows = dataset.test_windows
            self.labels = dataset.test_labels
            self.neighbors = dataset.test_sampled_neighbors
            self.global_start_index = self.dataset.phases[phase_idx]['global_indices']['test']
        
        self.num_companies = dataset.num_companies
        self.num_timesteps = len(self.windows)
        
        if self.num_timesteps == 0:
            raise ValueError(f"No data available for phase {phase_idx}, split {split}")
        if len(dates) != self.num_timesteps:
            raise ValueError(
                f"CRITICAL DATA MISMATCH: "
                f"Env has {self.num_timesteps} steps but received {len(dates)} dates. "
                f"Check dataset.create_windows_for_split logic."
            )
        
        self.dates = dates
        
        self.precomputed_states = precomputed_states
        self.precomputed_preds = precomputed_preds
        
        self.initial_balance = config.initial_balance
        self.transaction_cost = config.transaction_cost
        self.turbulence_threshold = config.turbulence_threshold
        
        # State variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.num_companies)
        self.portfolio_value_history = []
        self.action_history = []
        self.reward_history = []
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]
        self.recent_log_returns = deque(maxlen=self.config.reward_window_size)
        self.current_high_water_mark = self.initial_balance
        
        self._initialize_prices()
        
        # Action space: [Cash_Weight, Stock_1_Weight, ..., Stock_N_Weight]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_companies+1,), dtype=np.float32)
        
        # Observation space size inference
        sample_obs = self._compute_observation_for_step(0)
        self.observation_dim = len(sample_obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
    
        print(f"  Environment created: Phase {phase_idx}, Split {split}, Steps {self.num_timesteps}")
        
    def _initialize_prices(self):
        """Sets initial stock prices to 100.0 for simulation normalization."""
        self._base_prices = np.ones(self.num_companies) * 100.0
        self._price_history = [self._base_prices.copy()]
    
    def _get_current_prices(self):
        """
        Calculates current prices based on cumulative returns from the dataset.
        Returns cached price if already computed for the current step.
        """
        if self.current_step >= len(self._price_history):
            if self.current_step < self.num_timesteps:
                global_step_index = self.global_start_index + self.current_step
                if global_step_index >= self.dataset.all_returns.shape[1]:
                    print(f"Warning: Out of bounds access in all_returns. Step: {self.current_step}, Global: {global_step_index}")
                    returns = np.zeros(self.num_companies)
                else:
                    returns = self.dataset.all_returns[:, global_step_index]     
                new_prices = self._price_history[-1] * np.exp(returns)
                self._price_history.append(new_prices)
                return new_prices
            else:
                return self._price_history[-1]
        return self._price_history[self.current_step]
    
    def _get_gnn_state_for_step(self, step):
        """Retrieves precomputed GNN embeddings and predictions for the given step."""
        if step >= len(self.precomputed_states):
            return np.zeros((self.num_companies, 128)), np.zeros((self.num_companies, 3))
        
        return self.precomputed_states[step], self.precomputed_preds[step]
    
    def _calculate_turbulence(self):
        """Calculates market turbulence index based on recent portfolio returns."""
        if len(self.portfolio_value_history) < 10:
            return 0.0
        
        recent_returns = np.diff(self.portfolio_value_history[-10:]) / self.portfolio_value_history[-10:-1]
        if len(recent_returns) < 2:
            return 0.0
        
        turbulence = np.std(recent_returns) * np.sqrt(252) * 100
        return turbulence
    
    def _compute_observation_for_step(self, step):
        """
        Constructs the state observation vector.
        Includes: GNN embeddings, predictions, current portfolio weights, cash ratio, and market metrics.
        """
        gnn_state, predictions = self._get_gnn_state_for_step(step)
        
        flat_gnn_state = gnn_state.flatten()
        flat_predictions = predictions.flatten()
        
        # Portfolio level sentiment metrics
        bullish_ratio = (predictions[:, 2] > 0.5).sum() / self.num_companies
        bearish_ratio = (predictions[:, 0] > 0.5).sum() / self.num_companies
        neutral_ratio = (predictions[:, 1] > 0.5).sum() / self.num_companies
        avg_confidence = np.mean(np.max(predictions, axis=1))
        
        total_value = self._calculate_portfolio_value()
        if total_value > 0:
            prices = self._get_current_prices()
            portfolio_weights = (self.shares_held * prices) / total_value
            cash_ratio = self.balance / total_value
        else:
            portfolio_weights = np.zeros(self.num_companies)
            cash_ratio = 1.0
            
        turbulence = self._calculate_turbulence()
        time_indicator = step / max(self.num_timesteps - 1, 1)
        
        observation = np.concatenate([
            flat_gnn_state,
            flat_predictions,
            portfolio_weights,
            [cash_ratio],
            [bullish_ratio, bearish_ratio, neutral_ratio],
            [avg_confidence],
            [turbulence],
            [time_indicator]
        ])
        return observation.astype(np.float32)
    
    def _calculate_portfolio_value(self):
        """Computes total portfolio value (Cash + Stock Equity)."""
        prices = self._get_current_prices()
        stock_value = np.sum(self.shares_held * prices)
        return self.balance + stock_value
    
    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.num_companies)
        self.portfolio_value_history = [self.initial_balance]
        self.action_history = []
        self.reward_history = []
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]
        
        self._initialize_prices()
        
        observation = self._compute_observation_for_step(self.current_step)
        info = {}
        return observation, info
    
    def step(self, action):
        """
        Executes one time step in the environment.
        1. Normalizes action (portfolio weights).
        2. Checks turbulence constraint.
        3. Rebalances portfolio.
        4. Calculates reward (Sharpe-like, Sortino-like, Drawdown penalties).
        """
        # Normalize actions to sum to 1
        action = np.abs(action)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            # Default to all cash if action is invalid
            action = np.zeros(self.num_companies + 1)
            action[0] = 1.0
        
        # Turbulence check: Force selling to cash if market is too volatile
        turbulence = self._calculate_turbulence()
        if turbulence > self.turbulence_threshold:
            action = np.zeros(self.num_companies + 1)
            action[0] = 1.0
        
        old_value = self._calculate_portfolio_value()
        prices = self._get_current_prices()
        
        # Execute Rebalancing
        cash_target_weight = action[0]
        stock_target_weights = action[1:]
        target_stock_value = old_value * stock_target_weights
        target_shares = np.divide(target_stock_value, prices, out=np.zeros_like(target_stock_value), where=prices!=0)
        trades = target_shares - self.shares_held
        
        # Apply Transaction Costs
        transaction_value = np.sum(np.abs(trades) * prices)
        transaction_fee = transaction_value * self.transaction_cost
        self.shares_held = target_shares.copy()
        target_cash_value = old_value * cash_target_weight
        self.balance = target_cash_value - transaction_fee
        
        self.current_step += 1
        new_value = self._calculate_portfolio_value()
        
        # Reward Calculation
        if old_value > 0 and new_value > 0:
            step_log_return = np.log(new_value / old_value)
        else:
            step_log_return = 0.0
            
        self.recent_log_returns.append(step_log_return)
        
        # Volatility Penalty (Sharpe-like)
        if len(self.recent_log_returns) > 1:
            general_volatility = np.std(list(self.recent_log_returns))
        else:
            general_volatility = 0.0
        general_vol_penalty = self.config.general_vol_penalty * general_volatility

        # Downside Volatility Penalty (Sortino-like)
        downside_returns = [r for r in self.recent_log_returns if r < 0]
        if len(downside_returns) > 1:
            downside_volatility = np.std(downside_returns)
        else:
            downside_volatility = 0.0
        downside_vol_penalty = self.config.downside_vol_penalty * downside_volatility

        # Drawdown Penalty
        if new_value > self.current_high_water_mark:
            self.current_high_water_mark = new_value
            
        if self.current_high_water_mark > 0:
            current_drawdown = (self.current_high_water_mark - new_value) / self.current_high_water_mark
        else:
            current_drawdown = 0.0
        drawdown_penalty = self.config.drawdown_penalty_weight * current_drawdown

        # Transaction Cost Penalty
        if old_value > 0:
            cost_penalty = self.config.cost_penalty_weight * (transaction_fee / old_value)
        else:
            cost_penalty = 0.0
            
        reward = (step_log_return - general_vol_penalty - downside_vol_penalty - drawdown_penalty - cost_penalty)
        
        terminated = self.current_step >= self.num_timesteps - 1
        truncated = False
        
        self.portfolio_value_history.append(new_value)
        self.portfolio_values.append(new_value)
        self.action_history.append(action.copy())
        self.reward_history.append(reward)
        self.returns_history.append(step_log_return)
        
        if not terminated:
            observation = self._compute_observation_for_step(self.current_step)
        else:
            observation = np.zeros(self.observation_dim, dtype=np.float32)
        
        info = {'portfolio_value': new_value, 'return': step_log_return, 'turbulence': turbulence, 'transaction_cost': transaction_fee, 'step': self.current_step}
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Renders the current state of the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.num_timesteps}")
            print(f"Portfolio Value: ${self._calculate_portfolio_value():,.2f}")
            print(f"Balance: ${self.balance:,.2f}")


class MetricsCalculator:
    """Helper class to calculate financial performance metrics."""
    @staticmethod
    def calculate_metrics(portfolio_values, returns, risk_free_rate=0.02):
        portfolio_values = np.array(portfolio_values)
        returns = np.array(returns)
        
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0 or len(portfolio_values) < 2:
            return {
                'cumulative_return': 0.0,
                'final_value': portfolio_values[0] if len(portfolio_values) > 0 else 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'sortino_ratio': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
                'mean_return': 0.0,
                'volatility': 0.0
            }
        
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        cumulative_return = (final_value - initial_value) / initial_value
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe_ratio = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdowns)
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                sortino_ratio = (mean_return - risk_free_rate/252) / downside_std * np.sqrt(252)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = sharpe_ratio if std_return > 0 else 0.0
        
        if std_return > 0:
            beta = std_return / 0.15  
        else:
            beta = 0.0
        
        # Alpha: Excess return over CAPM expected return
        market_return = risk_free_rate / 252  
        expected_return = risk_free_rate/252 + beta * (market_return - risk_free_rate/252)
        alpha = (mean_return - expected_return) * 252  # Annualized
        
        return {
            'cumulative_return': float(cumulative_return),
            'final_value': float(final_value),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'sortino_ratio': float(sortino_ratio),
            'alpha': float(alpha),
            'beta': float(beta),
            'mean_return': float(mean_return * 252),  # Annualized
            'volatility': float(std_return * np.sqrt(252))  # Annualized
        }

class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress."""
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        for ep_info in self.model.ep_info_buffer:
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])

class RLTrainer:
    """
    Manages the RL training pipeline:
    1. Pre-computes GNN states.
    2. Initializes PPO agent.
    3. Trains agent on training split.
    4. Evaluates on test split.
    """
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        
        self.output_dir = os.path.join(config.log_dir, 'rl_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.phase_results = []
        
    def _precompute_gnn_data(self, gnn_model, windows_list, neighbors_list):
        """
        Runs the GNN model sequentially on input windows to generate state embeddings.
        Avoids OOM errors by processing one timestep at a time.
        """
        if not windows_list:
            return [], []
            
        states = []
        preds = []
        
        print(f"    Processing {len(windows_list)} steps sequentially...")
        
        for i in range(len(windows_list)):
            # Get data for this specific timestep
            window = windows_list[i]
            neighbor = neighbors_list[i]
            
            # Convert to Tensor
            win_tensor = tf.convert_to_tensor(window, dtype=tf.float32)
            nbr_tensor = tf.convert_to_tensor(neighbor, dtype=tf.int32)
            
            # Pass to model (Num_Companies acts as batch dimension)
            outputs = gnn_model({'windows': win_tensor, 'neighbors': nbr_tensor}, training=False)
            
            states.append(outputs['gnn_state'].numpy())
            preds.append(outputs['predictions'].numpy())
            
            if (i + 1) % 100 == 0:
                print(f"    Step {i+1}/{len(windows_list)} complete", end='\r')
            
        print("") 
        return states, preds
    
    def load_or_train_gnn(self, phase_idx):
        """Loads a pre-trained GNN checkpoint or trains one if missing."""
        print(f"\n[Phase {phase_idx}] Loading/Training GNN model...")
        
        self.dataset.set_phase(phase_idx)
        
        gnn_model = GNNModel(self.config)
        gnn_model.build_layers()
        
        phase_checkpoint_dir = os.path.join(self.config.checkpoint_dir, f'phase_{phase_idx}')
        checkpoint = tf.train.Checkpoint(model=gnn_model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, phase_checkpoint_dir, max_to_keep=1)
        
        if checkpoint_manager.latest_checkpoint:
            try:
                print(f"[Phase {phase_idx}] Building model with sample data...")
                
                # Initialize model weights by running a dummy pass
                sample_windows = tf.convert_to_tensor(self.dataset.train_windows[0], dtype=tf.float32)
                sample_neighbors = tf.convert_to_tensor(self.dataset.train_sampled_neighbors[0], dtype=tf.int32)
                
                _ = gnn_model({'windows': sample_windows, 'neighbors': sample_neighbors}, training=False)
                
                print(f"[Phase {phase_idx}] ✓ Model built, weights are ready for restore.")
            except Exception as e:
                print(f"[Phase {phase_idx}] ❌ Error building model with sample data: {e}")
                print("  This might happen if phase 0 has no training data.")
                raise e
            
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            print(f"[Phase {phase_idx}] ✓ Loaded GNN checkpoint: {checkpoint_manager.latest_checkpoint}")
        else:
            print(f"[Phase {phase_idx}] ⚠️  No GNN checkpoint found. Training GNN first...")
            trainer = Trainer(gnn_model, self.dataset, self.config, phase_idx=phase_idx)
            trainer.train()
            checkpoint_manager.save()
            print(f"[Phase {phase_idx}] ✓ GNN training complete")
        return gnn_model
    
    def train_phase(self, phase_idx):
        """
        Executes the full RL training cycle for a specific phase.
        1. Load GNN.
        2. Precompute embeddings.
        3. Create Environments.
        4. Train PPO Agent.
        5. Evaluate and Save Results.
        """
        print(f"\n{'='*80}")
        print(f"RL TRAINING - PHASE {phase_idx}")
        print(f"Training NEW RL agent from scratch for this phase")
        print(f"{'='*80}")
        
        gnn_model = self.load_or_train_gnn(phase_idx)
        print(f"[Phase {phase_idx}] Pre-computing GNN states (Batch Processing)...")
        
        train_states, train_preds = self._precompute_gnn_data(
            gnn_model, 
            self.dataset.train_windows, 
            self.dataset.train_sampled_neighbors
        )
    
        test_states, test_preds = self._precompute_gnn_data(
            gnn_model, 
            self.dataset.test_windows, 
            self.dataset.test_sampled_neighbors
        )
        print(f"[Phase {phase_idx}] ✓ GNN pre-computation complete.")
        del gnn_model
        tf.keras.backend.clear_session()
        # --------------------------------------

        print(f"[Phase {phase_idx}] Creating environments...")
        phase_data = self.dataset.phases[phase_idx]
        train_dates = phase_data.get('train_dates', []) 
        test_dates = phase_data['test_dates']
        try:
            train_env_raw = TradingEnvironment(self.dataset, train_states, train_preds, train_dates, self.config, phase_idx, 'train')
            train_env = Monitor(train_env_raw)
            test_env_raw = TradingEnvironment(self.dataset, test_states, test_preds, test_dates, self.config, phase_idx, 'test')
            test_env = Monitor(test_env_raw)
        except ValueError as e:
            print(f"[Phase {phase_idx}] ❌ Error creating environment: {e}")
            return None
        
        train_env_vec = DummyVecEnv([lambda: train_env])
        print(f"[Phase {phase_idx}] Observation space: {train_env.observation_space.shape}")
        print(f"[Phase {phase_idx}] Action space: {train_env.action_space.shape}")
    
        print(f"[Phase {phase_idx}] Creating NEW PPO agent...")
        agent = PPO(policy='MlpPolicy', env=train_env_vec,
                    learning_rate=self.config.rl_lr,
                    n_steps=self.config.n_steps,
                    batch_size=self.config.batch_size,
                    n_epochs=self.config.rl_epochs,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                    clip_range=self.config.clip_range,
                    clip_range_vf=None,
                    ent_coef=self.config.ent_coef,
                    vf_coef=self.config.vf_coef,
                    max_grad_norm=self.config.max_grad_norm,
                    use_sde=False, sde_sample_freq=-1, target_kl=None, tensorboard_log=None,
                    policy_kwargs=dict(net_arch=self.config.net_arch), verbose=1, seed=None, device='auto')
        
        total_timesteps = train_env_raw.num_timesteps * self.config.num_episodes
        
        print(f"[Phase {phase_idx}] Training for {total_timesteps} timesteps...")
        print(f"[Phase {phase_idx}] (≈{self.config.num_episodes} episodes through the training data)")
        callback = TrainingCallback()
        
        start_time = time.time()
        agent.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        training_time = time.time() - start_time
        
        print(f"[Phase {phase_idx}] ✓ Training complete in {training_time:.2f}s")
        
        print(f"\n[Phase {phase_idx}] Evaluating on TEST set...")
        test_results = self.evaluate(agent, test_env)
        
        metrics = MetricsCalculator.calculate_metrics(
            test_results['portfolio_values'],
            test_results['returns']
        )
        
        test_dates = self.dataset.phases[phase_idx]['test_dates']
        raw_returns = test_results['returns']
        
        daily_returns_list = []
        used_dates = test_results['dates'] 
        used_returns = test_results['returns']
        
        for i in range(len(used_returns)):
            daily_returns_list.append({"date": str(used_dates[i]), "return": float(used_returns[i])})
            
        if len(used_dates) == 0:
            print("⚠️ WARNING: No returns logged. Evaluation might have failed immediately.")

        phase_result = {
            'phase': phase_idx,
            'training_time': training_time,
            'metrics': metrics,
            'daily_returns': daily_returns_list
        }
        
        self.phase_results.append(phase_result)
        
        print(f"\n[Phase {phase_idx}] TEST RESULTS:")
        print(f"  {'='*60}")
        print(f"  Final Portfolio Value:  ${metrics['final_value']:>15,.2f}")
        print(f"  Initial Value:          ${self.config.initial_balance:>15,.2f}")
        print(f"  Cumulative Return:      {metrics['cumulative_return']*100:>15.2f}%")
        print(f"  {'='*60}")
        print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:>15.4f}")
        print(f"  Sortino Ratio:          {metrics['sortino_ratio']:>15.4f}")
        print(f"  Max Drawdown:           {metrics['max_drawdown']*100:>15.2f}%")
        print(f"  Alpha (Annual):         {metrics['alpha']*100:>15.2f}%")
        print(f"  Beta:                   {metrics['beta']:>15.4f}")
        print(f"  Volatility (Annual):    {metrics['volatility']*100:>15.2f}%")
        print(f"  {'='*60}")
        
        results_file = os.path.join(self.output_dir, f'phase_{phase_idx}_results.json')
        with open(results_file, 'w') as f:
            json.dump(self._convert_to_json_serializable(phase_result), f, indent=2)
        
        self.plot_phase_results(phase_idx, test_results['portfolio_values'], test_results['returns'], test_results['actions'], test_results['dates'], self.output_dir)
        
        print(f"[Phase {phase_idx}] Clearing TF session to free memory...")
        tf.keras.backend.clear_session() 
        import gc
        gc.collect()
        
        return phase_result

    def _convert_to_json_serializable(self, obj):
        """Helper to convert NumPy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    def evaluate(self, agent, env):
        """
        Runs the agent on the environment to collect performance metrics.
        Returns full history of portfolio values, returns, actions, and dates.
        """
        obs, _ = env.reset()
        
        # Unwrap to get access to the custom environment variables
        if hasattr(env, 'envs'):
            unwrapped_env = env.envs[0]
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
        else:
            unwrapped_env = env.unwrapped

        done = False
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        num_steps = len(unwrapped_env.returns_history)
        aligned_dates = unwrapped_env.dates[:num_steps]

        return {
            'portfolio_values': unwrapped_env.portfolio_values, 
            'returns': unwrapped_env.returns_history, 
            'actions': unwrapped_env.action_history, 
            'rewards': unwrapped_env.reward_history,
            'dates': aligned_dates 
        }
    
    def plot_phase_results(self, phase_idx, portfolio_values, returns, actions, dates, save_dir):
        """Generates a comprehensive 2x2 grid plot of the phase results."""
        # Create a 2x2 Grid
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.35, wspace=0.2)
        
        # Determine X-axis (Dates or Steps)
        if len(dates) == len(portfolio_values):
            x_ticks = range(0, len(dates), max(1, len(dates)//8))
            x_labels = [dates[i] for i in x_ticks]
        else:
            x_ticks = None
            x_labels = None

        # --- 1. Portfolio Value (Top Left) ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(portfolio_values, linewidth=2, color='#1f77b4', label='RL Agent')
        ax1.axhline(self.config.initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        
        ax1.set_title(f'Phase {phase_idx}: Portfolio Value')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        if x_ticks:
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels(x_labels, rotation=15)

        # --- 2. Cash Allocation Strategy (Top Right) ---
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Extract cash weights from actions history
        cash_weights = [a[0] for a in actions]
        
        min_len = min(len(cash_weights), len(portfolio_values))
        cash_weights = cash_weights[:min_len]
        
        ax2.fill_between(range(len(cash_weights)), cash_weights, 0, color='#2ca02c', alpha=0.3, label='Cash Position')
        ax2.plot(cash_weights, color='#2ca02c', linewidth=1.5)
        
        ax2.set_title('Agent Behavior: Cash Allocation')
        ax2.set_ylabel('Cash Weight (0.0 - 1.0)')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        if x_ticks:
            ax2.set_xticks(x_ticks)
            ax2.set_xticklabels(x_labels, rotation=15)

        # --- 3. Daily Returns Time Series (Middle Left) ---
        ax3 = fig.add_subplot(gs[1, 0])
        colors = ['red' if r < 0 else 'green' for r in returns]
        ax3.bar(range(len(returns)), returns, color=colors, alpha=0.8, width=1.0)
        ax3.axhline(0, color='black', linewidth=0.5)
        
        ax3.set_title('Daily Returns (Volatility)')
        ax3.set_ylabel('Return')
        ax3.grid(True, alpha=0.3)
        if x_ticks:
            ax3.set_xticks(x_ticks)
            ax3.set_xticklabels(x_labels, rotation=15)

        # --- 4. Drawdown (Middle Right) ---
        ax4 = fig.add_subplot(gs[1, 1])
        portfolio_values_arr = np.array(portfolio_values)
        cumulative_max = np.maximum.accumulate(portfolio_values_arr)
        drawdown = (portfolio_values_arr - cumulative_max) / cumulative_max * 100
        
        ax4.fill_between(range(len(drawdown)), drawdown, 0, color='#d62728', alpha=0.3)
        ax4.plot(drawdown, color='#d62728', linewidth=1.5)
        ax4.set_title('Drawdown (%)')
        ax4.set_ylabel('Drawdown')
        ax4.grid(True, alpha=0.3)
        if x_ticks:
            ax4.set_xticks(x_ticks)
            ax4.set_xticklabels(x_labels, rotation=15)
        
        # --- 5. Metrics Table (Bottom) ---
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        metrics = MetricsCalculator.calculate_metrics(portfolio_values, returns)
        
        metrics_text = (
            f"Phase {phase_idx} Summary\n"
            f"--------------------------------------------------\n"
            f"Total Return:      {metrics['cumulative_return']*100:.2f}%\n"
            f"Final Value:       ${metrics['final_value']:,.2f}\n"
            f"Sharpe Ratio:      {metrics['sharpe_ratio']:.4f}\n"
            f"Max Drawdown:      {metrics['max_drawdown']*100:.2f}%\n"
            f"Volatility (Ann.): {metrics['volatility']*100:.2f}%"
        )
        
        ax5.text(0.5, 0.5, metrics_text, transform=ax5.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='gray'))
        
        plot_file = os.path.join(save_dir, f'phase_{phase_idx}_results.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[Phase {phase_idx}] ✓ Saved plot: {plot_file}")
        plt.close()
        
    def train_all_phases(self):
        """Iterates through all phases and trains a fresh agent for each."""
        print("\n" + "="*80)
        print("MULTI-PHASE RL TRAINING")
        print("Each phase gets a COMPLETELY NEW RL agent trained from scratch")
        print(f"Total phases: {self.config.n_phases}")
        print("="*80)
        
        for phase_idx in range(self.config.n_phases):
            try:
                phase_result = self.train_phase(phase_idx)
                if phase_result is None:
                    print(f"[Phase {phase_idx}] ⚠️  Skipped due to errors")
                    continue
            except Exception as e:
                print(f"\n❌ ERROR in Phase {phase_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(self.phase_results) > 0:
            self.aggregate_results()
        
        print("\n" + "="*80)
        print("✅ MULTI-PHASE RL TRAINING COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
    
    def aggregate_results(self):
        """Compute and save aggregate statistics across all trained phases."""
        print("\n" + "="*80)
        print("AGGREGATE RESULTS ACROSS ALL PHASES")
        print("="*80)
        
        cumulative_returns = [r['metrics']['cumulative_return'] for r in self.phase_results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in self.phase_results]
        max_drawdowns = [r['metrics']['max_drawdown'] for r in self.phase_results]
        sortino_ratios = [r['metrics']['sortino_ratio'] for r in self.phase_results]
        alphas = [r['metrics']['alpha'] for r in self.phase_results]
        betas = [r['metrics']['beta'] for r in self.phase_results]
        
        summary = {
            'num_phases': len(self.phase_results),
            'cumulative_return': {
                'mean': float(np.mean(cumulative_returns)),
                'std': float(np.std(cumulative_returns)),
                'min': float(np.min(cumulative_returns)),
                'max': float(np.max(cumulative_returns))
            },
            'sharpe_ratio': {
                'mean': float(np.mean(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios))
            },
            'max_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'std': float(np.std(max_drawdowns))
            },
            'sortino_ratio': {
                'mean': float(np.mean(sortino_ratios)),
                'std': float(np.std(sortino_ratios))
            },
            'alpha': {
                'mean': float(np.mean(alphas)),
                'std': float(np.std(alphas))
            },
            'beta': {
                'mean': float(np.mean(betas)),
                'std': float(np.std(betas))
            }
        }
        
        print(f"\n{'Metric':<30} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
        print("-"*90)
        print(f"{'Cumulative Return (%)':<30} {summary['cumulative_return']['mean']*100:>14.2f} "
              f"{summary['cumulative_return']['std']*100:>14.2f} "
              f"{summary['cumulative_return']['min']*100:>14.2f} "
              f"{summary['cumulative_return']['max']*100:>14.2f}")
        print(f"{'Sharpe Ratio':<30} {summary['sharpe_ratio']['mean']:>14.4f} "
              f"{summary['sharpe_ratio']['std']:>14.4f} {'-':>14} {'-':>14}")
        print(f"{'Sortino Ratio':<30} {summary['sortino_ratio']['mean']:>14.4f} "
              f"{summary['sortino_ratio']['std']:>14.4f} {'-':>14} {'-':>14}")
        print(f"{'Max Drawdown (%)':<30} {summary['max_drawdown']['mean']*100:>14.2f} "
              f"{summary['max_drawdown']['std']*100:>14.2f} {'-':>14} {'-':>14}")
        print(f"{'Alpha (Annual %)':<30} {summary['alpha']['mean']*100:>14.2f} "
              f"{summary['alpha']['std']*100:>14.2f} {'-':>14} {'-':>14}")
        print(f"{'Beta':<30} {summary['beta']['mean']:>14.4f} "
              f"{summary['beta']['std']:>14.4f} {'-':>14} {'-':>14}")
        print("="*90)
        
        summary_file = os.path.join(self.output_dir, 'aggregate_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        self.plot_aggregate_results()
    
    def plot_aggregate_results(self):
        """Plots comparison bar charts for all metrics across all phases."""
        if len(self.phase_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Performance Across All Phases', fontsize=16, fontweight='bold')
        
        phases = [r['phase'] for r in self.phase_results]
        cumulative_returns = [r['metrics']['cumulative_return'] * 100 for r in self.phase_results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in self.phase_results]
        max_drawdowns = [r['metrics']['max_drawdown'] * 100 for r in self.phase_results]
        sortino_ratios = [r['metrics']['sortino_ratio'] for r in self.phase_results]
        alphas = [r['metrics']['alpha'] * 100 for r in self.phase_results]
        betas = [r['metrics']['beta'] for r in self.phase_results]
        
        axes[0, 0].bar(phases, cumulative_returns, color='skyblue', edgecolor='black')
        axes[0, 0].axhline(0, color='red', linestyle='--')
        axes[0, 0].set_title('Cumulative Returns by Phase')
        axes[0, 0].set_xlabel('Phase')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        axes[0, 1].bar(phases, sharpe_ratios, color='green', edgecolor='black')
        axes[0, 1].axhline(1, color='orange', linestyle='--', label='Good (>1)')
        axes[0, 1].axhline(2, color='red', linestyle='--', label='Excellent (>2)')
        axes[0, 1].set_title('Sharpe Ratios by Phase')
        axes[0, 1].set_xlabel('Phase')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        axes[0, 2].bar(phases, max_drawdowns, color='red', edgecolor='black')
        axes[0, 2].set_title('Maximum Drawdowns by Phase')
        axes[0, 2].set_xlabel('Phase')
        axes[0, 2].set_ylabel('Max Drawdown (%)')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        axes[1, 0].bar(phases, sortino_ratios, color='purple', edgecolor='black')
        axes[1, 0].axhline(1, color='orange', linestyle='--')
        axes[1, 0].set_title('Sortino Ratios by Phase')
        axes[1, 0].set_xlabel('Phase')
        axes[1, 0].set_ylabel('Sortino Ratio')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        axes[1, 1].bar(phases, alphas, color='gold', edgecolor='black')
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_title('Alpha (Annual) by Phase')
        axes[1, 1].set_xlabel('Phase')
        axes[1, 1].set_ylabel('Alpha (%)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        axes[1, 2].bar(phases, betas, color='coral', edgecolor='black')
        axes[1, 2].axhline(1, color='red', linestyle='--', label='Market Risk')
        axes[1, 2].set_title('Beta by Phase')
        axes[1, 2].set_xlabel('Phase')
        axes[1, 2].set_ylabel('Beta')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'aggregate_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved aggregate plot: {plot_file}")
        plt.close()

def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("HATS-RL: GNN + Reinforcement Learning Stock Trading System")
    print("Using Stable-Baselines3 PPO Implementation")
    print("="*80)
    
    config = get_args()
    
    print("\n📊 Loading dataset...")
    dataset = StockDataset(config)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Companies: {dataset.num_companies}")
    print(f"  - Phases: {len(dataset.phases)}")
    
    rl_trainer = RLTrainer(config, dataset)
    rl_trainer.train_all_phases()
    
    print("\n" + "="*80)
    print("✅ ALL TRAINING COMPLETE!")
    print(f"📁 Results saved to: {rl_trainer.output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print("  - phase_X/ppo_agent.zip: Trained RL model for each phase")
    print("  - phase_X/results.json: Detailed metrics for each phase")
    print("  - phase_X/comprehensive_results.png: Visualizations")
    print("  - aggregate_summary.json: Overall statistics")
    print("  - aggregate_comparison.png: Cross-phase comparison")
    print("="*80)


if __name__ == '__main__':
    main()