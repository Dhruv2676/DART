import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, callbacks, losses #type:ignore
import numpy as np
import pandas as pd
import os
import json
import time
from tqdm import tqdm
import gc

from dataset import BaselineStockDataset
from trading_strategy import TradingStrategy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class TqdmCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to display training progress using tqdm.
    Updates the progress bar at the end of each epoch with loss and metrics.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tqdm_bar = None

    def on_train_begin(self, logs=None):
        self.tqdm_bar = tqdm(
            total=self.params['epochs'],
            desc='Training Epochs',
            unit='epoch'
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_metric_key = 'val_f1_macro' if 'val_f1_macro' in logs else 'val_accuracy'
        val_metric = logs.get(val_metric_key, 0)
        self_log_str = (
            f"loss: {logs.get('loss', 0):.4f} - "
            f"acc: {logs.get('accuracy', 0):.4f} - "
            f"val_loss: {logs.get('val_loss', 0):.4f} - "
            f"{val_metric_key}: {val_metric:.4f}"
        )
        self.tqdm_bar.set_postfix_str(self_log_str)
        self.tqdm_bar.update(1)

    def on_train_end(self, logs=None):
        if self.tqdm_bar:
            self.tqdm_bar.close()
            self.tqdm_bar = None

class GraphF1Score(tf.keras.metrics.Metric):
    def __init__(self, average='macro', name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.internal_f1 = metrics.F1Score(average=average, name=name)

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = tf.shape(y_pred)[-1]
        
        y_true_flat = tf.reshape(y_true, [-1, num_classes])
        y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
        
        self.internal_f1.update_state(y_true_flat, y_pred_flat, sample_weight)

    def result(self):
        return self.internal_f1.result()

    def reset_state(self):
        self.internal_f1.reset_state()

class GATLayer(layers.Layer):
    def __init__(self, units, num_heads=4, dropout_rate=0.5, **kwargs):
        """
        Initializes the model parameters and hyperparameters.
        """
        super(GATLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.ReLU()
        self.layer_norm = layers.LayerNormalization()

    def build(self, input_shape):
        """Initialize weights for linear transformation and attention mechanism."""
        feature_dim = input_shape[0][-1]
        
        # Weight matrix for linear transformation of node features
        self.W = self.add_weight(
            shape=(feature_dim, self.num_heads * self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='W_gat'
        )
        # Weight vector for self-attention mechanism
        self.a = self.add_weight(
            shape=(self.num_heads, 2 * self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='a_gat'
        )
        super(GATLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        Forward pass of the GAT layer.
        inputs: [node_features, neighbor_indices]
        """
        node_features, neighbors = inputs
        
        batch_size = tf.shape(node_features)[0]
        N = tf.shape(node_features)[1]
        
        # 1. Linear Transformation
        h = tf.matmul(node_features, self.W)
        h = tf.reshape(h, (batch_size, N, self.num_heads, self.units))
        
        # 2. Gather Neighbor Features
        valid_mask = tf.greater(neighbors, 0)
        adjusted_neighbors = tf.maximum(neighbors - 1, 0) # Adjust 1-based indices to 0-based
        
        # Prepare batch indices for gather_nd
        batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1, 1))
        batch_indices = tf.tile(batch_indices, (1, N, tf.shape(neighbors)[2]))
        
        # Gather features from neighbor nodes
        gather_indices = tf.stack([batch_indices, adjusted_neighbors], axis=-1)
        h_neighbors = tf.gather_nd(h, gather_indices) 
        
        # Prepare self features for concatenation
        h_self = tf.expand_dims(h, 2) 
        h_self = tf.tile(h_self, (1, 1, tf.shape(neighbors)[2], 1, 1)) 
        
        # 3. Attention Mechanism
        h_concat = tf.concat([h_self, h_neighbors], axis=-1) 
        
        # Calculate attention scores
        attention_scores = tf.einsum('bnkhd,hd->bnkh', h_concat, self.a)
        attention_scores = self.leaky_relu(attention_scores)
        
        # Apply mask to handle padding (non-existent neighbors)
        mask = tf.cast(valid_mask, tf.float32) 
        mask = tf.expand_dims(mask, -1) 
        
        # Softmax over neighbors
        attention_scores = attention_scores * mask + (1.0 - mask) * -1e9
        alpha = tf.nn.softmax(attention_scores, axis=2) 
        
        if training:
            alpha = self.dropout(alpha)
            
        # 4. Aggregation (Weighted sum of neighbors)
        alpha_exp = tf.expand_dims(alpha, -1)
        context = tf.reduce_sum(alpha_exp * h_neighbors, axis=2) 
        
        # Reshape to flatten heads
        context = tf.reshape(context, (batch_size, N, self.num_heads * self.units))
        
        # 5. Activation and Normalization
        output = self.activation(context)
        output = self.layer_norm(output)
        
        if training:
            output = self.dropout(output)
            
        return output

class GATKerasWrapper:
    """
    Wrapper for the Keras Model to align the GAT model's output structure
    (Timesteps, Nodes, Labels) with the evaluation method's expected structure 
    (Timesteps * Nodes, Labels).
    """
    def __init__(self, internal_model):
        self.internal_model = internal_model
        
    def summary(self):
        return self.internal_model.summary()
        
    def fit(self, *args, **kwargs):
        return self.internal_model.fit(*args, **kwargs)
        
    def evaluate(self, *args, **kwargs):
        return self.internal_model.evaluate(*args, **kwargs)
        
    def predict(self, x, **kwargs):
        # Flatten graph predictions to simple list of predictions
        predictions = self.internal_model.predict(x, **kwargs)
        num_labels = predictions.shape[-1]
        return predictions.reshape((-1, num_labels))

class GATModel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.lookback = getattr(dataset, 'lookback', 50)
        self.num_features = 1
        self.label_proportions = [3, 4, 3]
        self.num_labels = len(self.label_proportions)
        
        self.hyperparameters = {
            'model_name': "GATModel",
            'learning_rate': 5e-3,
            'epochs': 100,
            'batch_size': 32, 
            'lstm_units': 32,
            'gat_units': 16,
            'num_heads': 4,
            'dropout_rate': 0,
        }
        self.model = None 
    
    def build_model(self):
        """
        Builds the GAT model architecture.
        """
        N = self.dataset.num_companies
        
        # Inputs: Temporal windows and Neighbor adjacency structure
        input_windows = layers.Input(shape=(N, self.lookback, self.num_features), name='windows')
        input_neighbors = layers.Input(shape=(N, getattr(self.dataset, 'k', 10)), dtype=tf.int32, name='neighbors')
        
        # 1. Feature Extraction (LSTM)
        # Applied independently to each node to capture temporal dynamics
        lstm_out = layers.LSTM(self.hyperparameters['lstm_units'], return_sequences=False)
        node_features = layers.TimeDistributed(lstm_out)(input_windows) 

        # 2. GAT Layer 1
        gat_1 = GATLayer(
            units=self.hyperparameters['gat_units'], 
            num_heads=self.hyperparameters['num_heads'],
            dropout_rate=self.hyperparameters['dropout_rate']
        )([node_features, input_neighbors])
        
        # Residual Connection 1 (with projection if dimensions mismatch)
        if self.hyperparameters['lstm_units'] != (self.hyperparameters['gat_units'] * self.hyperparameters['num_heads']):
            projected_features = layers.Dense(self.hyperparameters['gat_units'] * self.hyperparameters['num_heads'])(node_features)
            gat_1 = layers.Add()([projected_features, gat_1])
        else:
            gat_1 = layers.Add()([node_features, gat_1])

        # 3. GAT Layer 2
        gat_2 = GATLayer(
            units=self.hyperparameters['gat_units'], 
            num_heads=self.hyperparameters['num_heads'],
            dropout_rate=self.hyperparameters['dropout_rate']
        )([gat_1, input_neighbors])

        # Residual Connection 2
        gat_2 = layers.Add()([gat_1, gat_2])

        # 4. Aggregation and Output
        # Combine original node features with graph-aggregated features
        combined = layers.Concatenate()([node_features, gat_2])
        
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dropout(self.hyperparameters['dropout_rate'])(x)
        
        output = layers.Dense(self.num_labels, activation='softmax', name='output')(x)
        
        internal_model = models.Model(inputs=[input_windows, input_neighbors], outputs=output, name='GAT_Internal')
        
        internal_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy', 
                GraphF1Score(average='macro', name='f1_macro'), 
                GraphF1Score(average='micro', name='f1_micro')
            ]
        )
        
        # Wrap the model for compatibility
        self.model = GATKerasWrapper(internal_model)
    
    def train(self, X_train, y_train, X_val, y_val, run_idx=0):
        """
        Trains the GAT model.
        Uses ReduceLROnPlateau scheduler and TQDM callback.
        """
        if self.model is None:
            return None
        print(f"\n--- Training Run {run_idx+1}/{NUM_RUNS} for {self.hyperparameters['epochs']} epochs ---")
        
        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_f1_macro', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
        tqdm_callback = TqdmCallback()
        
        # X_train is a list: [windows, neighbors]
        history = self.model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=self.hyperparameters['epochs'], 
            batch_size=self.hyperparameters['batch_size'], 
            shuffle=True, 
            callbacks=[lr_scheduler, tqdm_callback], 
            verbose=0
        )
        
        if run_idx == 0:
            print("\n------------------------\n")
        return history
    
    def evaluate(self, X_test, y_test, run_idx=0):
        """
        Evaluates the model on the test set.
        """
        if self.model is None:
            return None
    
        results = self.model.evaluate(X_test, y_test, batch_size=self.hyperparameters['batch_size'], verbose=0)
        
        if run_idx == 0:
            print(f"\n--- Evaluation (Run {run_idx+1}/{NUM_RUNS}) ---")
            print(f"  Test Loss:     {results[0]:.4f}")
            print(f"  Test Accuracy: {results[1]:.4f}")
            print(f"  Test F1-Macro: {results[2]:.4f}")
            print(f"  Test F1-Micro: {results[3]:.4f}")
            print("------------------------------\n")
        return results

def get_split_data(dataset, split='train'):
    """
    Prepares data for GAT training.
    """
    if split == 'train':
        windows_list = dataset.train_windows
        labels_list = dataset.train_labels
    elif split == 'dev':
        windows_list = dataset.dev_windows
        labels_list = dataset.dev_labels
    elif split == 'test':
        windows_list = dataset.test_windows
        labels_list = dataset.test_labels
    else:
        raise ValueError("Split must be 'train', 'dev', or 'test'")

    if not windows_list:
        print(f"  No data for {split} split.")
        return [], []

    # 1. Convert windows/labels to numpy 
    windows_array = np.array(windows_list)
    labels_array = np.array(labels_list)
    
    if windows_array.size == 0:
         return [], []
         
    # 2. Fetch Neighbors
    if hasattr(dataset, f"{split}_neighbors"):
        neighbors_list = getattr(dataset, f"{split}_neighbors")
        neighbors_array = np.array(neighbors_list)
    else:
        # Fallback logic if neighbors aren't explicitly exposed
        try:
             gen = dataset.get_batch(split)
             _, _, sample_neigh = next(gen)
             raise AttributeError("Neighbors attribute expected.") 
        except:
             # Dummy neighbors if extraction fails (self-loops/padding)
             T, N, _, _ = windows_array.shape
             K = getattr(dataset, 'k', 10)
             neighbors_array = np.zeros((T, N, K), dtype=np.int32)

    # Return structured input list and labels
    return [windows_array, neighbors_array], labels_array


NUM_RUNS = 5
def main():
    """
    Main execution loop:
    1. Iterates through all dataset phases.
    2. Runs multiple training trials (NUM_RUNS) per phase.
    3. Averages metrics across runs.
    4. Performs backtesting strategy on test predictions.
    5. Aggregates and saves final results in a json log file.
    """
    print("=" * 80)
    print("GAT Model Training on Stock Dataset (with TQDM & JSON Logging)")
    print("=" * 80)
    
    dataset = BaselineStockDataset()
    
    gat = GATModel(dataset)
    strategy = TradingStrategy(initial_balance=1_000_000, transaction_cost=0.001)
    
    overall_results = {
        'model_name': gat.hyperparameters['model_name'],
        'num_phases_trained': 0,
        'phase_results': [],
        'hyperparameters': gat.hyperparameters
    }
    
    num_phases = [11]
    print(f"\nFound {num_phases} market phases to process.")
    
    for phase_idx in num_phases:
        print("\n" + "=" * 80)
        print(f"PROCESSING PHASE {phase_idx}")
        print("=" * 80)
        
        dataset.set_phase(phase_idx)
        print(f"\nLoading and reshaping data for Phase {phase_idx}...")
        X_train, y_train = get_split_data(dataset, 'train')
        X_val, y_val = get_split_data(dataset, 'dev')
        X_test, y_test = get_split_data(dataset, 'test')
        
        print("Loading backtest data (log returns)...")
        try:
            flat_log_returns_test = dataset.get_log_returns_for_split('test')
            num_companies_test = dataset.num_companies
            
            if flat_log_returns_test.size == 0:
                raise ValueError("No log returns found for this split.")
            
            num_test_timesteps = len(dataset.test_windows)
            expected_len = num_test_timesteps * num_companies_test
            if flat_log_returns_test.size != expected_len:
                 raise ValueError(f"Shape mismatch: X_test implies {expected_len} data points, but got {flat_log_returns_test.size} returns.")

            print(f"  Backtest data loaded: {len(flat_log_returns_test)} return points ({num_test_timesteps} timesteps, {num_companies_test} companies).")
            backtest_data_ready = True
        except Exception as e:
            print(f"  ⚠️  Could not load backtest data for phase {phase_idx}. Backtest will be skipped.")
            print(f"  Error: {e}")
            backtest_data_ready = False
        
        test_dates = None
        benchmark_returns = None
        
        if backtest_data_ready:
            try:
                test_dates = dataset.get_dates_for_split('test')
                if test_dates is None or len(test_dates) == 0:
                    print("  ⚠️  No dates available for test split.")
                else:
                    print(f"  Dates loaded: {len(test_dates)} dates for test split.")
                
                benchmark_returns_flat = dataset.get_snp500_returns_for_split('test')
                if benchmark_returns_flat is None or benchmark_returns_flat.size == 0:
                    print("  ⚠️  No S&P 500 benchmark returns available for test split.")
                    benchmark_returns = None
                else:
                    benchmark_returns = pd.Series(benchmark_returns_flat[::dataset.num_companies], dtype=float)
                    print(f"  S&P 500 benchmark returns loaded: {len(benchmark_returns)} return points.")
            except Exception as e:
                print(f"  ⚠️  Could not load dates or benchmark returns: {e}")
                test_dates = None
                benchmark_returns = None
        
        run_logs_for_this_phase = []
        all_run_daily_returns = []
        for run_idx in range(NUM_RUNS):
            gat.build_model()
            if phase_idx == 0 and run_idx == 0:
                print("\n--- Model Summary ---")
                gat.model.summary()
                print("---------------------\n")
        
            start_time = time.time()
            history = gat.train(X_train, y_train, X_val, y_val, run_idx)
            training_time = time.time() - start_time
            
            test_results = gat.evaluate(X_test, y_test, run_idx)
            
            if test_results and history:
                best_epoch_index = np.argmax(history.history['val_f1_macro'])
                best_epoch = best_epoch_index + 1
                best_dev_f1 = history.history['val_f1_macro'][best_epoch_index]

                phase_log = {
                    "phase": phase_idx,
                    'run': run_idx + 1,
                    "training_time": training_time,
                    "best_epoch": int(best_epoch),
                    "best_dev_f1": float(best_dev_f1),
                    "test": {
                        "loss": float(test_results[0]),
                        "accuracy": float(test_results[1]),
                        "f1_macro": float(test_results[2]),
                        "f1_micro": float(test_results[3])
                    },
                    "backtest_metrics": {}
                }
                run_logs_for_this_phase.append(phase_log)
            else:
                print(f"  ⚠️  Skipping logging for phase {phase_idx} due to training/eval error.")
                continue
            
            if backtest_data_ready:
                print(f"--- Backtest (Run {run_idx+1}/{NUM_RUNS}) ---")
                test_predictions = gat.model.predict(X_test, batch_size=gat.hyperparameters['batch_size'], verbose=0)
                
                returns_series, value_history, daily_returns_with_dates = strategy.run_backtest(
                    predictions=test_predictions,
                    actual_returns=flat_log_returns_test,
                    num_companies=num_companies_test,
                    dates=test_dates
                )
                
                all_run_daily_returns.append(daily_returns_with_dates)
                backtest_metrics = strategy.calculate_metrics(returns_series, value_history, benchmark_returns_series=benchmark_returns)
                
                if run_idx == 0:
                    print(f"  Final Portfolio Value: ${backtest_metrics['final_value']:,.2f}")
                    print(f"  Cumulative Returns:     {backtest_metrics['cumulative_returns'] * 100:.2f}%")
                    print(f"  Sharpe Ratio:           {backtest_metrics['sharpe_ratio']:.4f}")
                    print(f"  Sortino Ratio:          {backtest_metrics['sortino_ratio']:.4f}")
                    print(f"  Max Drawdown:           {backtest_metrics['max_drawdown'] * 100:.2f}%")
                    print(f"  Alpha:                  {backtest_metrics['alpha']:.4f}")
                    print(f"  Beta:                   {backtest_metrics['beta']:.4f}")
                    print("------------------------------\n")
                phase_log["backtest_metrics"] = backtest_metrics
            
            print(f"  Cleaning up resources for [Phase {phase_idx}] - Run {run_idx+1}...")
            if 'history' in locals(): del history
            if 'test_predictions' in locals(): del test_predictions
            if gat.model is not None:
                del gat.model
                gat.model = None
            tf.keras.backend.clear_session()
            gc.collect()
    
        if run_logs_for_this_phase:
            avg_training_time = np.mean([log['training_time'] for log in run_logs_for_this_phase])
            avg_best_epoch = np.mean([log['best_epoch'] for log in run_logs_for_this_phase])
            avg_best_dev_f1 = np.mean([log['best_dev_f1'] for log in run_logs_for_this_phase])
            
            avg_test_loss = np.mean([log['test']['loss'] for log in run_logs_for_this_phase])
            avg_test_acc = np.mean([log['test']['accuracy'] for log in run_logs_for_this_phase])
            avg_test_f1_macro = np.mean([log['test']['f1_macro'] for log in run_logs_for_this_phase])
            avg_test_f1_micro = np.mean([log['test']['f1_micro'] for log in run_logs_for_this_phase])

            final_averaged_phase_log = {
                "phase": phase_idx,
                "training_time": avg_training_time,
                "best_epoch": avg_best_epoch,
                "best_dev_f1": avg_best_dev_f1,
                "test": {
                    "loss": avg_test_loss,
                    "accuracy": avg_test_acc,
                    "f1_macro": avg_test_f1_macro,
                    "f1_micro": avg_test_f1_micro
                },
                "backtest_metrics": {}
            }
            if backtest_data_ready and all_run_daily_returns:
                date_to_returns = {}
                for run_daily_returns in all_run_daily_returns:
                    for entry in run_daily_returns:
                        date = entry['date']
                        ret = entry['return']
                        if date not in date_to_returns:
                            date_to_returns[date] = []
                        date_to_returns[date].append(ret)
                
                averaged_daily_returns = []
                for date in sorted(date_to_returns.keys()):
                    avg_return = np.mean(date_to_returns[date])
                    averaged_daily_returns.append({
                        'date': date,
                        'return': float(avg_return)
                    })
                    
                final_averaged_phase_log['daily_returns'] = averaged_daily_returns
                
            if backtest_data_ready:
                all_backtest_metrics = [log['backtest_metrics'] for log in run_logs_for_this_phase if log['backtest_metrics']]
                if all_backtest_metrics:
                    avg_backtest_metrics = strategy.average_run_metrics(all_backtest_metrics)
                    final_averaged_phase_log["backtest_metrics"] = avg_backtest_metrics
                
            overall_results["phase_results"].append(final_averaged_phase_log)
            overall_results['num_phases_trained'] += 1
            
            print(f"--- Phase {phase_idx} Averages (across {NUM_RUNS} runs) ---")
            print(f"  Avg Test Loss:     {avg_test_loss:.4f}")
            print(f"  Avg Test F1-Macro: {avg_test_f1_macro:.4f}")
            
            if "cumulative_returns" in final_averaged_phase_log["backtest_metrics"]:
                avg_bt_metrics = final_averaged_phase_log["backtest_metrics"]
                print("  --- Avg Backtest ---")
                print(f"  Avg Cum. Returns:      {avg_bt_metrics['cumulative_returns'] * 100:.2f}%")
                print(f"  Avg Sharpe Ratio:      {avg_bt_metrics['sharpe_ratio']:.4f}")
                print(f"  Avg Max Drawdown:      {avg_bt_metrics['max_drawdown'] * 100:.2f}%")
                print(f"  Avg Alpha:              {avg_bt_metrics['alpha']:.4f}")
                print(f"  Avg Beta:               {avg_bt_metrics['beta']:.4f}")
            
            print("------------------------------------------")
        else:
            print(f"  ⚠️  No successful runs to average for phase {phase_idx}.")
        
        with open(os.path.join("./logs", f'gat_model_phase_{phase_idx}.json'), 'w') as f:
            json.dump(final_averaged_phase_log, f, indent=4)
        
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'GATModel_overall_result.json')
    with open(log_file_path, 'w') as f:
        json.dump(overall_results, f, indent=4)
    print(f"\n✅ Successfully saved results to {log_file_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - OVERALL RESULTS")
    print("=" * 80)
    
    all_test_metrics = [p["test"] for p in overall_results["phase_results"]]
    
    avg_loss = np.mean([m["loss"] for m in all_test_metrics])
    avg_acc = np.mean([m["accuracy"] for m in all_test_metrics])
    avg_f1_macro = np.mean([m["f1_macro"] for m in all_test_metrics])
    avg_f1_micro = np.mean([m["f1_micro"] for m in all_test_metrics])

    print(f"Average results across {overall_results['num_phases_trained']} phases:")
    print(f"  Average Test Loss:     {avg_loss:.4f}")
    print(f"  Average Test Accuracy: {avg_acc:.4f}")
    print(f"  Average Test F1-Macro: {avg_f1_macro:.4f}")
    print(f"  Average Test F1-Micro: {avg_f1_micro:.4f}")
    
    all_phase_backtest_metrics = [p["backtest_metrics"] for p in overall_results["phase_results"] if p["backtest_metrics"]]
    if all_phase_backtest_metrics:
        overall_avg_backtest_metrics = strategy.average_run_metrics(all_phase_backtest_metrics)
        print("\n--- Overall Average Backtest Metrics ---")
        print(f"  Avg Cumulative Returns: {overall_avg_backtest_metrics['cumulative_returns'] * 100:.2f}%")
        print(f"  Avg Final Value:        ${overall_avg_backtest_metrics['final_value']:,.2f}")
        print(f"  Avg Sharpe Ratio:       {overall_avg_backtest_metrics['sharpe_ratio']:.4f}")
        print(f"  Avg Sortino Ratio:      {overall_avg_backtest_metrics['sortino_ratio']:.4f}")
        print(f"  Avg Max Drawdown:       {overall_avg_backtest_metrics['max_drawdown'] * 100:.2f}%")
        print(f"  Avg Alpha:              {overall_avg_backtest_metrics['alpha']:.4f}")
        print(f"  Avg Beta:               {overall_avg_backtest_metrics['beta']:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()