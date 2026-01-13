import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, Input, activations #type:ignore
import numpy as np
import os
import json
import time
from tqdm import tqdm

from dataset import BaselineStockDataset
from trading_strategy import TradingStrategy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
            f"acc: {logs.get('categorical_accuracy', 0):.4f} - "
            f"val_loss: {logs.get('val_loss', 0):.4f} - "
            f"{val_metric_key}: {val_metric:.4f}"
        )
        self.tqdm_bar.set_postfix_str(self_log_str)
        self.tqdm_bar.update(1)

    def on_train_end(self, logs=None):
        if self.tqdm_bar:
            self.tqdm_bar.close()
            self.tqdm_bar = None

class GraphF1Score(metrics.Metric):
    def __init__(self, num_classes, average='macro', name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.internal_f1 = metrics.F1Score(average=average, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_flat = tf.reshape(y_true, (-1, self.num_classes))
        y_pred_flat = tf.reshape(y_pred, (-1, self.num_classes))
        self.internal_f1.update_state(y_true_flat, y_pred_flat, sample_weight)

    def result(self):
        return self.internal_f1.result()

    def reset_state(self):
        self.internal_f1.reset_state()

class GCNLayer(layers.Layer):
    def __init__(self, output_dim, num_relations, activation='relu', dropout_rate=0.0, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        """Initialize kernels for relations and self-loops."""
        input_dim = input_shape[0][-1]
        
        # Kernel for neighbor messages (one per relation type)
        self.relation_kernels = self.add_weight(
            shape=(self.num_relations, input_dim, self.output_dim),
            initializer='glorot_uniform',
            name='relation_kernels',
            trainable=True
        )
        
        # Kernel for self-loop message
        self.self_kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer='glorot_uniform',
            name='self_kernel',
            trainable=True
        )
        
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            name='bias',
            trainable=True
        )
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass of the GCN layer.
        inputs: [node_features, neighbor_indices]
        """
        node_features, neighbor_indices = inputs
        
        batch_size = tf.shape(node_features)[0]
        feature_dim = tf.shape(node_features)[2]
        
        # Pad features to handle potentially 0-index padding in neighbor_indices
        zero_pad = tf.zeros((batch_size, 1, feature_dim), dtype=node_features.dtype)
        padded_features = tf.concat([zero_pad, node_features], axis=1)
        
        # Gather features for all neighbors
        neighbor_features = tf.gather(padded_features, neighbor_indices, batch_dims=1)
        # Average features across the 'k' sampled neighbors
        aggregated_neighbors = tf.reduce_mean(neighbor_features, axis=3)
        
        # Compute messages from neighbors using relation kernels
        # Einstein summation: (Batch, Rel, Node, Dim) * (Rel, Dim, Out) -> (Batch, Rel, Node, Out)
        neighbor_messages = tf.einsum('brnd,rdo->brno', aggregated_neighbors, self.relation_kernels)
        total_neighbor_message = tf.reduce_sum(neighbor_messages, axis=1)
        
        # Compute message from self
        self_message = tf.matmul(node_features, self.self_kernel)
        
        # Combine messages and add bias
        output = total_neighbor_message + self_message + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
            
        return self.dropout(output)

def get_weighted_loss(class_weights_tensor):
    """
    Returns a weighted categorical crossentropy loss function.
    Useful for handling class imbalance.
    """
    def weighted_categorical_crossentropy(y_true, y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        base_loss = cce(y_true, y_pred)
        
        # Apply weights based on true class
        true_indices = tf.argmax(y_true, axis=-1)
        weights = tf.gather(class_weights_tensor, true_indices)
        
        return base_loss * weights
    return weighted_categorical_crossentropy

class GCNModel:
    def __init__(self, dataset):
        """
        Initializes model parameters and class weights based on dataset statistics.
        """
        self.dataset = dataset
        self.label_proportions = dataset.label_proportion
        self.num_labels = len(self.label_proportions)
        
        # Calculate Class Weights to handle imbalance
        total = sum(self.label_proportions)
        self.class_weights_dict = {
            i: total / (len(self.label_proportions) * prop) 
            for i, prop in enumerate(self.label_proportions)
        }
        self.class_weights_tensor = tf.constant(
            [self.class_weights_dict[i] for i in range(self.num_labels)], 
            dtype=tf.float32
        )
        
        self.hyperparameters = {
            'model_name': "GCNModel",
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 1,
            'lstm_units': 64,
            'gcn_units': 64,
            'dropout_rate': 0.5,
        }
        self.model = None
    
    def build_model(self, num_companies, lookback, num_features, num_relations, neighbors_k):
        """
        Constructs the GCN model architecture.
        Structure: Input -> LSTM -> GCN Layer 1 -> GCN Layer 2 -> Softmax Output.
        """
        window_input = Input(shape=(num_companies, lookback, num_features), name='window_input')
        neighbor_input = Input(shape=(num_relations, num_companies, neighbors_k), dtype=tf.int32, name='neighbor_input')
        
        # 1. LSTM Layer (Time Aggregation)
        # Reshape to treat all companies as a single batch for LSTM
        x_flat = layers.Lambda(lambda x: tf.reshape(x, (-1, lookback, num_features)))(window_input)
        x_encoded = layers.LSTM(self.hyperparameters['lstm_units'], return_sequences=False)(x_flat)
        x_encoded = layers.Dropout(self.hyperparameters['dropout_rate'])(x_encoded)
        # Reshape back to (Batch, Companies, Features)
        x_nodes = layers.Lambda(lambda x: tf.reshape(x, (-1, num_companies, self.hyperparameters['lstm_units'])))(x_encoded)
        
        # 2. GCN Layers (Spatial Aggregation)
        x_gcn = GCNLayer(
            output_dim=self.hyperparameters['gcn_units'], 
            num_relations=num_relations,
            activation='relu',
            dropout_rate=self.hyperparameters['dropout_rate']
        )([x_nodes, neighbor_input])
        
        x_gcn = GCNLayer(
            output_dim=self.hyperparameters['gcn_units'], 
            num_relations=num_relations,
            activation='relu',
            dropout_rate=self.hyperparameters['dropout_rate']
        )([x_gcn, neighbor_input])
        
        # 3. Output Layer
        output = layers.Dense(self.num_labels, activation='softmax')(x_gcn)
        
        self.model = models.Model(inputs=[window_input, neighbor_input], outputs=output, name='StockGCN')
        
        # Compile with weighted loss and custom graph metrics
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss=get_weighted_loss(self.class_weights_tensor),
            metrics=[
                metrics.CategoricalAccuracy(name='categorical_accuracy'),
                GraphF1Score(num_classes=self.num_labels, average='macro', name='f1_macro'),
                GraphF1Score(num_classes=self.num_labels, average='micro', name='f1_micro')
            ]
        )
    
    def train(self, train_gen, val_gen, train_steps, val_steps, run_idx=0):
        """
        Trains the GCN model using generators.
        """
        if self.model is None: return None

        print(f"\n--- Training Run {run_idx+1}/{NUM_RUNS} for {self.hyperparameters['epochs']} epochs ---")
        
        # Early stopping and LR reduction based on F1 Macro score
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_macro', 
            patience=20,
            restore_best_weights=True, 
            mode='max',
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_macro', 
            factor=0.5, 
            patience=7, 
            mode='max', 
            min_lr=1e-6, 
            verbose=0
        )
        
        callbacks_list = [early_stopping, reduce_lr]
        if run_idx == 0:
            callbacks_list.append(TqdmCallback())
            
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=self.hyperparameters['epochs'],
            callbacks=callbacks_list,
            verbose=0
        )
        
        if run_idx == 0:
            print("\n------------------------\n")
        return history
    
    def evaluate(self, test_gen, test_steps, run_idx=0):
        """
        Evaluates the model on the test generator.
        """
        if self.model is None: return None
        
        results = self.model.evaluate(test_gen, steps=test_steps, verbose=0)
        if run_idx == 0:
            print(f"\n--- Evaluation (Run {run_idx+1}/{NUM_RUNS}) ---")
            print(f"  Test Loss:     {results[0]:.4f}")
            print(f"  Test Accuracy: {results[1]:.4f}")
            print(f"  Test F1-Macro: {results[2]:.4f}")
            print(f"  Test F1-Micro: {results[3]:.4f}")
            print("------------------------------\n")
        return results

def data_generator(dataset, split, shuffle=False):
    """
    Generator function to yield batches of data in the format required by the GCN.
    Yields: ([windows, neighbors], labels)
    """
    while True:
        if split == 'train' and len(dataset.train_windows) == 0: break
        
        for windows, labels, neighbors in dataset.get_batch(split, shuffle=shuffle):
            x_win = np.expand_dims(windows, axis=0)
            x_neigh = np.expand_dims(neighbors, axis=0)
            y = np.expand_dims(labels, axis=0)
            yield (x_win, x_neigh), y


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
    print(f"GCN Model Training - Averaging {NUM_RUNS} runs per phase")
    print("=" * 80)
    
    print("Initializing Dataset...")
    dataset = BaselineStockDataset()
    
    gcn_manager = GCNModel(dataset)
    strategy = TradingStrategy(initial_balance=1_000_000, transaction_cost=0.001)
    
    overall_results = {
        'model_name': gcn_manager.hyperparameters['model_name'],
        'num_phases_trained': 0,
        'num_runs_averaged': NUM_RUNS,
        'phase_results': [],
        'hyperparameters': gcn_manager.hyperparameters
    }
    
    num_phases = len(dataset.phases)
    print(f"\nFound {num_phases} market phases to process.")
    
    for phase_idx in range(num_phases):
        print("\n" + "=" * 80)
        print(f"PROCESSING PHASE {phase_idx}")
        print("=" * 80)
        
        dataset.set_phase(phase_idx)
        
        train_steps = len(dataset.train_windows)
        val_steps = len(dataset.dev_windows)
        test_steps = len(dataset.test_windows)
        
        if train_steps == 0:
            print("Skipping phase due to empty data.")
            continue
            
        print("Loading backtest data (log returns)...")
        try:
            flat_log_returns_test = dataset.get_log_returns_for_split('test')
            num_companies_test = dataset.num_companies
            
            if flat_log_returns_test.size == 0:
                raise ValueError("No log returns found for this split.")
            
            expected_len = test_steps * num_companies_test
            if flat_log_returns_test.size != expected_len:
                 raise ValueError(f"Shape mismatch: X_test implies {expected_len} data points, but got {flat_log_returns_test.size} returns.")

            print(f"  Backtest data loaded: {len(flat_log_returns_test)} return points ({test_steps} timesteps, {num_companies_test} companies).")
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
                    import pandas as pd
                    benchmark_returns = pd.Series(benchmark_returns_flat, dtype=float)
                    print(f"  S&P 500 benchmark returns loaded: {len(benchmark_returns)} return points.")
            except Exception as e:
                print(f"  ⚠️  Could not load dates or benchmark returns: {e}")
                test_dates = None
                benchmark_returns = None
            
        run_logs_for_this_phase = []
        all_run_daily_returns = []
        for run_idx in range(NUM_RUNS):
            train_gen = data_generator(dataset, 'train', shuffle=True)
            val_gen = data_generator(dataset, 'dev', shuffle=False)
            test_gen = data_generator(dataset, 'test', shuffle=False)
            
            # Initialize model if needed (or rebuild for new run)
            if gcn_manager.model is None:
                gen = dataset.get_batch('train', shuffle=False)
                w, l, n = next(gen)
                gcn_manager.build_model(
                    num_companies=w.shape[0], 
                    lookback=w.shape[1], 
                    num_features=w.shape[2], 
                    num_relations=n.shape[0], 
                    neighbors_k=n.shape[2]
                )
            else:
                gen = dataset.get_batch('train', shuffle=False)
                w, l, n = next(gen)
                gcn_manager.build_model(
                    num_companies=w.shape[0], 
                    lookback=w.shape[1], 
                    num_features=w.shape[2], 
                    num_relations=n.shape[0], 
                    neighbors_k=n.shape[2]
                )

            if phase_idx == 0 and run_idx == 0:
                print("\n--- Model Summary ---")
                gcn_manager.model.summary()
                print("---------------------\n")
            
            start_time = time.time()
            history = gcn_manager.train(train_gen, val_gen, train_steps, val_steps, run_idx)
            training_time = time.time() - start_time
            
            test_results = gcn_manager.evaluate(test_gen, test_steps, run_idx)
            
            if test_results and history:
                best_epoch_index = np.argmax(history.history['val_f1_macro'])
                best_epoch = best_epoch_index + 1
                best_dev_f1 = history.history['val_f1_macro'][best_epoch_index]

                phase_log = {
                    "phase": phase_idx,
                    "run": run_idx + 1,
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
                print(f"  ⚠️  Skipping logging for Phase {phase_idx}, Run {run_idx+1} due to error.")
            
            if backtest_data_ready:
                print(f"--- Backtest (Run {run_idx+1}/{NUM_RUNS}) ---")
                pred_gen = data_generator(dataset, 'test', shuffle=False)
                predictions_batched = gcn_manager.model.predict(pred_gen, steps=test_steps, verbose=0)
                flat_predictions = predictions_batched.reshape(-1, len(dataset.label_proportion))
                
                returns_series, value_history, daily_returns_with_dates = strategy.run_backtest(
                    predictions=flat_predictions, 
                    actual_returns=flat_log_returns_test, 
                    num_companies=num_companies_test, 
                    dates=test_dates
                )
                all_run_daily_returns.append(daily_returns_with_dates)
                backtest_metrics = strategy.calculate_metrics(returns_series, value_history, benchmark_returns_series=benchmark_returns)
                
                if run_idx == 0:
                    print(f"  Final Portfolio Value: ${backtest_metrics['final_value']:,.2f}")
                    print(f"  Cumulative Returns:    {backtest_metrics['cumulative_returns'] * 100:.2f}%")
                    print(f"  Sharpe Ratio:          {backtest_metrics['sharpe_ratio']:.4f}")
                    print(f"  Sortino Ratio:          {backtest_metrics['sortino_ratio']:.4f}")
                    print(f"  Max Drawdown:          {backtest_metrics['max_drawdown'] * 100:.2f}%")
                    print(f"  Alpha:                 {backtest_metrics['alpha']:.4f}")
                    print(f"  Beta:                  {backtest_metrics['beta']:.4f}")
                    print("------------------------------\n")
                phase_log["backtest_metrics"] = backtest_metrics

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
                print(f"  Avg Cum. Returns:    {avg_bt_metrics['cumulative_returns'] * 100:.2f}%")
                print(f"  Avg Sharpe Ratio:      {avg_bt_metrics['sharpe_ratio']:.4f}")
                print(f"  Avg Max Drawdown:      {avg_bt_metrics['max_drawdown'] * 100:.2f}%")
                print(f"  Avg Alpha:           {avg_bt_metrics['alpha']:.4f}")
                print(f"  Avg Beta:            {avg_bt_metrics['beta']:.4f}")
            print("------------------------------------------")
        else:
            print(f"  ⚠️  No successful runs to average for phase {phase_idx}.")
    
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'GCNModel_overall_result.json')
    with open(log_file_path, 'w') as f:
        json.dump(overall_results, f, indent=4)
    print(f"\n✅ Successfully saved results to {log_file_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - OVERALL RESULTS")
    print("=" * 80)
    
    all_test_metrics = [p["test"] for p in overall_results["phase_results"]]
    
    if all_test_metrics:
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