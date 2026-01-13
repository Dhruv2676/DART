import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics #type:ignore
import numpy as np
import os
import json
import time
from tqdm import tqdm

from dataset import BaselineStockDataset
from trading_strategy import TradingStrategy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class TqdmCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to integrate tqdm progress bars with model training.
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


class CNN_LSTMModel:
    def __init__(self):
        """
        Initializes model configuration and hyperparameters.
        """
        self.lookback = 50
        self.num_features = 1
        self.label_proportions = [3, 4, 3]
        self.num_labels = len(self.label_proportions)
        
        self.hyperparameters = {
            'model_name': "CNN_LSTMModel",
            'learning_rate': 1e-4,
            'epochs': 100,
            'batch_size': 128,
            'cnn_filters': 32,
            'kernel_size': 3,
            'pool_size': 2,
            'lstm_units': 50
        }
        self.model = None
    
    def build_model(self):
        """
        Constructs the hybrid architecture: Input -> Conv1D -> MaxPool -> LSTM -> Dense.
        """
        input_shape = (self.lookback, self.num_features)
        
        inputs = layers.Input(shape=input_shape)
        
        # Convolutional layer for extracting local temporal features
        x = layers.Conv1D(
            filters=self.hyperparameters['cnn_filters'],
            kernel_size=self.hyperparameters['kernel_size'],
            activation='relu',
            padding='causal',  
            name='conv1d'
        )(inputs)
        
        # Downsampling layer
        x = layers.MaxPooling1D(
            pool_size=self.hyperparameters['pool_size'],
            name='max_pooling1d'
        )(x)
        
        # LSTM layer to capture long-term dependencies from CNN features
        x = layers.LSTM(
            units=self.hyperparameters['lstm_units'],
            activation='tanh',  
            name='lstm'
        )(x)  
        
        # Output classification dense layer
        outputs = layers.Dense(
            self.num_labels,
            activation='softmax',
            name='output'
        )(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name='LSTM_CNN')
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', metrics.F1Score(average='macro', name='f1_macro'), metrics.F1Score(average='micro', name='f1_micro')]
        )
    
    def train(self, X_train, y_train, X_val, y_val, run_idx=0):
        """
        Trains the model with Early Stopping and TQDM callbacks.
        """
        if self.model is None:
            return None
        print(f"\n--- Training Run {run_idx+1}/{NUM_RUNS} for {self.hyperparameters['epochs']} epochs ---")
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_macro',
            patience=20,
            restore_best_weights=True,
            mode='max'
        )
        
        tqdm_callback = TqdmCallback()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.hyperparameters['epochs'],
            batch_size=self.hyperparameters['batch_size'],
            shuffle=True,
            callbacks=[early_stopping, tqdm_callback],
            verbose=0
        )
        
        if run_idx == 0:
            print("\n------------------------\n")
        return history
    
    def evaluate(self, X_test, y_test, run_idx=0):
        """
        Evaluates the model on the test set and prints metrics for the first run.
        """
        if self.model is None:
            return None
    
        results = self.model.evaluate(
            X_test, y_test,
            batch_size=self.hyperparameters['batch_size'],
            verbose=0
        )
        
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
    Extracts and reshapes windowed data from the dataset for a specific split.
    Returns flattened arrays suitable for the model input (Batch, Lookback, Features).
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
        return np.array([]), np.array([])

    all_windows = np.array(windows_list)
    all_labels = np.array(labels_list)
        
    if all_windows.size == 0 or all_labels.size == 0:
        print(f"  Empty data for {split} split after conversion.")
        return np.array([]), np.array([])

    num_timesteps, num_companies, lookback, num_features = all_windows.shape 
    X = all_windows.reshape((-1, lookback, num_features))
    num_labels = all_labels.shape[-1]
    y = all_labels.reshape((-1, num_labels))
    
    return X, y


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
    print("LSTM-CNN Model Training on Stock Dataset (with TQDM & JSON Logging)")
    print("=" * 80)
    
    dataset = BaselineStockDataset()
    
    cnn_lstm_model = CNN_LSTMModel()
    strategy = TradingStrategy(initial_balance=1_000_000, transaction_cost=0.001)
    
    overall_results = {
        'model_name': cnn_lstm_model.hyperparameters['model_name'],
        'num_phases_trained': 0,
        'phase_results': [],
        'hyperparameters': cnn_lstm_model.hyperparameters
    }
    
    num_phases = len(dataset.phases)
    print(f"\nFound {num_phases} market phases to process.")
    
    for phase_idx in range(num_phases):
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
                    import pandas as pd
                    benchmark_returns = pd.Series(benchmark_returns_flat, dtype=float)
                    print(f"  S&P 500 benchmark returns loaded: {len(benchmark_returns)} return points.")
            except Exception as e:
                print(f"  ⚠️  Could not load dates or benchmark returns: {e}")
                test_dates = None
                benchmark_returns = None
        
        run_logs_for_this_phase = []
        all_run_daily_returns = []
        
        # Execute multiple runs for robustness
        for run_idx in range(NUM_RUNS):
            cnn_lstm_model.build_model()
            if phase_idx == 0 and run_idx == 0:
                print("\n--- Model Summary ---")
                cnn_lstm_model.model.summary()
                print("---------------------\n")
        
            start_time = time.time()
            history = cnn_lstm_model.train(X_train, y_train, X_val, y_val, run_idx)
            training_time = time.time() - start_time
            
            test_results = cnn_lstm_model.evaluate(X_test, y_test, run_idx)
            
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
                print(f"  ⚠️  Skipping logging for phase {phase_idx}, run {run_idx+1} due to training/eval error.")
                continue
            
            if backtest_data_ready:
                print(f"--- Backtest (Run {run_idx+1}/{NUM_RUNS}) ---")
                test_predictions = cnn_lstm_model.model.predict(X_test, batch_size=cnn_lstm_model.hyperparameters['batch_size'], verbose=0)
                
                returns_series, value_history, daily_returns_with_dates = strategy.run_backtest(predictions=test_predictions, actual_returns=flat_log_returns_test, num_companies=num_companies_test, dates=test_dates)
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
        
        # Aggregate results across runs
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
    log_file_path = os.path.join(log_dir, 'CNN_LSTMModel_overall_result.json')
    with open(log_file_path, 'w') as f:
        json.dump(overall_results, f, indent=4)
    print(f"\n✅ Successfully saved results to {log_file_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - OVERALL RESULTS")
    print("=" * 80)
    
    if overall_results["num_phases_trained"] > 0:
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
    else:
        print("  No phases were successfully trained.")
        
    print("=" * 80)

if __name__ == '__main__':
    main()