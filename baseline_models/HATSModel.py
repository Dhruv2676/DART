import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics #type:ignore
import numpy as np
import os
import json
import time
from tqdm import tqdm
import pandas as pd

from dataset import BaselineStockDataset
from trading_strategy import TradingStrategy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class HATS(tf.keras.Model):
    def __init__(self, config):
        """
        Initializes the HATS model configuration and layers.
        
        Parameters:
        - config: Dictionary containing hyperparameters and model settings.
        """
        super(HATS, self).__init__()
        self.model_name = "HATS"
        self.uses_graph = True
        
        self.n_labels = len(config['label_proportions'])
        self.num_relations = config['num_relations']
        self.lookback = 50 
        self.neighbors_sample = config['neighbors_sample']
        self.node_feat_size = config['node_feat_size']
        
        self.hyperparameters = config
        
        # Initialize layers placeholders
        self.lstm = None
        self.lstm_dropout = None
        self.relation_embeddings = None
        self.state_attention_layers = []
        self.relation_attention = None
        self.prediction_layer = None
        
        self.build_layers()

    def build_layers(self):
        """
        Constructs the TensorFlow layers for the model.
        Includes LSTM, relation embeddings, attention layers, and prediction head.
        """
        node_feat_size = self.node_feat_size
        dropout_rate = self.hyperparameters['dropout_rate']
        
        reg = tf.keras.regularizers.l2(1e-4)
        
        # LSTM for feature extraction from time-series data
        self.lstm = layers.GRU(
            units=node_feat_size,
            return_sequences=False,
            name='lstm_feature_extractor',
            kernel_regularizer=reg,
        )
        self.lstm_dropout = layers.Dropout(dropout_rate, name='lstm_dropout')
        
        # Relation Embeddings: Learnable vector for each relation type
        self.relation_embeddings = self.add_weight(
            name="relation_embeddings",
            shape=(self.num_relations, 32), 
            initializer="glorot_uniform",
            trainable=True
        )
        
        # State Attention Layers (Node-level attention)
        # We create separate projection and scoring layers for each relation type
        self.relation_projection_layers = []
        self.state_attention_layers = []
        
        for rel_idx in range(self.num_relations):
            # Projection layer to transform neighbor features into the relation space
            proj_layer = layers.Dense(node_feat_size, activation=tf.nn.leaky_relu, 
                                      name=f'rel_projection_{rel_idx}', kernel_regularizer=reg)
            self.relation_projection_layers.append(proj_layer)
            
            # Attention scoring layer to calculate importance of each neighbor
            att_layer = layers.Dense(1, activation=tf.nn.leaky_relu, 
                                     name=f'state_attention_{rel_idx}', kernel_regularizer=reg)
            self.state_attention_layers.append(att_layer)
        
        # Relation Attention Layer (to aggregate across different relations)
        self.relation_attention = layers.Dense(1, activation=tf.nn.leaky_relu, name='relation_attention', kernel_regularizer=reg)
        
        # Final classification layer
        self.prediction_layer = layers.Dense(self.n_labels, activation='softmax', name='prediction')
    
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        
        Parameters:
        - inputs: List [X, neighbors]
            - X: (batch_size, num_companies, lookback, features)
            - neighbors: (batch_size, num_relations, num_companies, k)
        """
        X, neighbors = inputs
        
        # 1. Extract features from time series
        node_features = self.lstm_feature_extraction(X, training)
        
        # 2. Apply Hierarchical Attention (State + Relation)
        updated_features = self.graph_attention(node_features, neighbors, training)
        
        # 3. Prediction
        logits = self.prediction_layer(updated_features)
        
        return logits
    
    def lstm_feature_extraction(self, windows, training):
        """
        Processes the input windows using LSTM.
        Flattens the batch and company dimensions to process all nodes in parallel.
        """
        batch_size = tf.shape(windows)[0]
        num_companies = tf.shape(windows)[1]
        lookback = tf.shape(windows)[2]
        features = tf.shape(windows)[3]
        
        # Flatten: [batch, companies, ...] -> [batch * companies, ...]
        windows_reshaped = tf.reshape(windows, [batch_size * num_companies, lookback, features])
        
        lstm_out = self.lstm(windows_reshaped, training=training) # [batch * num_companies, feat]
        lstm_out = self.lstm_dropout(lstm_out, training=training)
        
        # Reshape back to separate batch and companies
        node_features = tf.reshape(lstm_out, [batch_size, num_companies, self.node_feat_size])
        
        # Add Zero Padding at index 0. 
        # This acts as a 'dummy node' for padding neighbors (index 0 in neighbors array).
        zero_padding = tf.zeros([batch_size, 1, self.node_feat_size], dtype=node_features.dtype)
        padded_node_features = tf.concat([zero_padding, node_features], axis=1)
        
        return padded_node_features
    
    def graph_attention(self, node_features, neighbors, training):
        """
        Aggregates information using hierarchical attention.
        Iterates through relations to apply State Attention, then applies Relation Attention.
        """
        # node_features: [batch, num_companies + 1, feat]
        # neighbors: [batch, num_relations, num_companies, k]
        
        relation_reps = []
        
        # Level 1: State (Node) Attention per relation
        for rel_idx in range(self.num_relations):
            # Extract neighbors for this relation: [batch, num_companies, k]
            neighbor_indices = neighbors[:, rel_idx, :, :]
            
            # Gather neighbor features. Result: [batch, num_companies, k, feat]
            neighbor_features = tf.gather(node_features, neighbor_indices, batch_dims=1)
            neighbor_features = self.relation_projection_layers[rel_idx](neighbor_features)
            
            # Current node features (exclude padding index 0)
            current_features = node_features[:, 1:, :]
            
            # Apply attention to aggregate neighbors for this relation
            relation_rep = self.apply_state_attention(
                current_features, neighbor_features, neighbor_indices, rel_idx, training
            )
            relation_reps.append(relation_rep)
        
        # Stack representations: [num_rel, batch, num_companies, feat]
        relation_reps = tf.stack(relation_reps, axis=0)
        
        current_features = node_features[:, 1:, :] 
        
        # Level 2: Relation Attention to combine all relation representations
        updated_features = self.apply_relation_attention(
            current_features, relation_reps, training
        )
        
        return updated_features
    
    def apply_state_attention(self, current_features, neighbor_features, neighbor_indices, rel_idx, training):
        """
        Calculates attention scores for neighbors and aggregates them.
        Includes masking for padded neighbors (index 0).
        """
        batch_size = tf.shape(current_features)[0]
        num_companies = tf.shape(current_features)[1]
        k = tf.shape(neighbor_features)[2]
        
        # Expand dims for broadcasting
        current_expanded = tf.expand_dims(current_features, axis=2)
        current_tiled = tf.tile(current_expanded, [1, 1, k, 1])
        
        # Get relation embedding and tile it
        rel_emb = self.relation_embeddings[rel_idx]
        rel_emb_expanded = tf.reshape(rel_emb, [1, 1, 1, -1])
        rel_emb_tiled = tf.tile(rel_emb_expanded, [batch_size, num_companies, k, 1])
        
        # Concatenate [Self, Neighbor, RelationEmbedding] for attention calculation
        attention_input = tf.concat([current_tiled, neighbor_features, rel_emb_tiled], axis=-1)
        
        # Compute raw scores
        attention_scores = self.state_attention_layers[rel_idx](attention_input) # Shape: [Batch, Comp, K, 1]
        
        # Masking: Set score of padding neighbors (index 0) to negative infinity
        mask = tf.cast(tf.equal(neighbor_indices, 0), tf.float32) 
        mask = tf.expand_dims(mask, -1) * -1e9 
        
        attention_scores += mask
        
        # Softmax over neighbors (dim 2)
        attention_weights = tf.nn.softmax(attention_scores, axis=2)
        
        # Weighted Sum aggregation
        aggregated = tf.reduce_sum(neighbor_features * attention_weights, axis=2)
        
        return aggregated
    
    def apply_relation_attention(self, current_features, relation_reps, training):
        """
        Aggregates representations from different relations using an attention mechanism.
        """
        batch_size = tf.shape(current_features)[0]
        num_companies = tf.shape(current_features)[1]
        
        # Transpose to [batch, num_companies, num_rel, feat]
        relation_reps_t = tf.transpose(relation_reps, [1, 2, 0, 3])
        
        # Expand current features for concatenation
        current_expanded = tf.expand_dims(current_features, axis=2)
        current_tiled = tf.tile(current_expanded, [1, 1, self.num_relations, 1])
        
        # Expand relation embeddings
        rel_embs = tf.reshape(self.relation_embeddings, [1, 1, self.num_relations, -1])
        rel_embs_tiled = tf.tile(rel_embs, [batch_size, num_companies, 1, 1])
        
        # Concatenate for attention input
        attention_input = tf.concat([current_tiled, relation_reps_t, rel_embs_tiled], axis=-1)
        
        # Calculate attention scores across relations
        attention_scores = self.relation_attention(attention_input)
        attention_weights = tf.nn.softmax(attention_scores, axis=2)
        
        # Weighted sum of relation representations
        relation_agg = tf.reduce_sum(relation_reps_t * attention_weights, axis=2)
        
        # Residual Connection with original node features
        updated_features = current_features + relation_agg
        
        return updated_features

class HATSModel:
    def __init__(self):
        self.label_proportions = [3, 4, 3]
        self.num_labels = len(self.label_proportions)
        
        self.hyperparameters = {
            'model_name': "HATS",
            'learning_rate': 4e-4,
            'epochs': 100,
            'dropout_rate': 0.4,
            'node_feat_size': 128,
            'neighbors_sample': 15,
            'batch_size': 32,
            'label_proportions': self.label_proportions,
            'num_relations': 0,
        }
        self.model = None
        self.dataset = None

    def build_model(self, num_relations):
        """
        Instantiates the HATS model, creates dummy inputs to initialize weights, 
        and compiles the model with optimizer and loss.
        """
        config = self.hyperparameters.copy()
        config['num_relations'] = num_relations
        config['neighbors_sample'] = 15 
        
        self.model = HATS(config)
        
        if self.dataset is not None:
            if hasattr(self.dataset, 'price_feature_list'):
                input_feat_size = len(self.dataset.price_feature_list)
            else:
                input_feat_size = 1 
            
            # Create dummy inputs to initialize the graph
            dummy_X = tf.zeros(
                (1, self.dataset.num_companies, self.dataset.lookback, input_feat_size), 
                dtype=tf.float32
            )
            dummy_neighbors = tf.zeros(
                (1, num_relations, self.dataset.num_companies, config['neighbors_sample']), 
                dtype=tf.int32
            )
            # Run one eager pass to build variables
            _ = self.model([dummy_X, dummy_neighbors])

        self.optimizer = optimizers.Adam(
            learning_rate=self.hyperparameters['learning_rate'],
            global_clipnorm=1.0
        )
        
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn,
                           metrics=['accuracy', metrics.F1Score(average='macro', name='f1_macro'), metrics.F1Score(average='micro', name='f1_micro')])
    
    @tf.function
    def _train_step(self, X, y, neighbors):
        """Graph-compiled training step for performance."""
        with tf.GradientTape() as tape:
            logits = self.model([X, neighbors], training=True)
            loss_value = self.loss_fn(y, logits)
        
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return logits, loss_value

    @tf.function
    def _test_step(self, X, y, neighbors):
        """Graph-compiled evaluation step."""
        logits = self.model([X, neighbors], training=False)
        loss_value = self.loss_fn(y, logits)
        return logits, loss_value

    @tf.function
    def _predict_step(self, X, neighbors):
        """Graph-compiled prediction step."""
        logits = self.model([X, neighbors], training=False)
        return logits

    def train_epoch(self, split='train'):
        """Runs one training epoch."""
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        data_gen = self.dataset.get_batch(split, shuffle=(split=='train'))
        
        for batch_windows, batch_labels, batch_neighbors in data_gen:
            X = tf.expand_dims(batch_windows, 0)
            y = tf.expand_dims(batch_labels, 0)
            neighbors = tf.expand_dims(batch_neighbors, 0)
            
            logits, loss_value = self._train_step(X, y, neighbors)
            
            pred_labels = tf.argmax(logits, axis=-1)
            true_labels = tf.argmax(y, axis=-1)
            acc = np.mean(pred_labels.numpy() == true_labels.numpy())
            
            total_loss += float(loss_value)
            total_acc += acc
            num_batches += 1
            
        return total_loss / num_batches, total_acc / num_batches

    def evaluate(self, split='dev'):
        """Evaluates the model on the specified split."""
        total_loss = 0
        all_preds = []
        all_labels = []
        
        data_gen = self.dataset.get_batch(split, shuffle=False)
        
        for batch_windows, batch_labels, batch_neighbors in data_gen:
            X = tf.expand_dims(batch_windows, 0)
            y = tf.expand_dims(batch_labels, 0)
            neighbors = tf.expand_dims(batch_neighbors, 0)
            
            logits, loss_value = self._test_step(X, y, neighbors)
            
            total_loss += float(loss_value)
            
            all_preds.append(logits.numpy().reshape(-1, self.num_labels))
            all_labels.append(batch_labels.reshape(-1, self.num_labels))
            
        if not all_preds:
            return [0.0, 0.0, 0.0, 0.0]

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        pred_classes = np.argmax(all_preds, axis=1)
        true_classes = np.argmax(all_labels, axis=1)
        
        acc = np.mean(pred_classes == true_classes)
        
        from sklearn.metrics import f1_score
        f1_macro = f1_score(true_classes, pred_classes, average='macro')
        f1_micro = f1_score(true_classes, pred_classes, average='micro')
        
        return [total_loss / (len(all_labels) // self.dataset.num_companies), acc, f1_macro, f1_micro]

    def predict(self, split='test'):
        """Generates predictions for the specified split."""
        all_preds = []
        data_gen = self.dataset.get_batch(split, shuffle=False)
        
        for batch_windows, _, batch_neighbors in data_gen:
            X = tf.expand_dims(batch_windows, 0)
            neighbors = tf.expand_dims(batch_neighbors, 0)
            
            logits = self._predict_step(X, neighbors)
            all_preds.append(logits.numpy()) 
            
        if not all_preds:
            return np.array([])
            
        concat_preds = np.concatenate(all_preds, axis=0) 
        flat_preds = concat_preds.reshape(-1, self.num_labels)
        return flat_preds


NUM_RUNS = 2
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
    print(f"HATS Model Training - Averaging {NUM_RUNS} runs per phase")
    print("=" * 80)
    
    dataset = BaselineStockDataset()
    hats_model = HATSModel()
    hats_model.dataset = dataset 
    
    strategy = TradingStrategy(initial_balance=1_000_000, transaction_cost=0.001)
    
    overall_results = {
        'model_name': hats_model.hyperparameters['model_name'],
        'num_phases_trained': 0,
        'num_runs_averaged': NUM_RUNS,
        'phase_results': [],
        'hyperparameters': hats_model.hyperparameters
    }
    
    num_phases = len(dataset.phases)
    print(f"\nFound {num_phases} market phases to process.")
    
    for phase_idx in range(num_phases):
        print("\n" + "=" * 80)
        print(f"PROCESSING PHASE {phase_idx}")
        print("=" * 80)
        
        dataset.set_phase(phase_idx)
        
        print("Loading backtest data (log returns)...")
        backtest_data_ready = False
        try:
            flat_log_returns_test = dataset.get_log_returns_for_split('test')
            num_companies_test = dataset.num_companies
            if flat_log_returns_test.size > 0:
                print(f"  Backtest data loaded: {len(flat_log_returns_test)} return points.")
                backtest_data_ready = True
        except Exception as e:
            print(f"  ⚠️  Could not load backtest data: {e}")
            
        test_dates = dataset.get_dates_for_split('test')
        benchmark_returns_flat = dataset.get_snp500_returns_for_split('test')
        benchmark_returns = pd.Series(benchmark_returns_flat) if benchmark_returns_flat is not None else None
        
        run_logs_for_this_phase = []
        all_run_daily_returns = []
        
        for run_idx in range(NUM_RUNS):
            print(f"\n--- Run {run_idx+1}/{NUM_RUNS} ---")
            
            hats_model.build_model(num_relations=dataset.num_relations)
            
            start_time = time.time()
            
            best_dev_f1 = -1
            best_epoch = 0
            patience = 20
            wait = 0
            
            temp_weights_path = f"temp_hats_best_{phase_idx}_{run_idx}.weights.h5"
            
            t = tqdm(range(hats_model.hyperparameters['epochs']), desc='Training', unit='epoch')
            for epoch in t:
                loss, acc = hats_model.train_epoch('train')
                
                val_metrics = hats_model.evaluate('dev')
                val_f1 = val_metrics[2]
                
                t.set_postfix_str(f"loss: {loss:.4f} - acc: {acc:.4f} - val_f1: {val_f1:.4f}")
                
                if val_f1 > best_dev_f1:
                    best_dev_f1 = val_f1
                    best_epoch = epoch + 1
                    wait = 0
                    hats_model.model.save_weights(temp_weights_path)
                else:
                    wait += 1
                    if wait >= patience:
                        break
            
            training_time = time.time() - start_time
            
            try:
                if os.path.exists(temp_weights_path):
                    hats_model.model.load_weights(temp_weights_path)
            except:
                print("  Warning: Could not load best weights, using final state.")
            
            test_results = hats_model.evaluate('test')
            
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
            
            if backtest_data_ready:
                test_predictions = hats_model.predict('test') 
                returns_series, value_history, daily_returns_with_dates = strategy.run_backtest(
                    predictions=test_predictions, 
                    actual_returns=flat_log_returns_test, 
                    num_companies=num_companies_test, 
                    dates=test_dates
                )
                
                all_run_daily_returns.append(daily_returns_with_dates)
                backtest_metrics = strategy.calculate_metrics(returns_series, value_history, benchmark_returns)
                phase_log["backtest_metrics"] = backtest_metrics
                
                if run_idx == 0:
                    print(f"  Final Portfolio Value: ${backtest_metrics['final_value']:,.2f}")
                    print(f"  Cumulative Returns:    {backtest_metrics['cumulative_returns'] * 100:.2f}%")
                    print(f"  Sharpe Ratio:          {backtest_metrics['sharpe_ratio']:.4f}")
                    print(f"  Sortino Ratio:          {backtest_metrics['sortino_ratio']:.4f}")
                    print(f"  Max Drawdown:          {backtest_metrics['max_drawdown'] * 100:.2f}%")
                    print(f"  Alpha:                 {backtest_metrics['alpha']:.4f}")
                    print(f"  Beta:                  {backtest_metrics['beta']:.4f}")
                    print("------------------------------\n")

            run_logs_for_this_phase.append(phase_log)
            
            if os.path.exists(temp_weights_path):
                os.remove(temp_weights_path)

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
    log_file_path = os.path.join(log_dir, 'HATSModel_overall_result.json')
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