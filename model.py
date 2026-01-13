import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import json
from tensorflow.keras import layers # type: ignore
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

class GNNModel(keras.Model):
    def __init__(self, config):
        """
        Initializes the GNN model components.
        """
        super(GNNModel, self).__init__()
        self.config = config
        self.num_relations = 58 + 1 
        
        self.lstm = None
        self.lstm_dropout = None
        self.relation_embeddings = None
        self.state_attention_layers = []
        self.relation_attention = None
        self.prediction_dropout = None
        self.prediction_layer = None
    
    def build_layers(self):
        """
        Constructs the TensorFlow layers for the model.
        Initializes LSTM, embeddings, attention mechanisms, and prediction head.
        """
        self.lstm = layers.LSTM(units=self.config.node_feat_size, return_sequences=False, name='lstm_feature_extractor')
        self.lstm_dropout = layers.Dropout(self.config.dropout, name='lstm_dropout')
        
        self.relation_embeddings = tf.eye(self.num_relations, dtype=tf.float32)
        
        if self.config.use_feat_attention:
            for rel_idx in range(self.num_relations):
                layer = layers.Dense(1, activation='relu', name=f'state_attention_{rel_idx}')
                self.state_attention_layers.append(layer)
        
        if self.config.use_rel_attention:
            self.relation_attention = layers.Dense(1, activation='relu', name='relation_attention')
        
        self.prediction_dropout = layers.Dropout(self.config.dropout, name='pred_dropout')
        self.prediction_layer = layers.Dense(len(self.config.label_proportion), activation=None, name='prediction_layer')
    
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        
        Parameters:
        - inputs: dict containing 'windows' (features) and 'neighbors' (graph structure).
        - training: bool, indicates whether the model is in training mode.
        
        Returns:
        - dict: contains logits, softmax predictions, and the final GNN state vector.
        """
        windows = inputs['windows']
        neighbors = inputs['neighbors']
        
        node_features = self.lstm_feature_extraction(windows, training)
        updated_features = self.graph_attention(node_features, neighbors, training)
        logits = self.prediction_layer(updated_features)
        predictions = tf.nn.softmax(logits)
        
        return {'logits': logits, 'predictions': predictions, 'gnn_state': updated_features}
        
    def lstm_feature_extraction(self, windows, training):
        """
        Processes temporal stock data using an LSTM.
        
        Parameters:
        - windows: tensor, historical window data for all nodes.
        - training: bool, training mode flag.
        
        Returns:
        - node_features: tensor, encoded features with zero padding for the first element.
        """
        lstm_out = self.lstm(windows, training=training)
        
        zero_padding = tf.zeros([1, self.config.node_feat_size], dtype=tf.float32)
        node_features = tf.concat([zero_padding, lstm_out], axis=0)
        return node_features
    
    def graph_attention(self, node_features, neighbors, training):
        """
        Aggregates information from neighbors using hierarchical attention.
        
        Parameters:
        - node_features: tensor, features for all nodes.
        - neighbors: tensor, indices of neighbors for each relation.
        - training: bool, training mode flag.
        
        Returns:
        - updated_features: tensor, final node embeddings after graph aggregation.
        """
        batch_size = tf.shape(node_features)[0] - 1
        
        relation_reps = []
        for rel_idx in range(self.num_relations):
            neighbor_indices = neighbors[rel_idx]
            neighbor_features = tf.gather(node_features, neighbor_indices)
            current_features = node_features[1:]
            if self.config.use_feat_attention:
                relation_rep = self.apply_state_attention(current_features, neighbor_features, rel_idx, training)
            else:
                relation_rep = tf.reduce_mean(neighbor_features, axis=1)
            relation_reps.append(relation_rep)
        relation_reps = tf.stack(relation_reps, axis=0)
        
        current_features = node_features[1:]
        if self.config.use_rel_attention:
            updated_features = self.apply_relation_attention(current_features, relation_reps, training)
        else:
            relation_agg = tf.reduce_mean(relation_reps, axis=0)
            updated_features = current_features + relation_agg
        
        return updated_features
    
    def apply_state_attention(self, current_features, neighbor_features, rel_idx, training):
        """
        Applies node-level attention (State Attention) to weight specific neighbors.
        
        Parameters:
        - current_features: tensor, features of the target nodes.
        - neighbor_features: tensor, features of the sampled neighbors.
        - rel_idx: int, index of the relation type being processed.
        - training: bool, training mode flag.
        
        Returns:
        - aggregated: tensor, weighted sum of neighbor features.
        """
        batch_size = tf.shape(current_features)[0]
        
        current_expanded = tf.expand_dims(current_features, axis=1)
        current_tiled = tf.tile(current_expanded, [1, self.config.neighbors_sample, 1])
        
        rel_emb = self.relation_embeddings[rel_idx]
        rel_emb_expanded = tf.reshape(rel_emb, [1, 1, -1])
        rel_emb_expanded = tf.tile(rel_emb_expanded, [batch_size, self.config.neighbors_sample, 1])
        attention_input = tf.concat([current_tiled, neighbor_features, rel_emb_expanded], axis=2)
        
        attention_scores = self.state_attention_layers[rel_idx](attention_input)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        aggregated = tf.reduce_sum(neighbor_features * attention_weights, axis=1)
        
        return aggregated
    
    def apply_relation_attention(self, current_features, relation_reps, training):
        """
        Applies relation-level attention to weight different relation types.
        
        Parameters:
        - current_features: tensor, features of the target nodes.
        - relation_reps: tensor, aggregated features from each relation type.
        - training: bool, training mode flag.
        
        Returns:
        - tensor: final embeddings combining self-features and relation-aggregated features.
        """
        batch_size = tf.shape(current_features)[0]
        
        current_expanded = tf.expand_dims(current_features, axis=0)
        current_tiled = tf.tile(current_expanded, [self.num_relations, 1, 1])
        
        rel_embs = tf.expand_dims(self.relation_embeddings, axis=1)
        rel_embs_expanded = tf.tile(rel_embs, [1, batch_size, 1])
        attention_input = tf.concat([current_tiled, relation_reps, rel_embs_expanded], axis=2)
        
        attention_scores = self.relation_attention(attention_input)
        attention_weights = tf.nn.softmax(attention_scores, axis=0)
        relation_agg = tf.reduce_sum(relation_reps * attention_weights, axis=0)
        
        return current_features + relation_agg
    
    def get_gnn_state(self, timestep, split='test'):
        """
        Retrieves the GNN latent state for a specific timestep and split.
        
        Parameters:
        - timestep: int, the time index to query.
        - split: str, 'train', 'dev', or 'test'.
        
        Returns:
        - state_dict: dict, includes GNN embeddings, raw windows, and labels.
        """
        if split == 'train':
            windows = self.dataset.train_windows[timestep]
            labels = self.dataset.train_labels[timestep]
            neighbors = self.dataset.train_sampled_neighbors[timestep]
        elif split == 'dev':
            windows = self.dataset.dev_windows[timestep]
            labels = self.dataset.dev_labels[timestep]
            neighbors = self.dataset.dev_sampled_neighbors[timestep]
        else:  # test
            windows = self.dataset.test_windows[timestep]
            labels = self.dataset.test_labels[timestep]
            neighbors = self.dataset.test_sampled_neighbors[timestep]
        
        state_dict = self.model.get_market_state(windows, neighbors)
        
        state_dict['true_labels'] = labels
        state_dict['raw_windows'] = windows
        state_dict['timestep'] = timestep
        state_dict['split'] = split
        
        return state_dict
    
    def get_portfolio_features(self, state_dict):
        """
        Calculates portfolio-level statistics from GNN predictions and embeddings.
        
        Parameters:
        - state_dict: dict, output from the model containing predictions and embeddings.
        
        Returns:
        - portfolio_features: dict, aggregate metrics (e.g., bullish ratio, mean confidence).
        """
        gnn_state = state_dict['gnn_state']
        predictions = state_dict['predictions']
        
        portfolio_features = {
            'bullish_ratio': float((predictions[:, -1] > 0.5).sum() / len(predictions)),
            'bearish_ratio': float((predictions[:, 0] > 0.5).sum() / len(predictions)),
            'neutral_ratio': float((predictions[:, 1] > 0.5).sum() / len(predictions)),
            'mean_embedding': np.mean(gnn_state, axis=0),  # [128]
            'std_embedding': np.std(gnn_state, axis=0),    # [128]
            'avg_confidence': float(np.mean(np.max(predictions, axis=1))),
            'max_confidence': float(np.max(np.max(predictions, axis=1))),
            'min_confidence': float(np.min(np.max(predictions, axis=1))),
            'num_companies': len(predictions)
        }
        return portfolio_features
    
class Trainer:
    """
    Handles the training, evaluation, and checkpointing loop for the GNN model.
    """
    def __init__(self, model, dataset, config, phase_idx=None):
        """
        Initializes the Trainer.
        
        Parameters:
        - model: instance of GNNModel.
        - dataset: instance of StockDataset.
        - config: configuration object.
        - phase_idx: int, current phase index (optional).
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.phase_idx = phase_idx if phase_idx is not None else dataset.current_phase
        
        self.optimizer = keras.optimizers.Adam(learning_rate=config.lr) 
        self.loss_function = keras.losses.CategoricalCrossentropy(from_logits=True) 
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.train_acc_metric = keras.metrics.CategoricalAccuracy(name='train_acc')
        
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.print_step = 100
        self.eval_step = 10
        
        phase_checkpoint_dir = os.path.join(config.checkpoint_dir, f'phase_{self.phase_idx}')
        os.makedirs(phase_checkpoint_dir, exist_ok=True)
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, phase_checkpoint_dir, max_to_keep=config.max_to_keep)
    
    @tf.function
    def train_step(self, windows, labels, neighbors):
        """
        Performs a single training step: forward pass, loss calculation, gradient computation, and update.
        
        Parameters:
        - windows, labels, neighbors: input tensors for the batch.
        
        Returns:
        - total_loss: float, sum of cross-entropy and L2 regularization loss.
        """
        with tf.GradientTape() as tape:
            outputs = self.model({'windows': windows, 'neighbors': neighbors}, training=True)
            logits = outputs['logits']
            loss = self.loss_function(labels, logits)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables if 'bias' not in v.name]) * self.config.weight_decay
            total_loss = loss + l2_loss
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Debugging block: Prints gradient info on the very first iteration
        if self.optimizer.iterations == 0:
            tf.print("--- GRADIENT CHECK (First Step) ---")
            tf.print("Loss:", total_loss)
            
            is_none = [g is None for g in gradients]
            if tf.reduce_any(is_none):
                tf.print("WARNING: At least one gradient is None!")
                
            valid_gradients = [g for g in gradients if g is not None]
            global_norm = tf.linalg.global_norm(valid_gradients)
            tf.print("Global Norm (Before Clip):", global_norm)
        
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_max_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss_metric.update_state(total_loss)
        self.train_acc_metric.update_state(labels, logits)
        return total_loss
    
    def train(self):
        """ 
        Main training loop for the current phase.
        Iterates through epochs, updates weights, and performs periodic evaluation/early stopping.
        """
        print(f"[Phase {self.phase_idx}] Starting training... ")
        
        total_train_steps = len(self.dataset.train_windows)
        if total_train_steps == 0:
            print("⚠️ ERROR: No training data found. Stopping training.")
            return

        for epoch in range(self.config.n_epochs):
            epoch_start_time = time.time()
            
            self.train_loss_metric.reset_state()
            self.train_acc_metric.reset_state()
            
            train_generator = self.dataset.get_batch('train')
            pbar = tqdm(
                train_generator, 
                total=total_train_steps, 
                desc=f"Epoch {epoch + 1}/{self.config.n_epochs}",
                unit="step"
            )
            
            for batch_windows, batch_labels, batch_neighbors in pbar:
                windows_tensor = tf.convert_to_tensor(batch_windows, dtype=tf.float32)
                labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
                neighbors_tensor = tf.convert_to_tensor(batch_neighbors, dtype=tf.int32)
                
                loss = self.train_step(windows_tensor, labels_tensor, neighbors_tensor)
                
                pbar.set_postfix(
                    loss=f"{self.train_loss_metric.result():.4f}", 
                    acc=f"{self.train_acc_metric.result():.4f}"
                )
            pbar.close()
            
            if (epoch + 1) % self.eval_step == 0:
                dev_metrics = self.evaluate(split='dev')
                print(f"    Dev Loss:   {dev_metrics['loss']:.4f}")
                print(f"    Dev Acc:    {dev_metrics['accuracy']:.4f}")
                print(f"    Dev F1:     {dev_metrics['f1_macro']:.4f}")
                
                if self.check_improvement(dev_metrics, epoch):
                    self.save_checkpoint()
                    print(f"  ✓ New best F1: {self.best_f1:.4f}")
                else:
                    if self.patience_counter >= self.config.early_stop_patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
            
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % self.print_step == 0:
                print(f"    Epoch time: {epoch_time:.2f}s")
        print(f"[Phase {self.phase_idx}] Best Dev F1: {self.best_f1:.4f} at epoch {self.best_epoch + 1}")
    
    def evaluate(self, split='dev'):
        """
        Evaluates the model on the specified split (dev/test).
        
        Parameters:
        - split: str, dataset split to evaluate on.
        
        Returns:
        - metrics: dict, containing loss, accuracy, and F1 scores.
        """
        all_predictions = []
        all_labels = []
        all_losses = []
        
        batch_count = 0
        
        for batch_windows, batch_labels, batch_neighbors in self.dataset.get_batch(split):
            if batch_count % 100 == 0:  
                print(f"    ... evaluating {split} timestep {batch_count}")
            batch_count += 1
            
            windows_tensor = tf.convert_to_tensor(batch_windows, dtype=tf.float32)
            labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
            neighbors_tensor = tf.convert_to_tensor(batch_neighbors, dtype=tf.int32)
            
            outputs = self.model({'windows': windows_tensor, 'neighbors': neighbors_tensor}, training=False)
            
            loss = self.loss_function(labels_tensor, outputs['logits'])
            all_predictions.append(outputs['predictions'].numpy())
            all_labels.append(batch_labels)
            all_losses.append(loss.numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        pred_classes = np.argmax(all_predictions, axis=1)
        true_classes = np.argmax(all_labels, axis=1)
        metrics = {'loss': float(np.mean(all_losses)), 
                   'accuracy': float(np.mean(pred_classes == true_classes)),
                   'f1_macro': float(f1_score(true_classes, pred_classes, average='macro')),
                   'f1_micro': float(f1_score(true_classes, pred_classes, average='micro')),
                   'predictions': all_predictions,
                   'true_labels': all_labels}
        return metrics
    
    def check_improvement(self, metrics, epoch):
        """
        Checks if the current model improves upon the best F1 score.
        
        Returns:
        - bool: True if improved, False otherwise.
        """
        current_f1 = metrics['f1_macro']
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_epoch = epoch
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
    
    def save_checkpoint(self):
        """Saves the current model state to disk."""
        save_path = self.checkpoint_manager.save()
    
    def load_best_checkpoint(self):
        """
        Restores the model weights from the best saved checkpoint.
        
        Returns:
        - path: path to the loaded checkpoint, or None if not found.
        """
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"[Phase {self.phase_idx}] Loaded checkpoint: {latest_checkpoint}")
            return latest_checkpoint
        else:
            print(f"[Phase {self.phase_idx}] WARNING: No checkpoint found")
            return None
    
    def get_market_state_for_rl(self, timestep, split='train'):
        """
        Helper method to retrieve market state for the RL agent.
        Contains legacy typo fixes in attribute names (datatset, datadet, lavels).
        """
        if split == 'train':
            windows = self.dataset.train_windows(timestep)
            labels = self.datatset.train_labels(timestep)
            neighbors = self.dataset.train_sampled_neighbors(timestep)
        elif split == 'dev':
            windows = self.dataset.dev_windows(timestep)
            labels = self.datadet.dev_labels(timestep)
            neighbors = self.dataset.dev_sampled_neighbors(timestep)
        else:
            windows = self.dataset.test_windows(timestep)
            labels = self.dataset.test_lavels(timestep)
            neighbors = self.dataset.test_sampled_neighbors(timestep)
        windows_tensor = tf.convert_to_tensor(windows, dtype=tf.float32)
        neighbors_tensor = tf.convert_to_tensor(neighbors, dtype=tf.int32)
        
        state_dict = self.model.get_market_state(windows_tensor, neighbors_tensor)
        
        state_dict['true_labels'] = labels
        state_dict['raw_windows'] = windows
        state_dict['timestep'] = timestep
        state_dict['split'] = split
        
        return state_dict
    
    def get_portfolio_features(self, state_dict):
        """
        Calculates portfolio metrics from the state dictionary.
        Duplicate of GNNModel.get_portfolio_features for convenience.
        """
        gnn_state = state_dict['gnn_state']
        predictions = state_dict['predictions']
        
        portfolio_features = {
            'bullish_ratio': float((predictions[:, -1] > 0.5).sum() / len(predictions)),
            'bearish_ratio': float((predictions[:, 0] > 0.5).sum() / len(predictions)),
            'neutral_ratio': float((predictions[:, 1] > 0.5).sum() / len(predictions)),
            'mean_embedding': np.mean(gnn_state, axis=0),  # [128]
            'std_embedding': np.std(gnn_state, axis=0),    # [128]
            'avg_confidence': float(np.mean(np.max(predictions, axis=1))),
            'max_confidence': float(np.max(np.max(predictions, axis=1))),
            'min_confidence': float(np.min(np.max(predictions, axis=1))),
            'num_companies': len(predictions)
        }
        return portfolio_features



class MultiPhaseTrainer:
    """
    Orchestrates the training process across multiple temporal phases (sliding windows).
    """
    def __init__(self, config, dataset):
        """
        Initializes the multi-phase trainer.
        
        Parameters:
        - config: configuration object.
        - dataset: instance of StockDataset.
        """
        self.config = config
        self.dataset = dataset
        self.num_phases = config.n_phases
        
        self.phase_results = []
        
        self.output_dir = os.path.join(config.log_dir, 'gnn_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train_all_phases(self):
        """
        Iteratively trains and evaluates the model on all defined phases.
        Aggregates results and generates final plots.
        """
        print("\n" + "="*80)
        print("MULTI-PHASE TRAINING STARTED")
        print(f"Total phases: {self.num_phases}")
        print("="*80)
        
        for phase_idx in range(self.num_phases):
            try:
                phase_result = self.train_single_phase(phase_idx)
                self.phase_results.append(phase_result)
                
                # Save intermediate results
                self.save_results()
                
            except Exception as e:
                print(f"\n❌ ERROR in Phase {phase_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute and display final statistics
        self.compute_final_statistics()
        self.plot_results()
        
        print("\n" + "="*80)
        print("✅ MULTI-PHASE TRAINING COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
    
    def train_single_phase(self, phase_idx):
        """ 
        Trains and tests the model on a single specific phase.
        
        Parameters:
        - phase_idx: int, index of phase to process.
        
        Returns:
        - phase_result: dict, metrics for this phase.
        """
        
        self.dataset.set_phase(phase_idx)
        
        model = GNNModel(self.config)
        model.build_layers()
        trainer = Trainer(model, self.dataset, self.config, phase_idx=phase_idx)
        
        train_start = time.time()
        trainer.train()
        training_time = time.time() - train_start
        
        print(f"\n[Phase {phase_idx}] Loading best checkpoint for testing...")
        trainer.load_best_checkpoint()
        
        print(f"[Phase {phase_idx}] Evaluating on test set...")
        test_metrics = trainer.evaluate(split='test')
        
        phase_result = {
            'phase': phase_idx,
            'training_time': training_time,
            'best_epoch': trainer.best_epoch,
            'best_dev_f1': trainer.best_f1,
            'test': {
                'loss': test_metrics['loss'],
                'accuracy': test_metrics['accuracy'],
                'f1_macro': test_metrics['f1_macro'],
                'f1_micro': test_metrics['f1_micro']
            }
        }
        
        # Print results
        print(f"\n[Phase {phase_idx}] RESULTS:")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Best Dev F1: {trainer.best_f1:.4f} at epoch {trainer.best_epoch + 1}")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 Macro: {test_metrics['f1_macro']:.4f}")
        print(f"  Test F1 Micro: {test_metrics['f1_micro']:.4f}")
        
        # Save phase-specific results
        phase_file = os.path.join(self.output_dir, f'phase_{phase_idx}_results.json')
        with open(phase_file, 'w') as f:
            json.dump(phase_result, f, indent=2)
        
        return phase_result
    
    def compute_final_statistics(self):
        """Compute average metrics across all phases"""
        if len(self.phase_results) == 0:
            print("\n❌ No results to compute statistics")
            return
        
        # Extract metrics
        test_losses = [r['test']['loss'] for r in self.phase_results]
        test_accs = [r['test']['accuracy'] for r in self.phase_results]
        test_f1_macros = [r['test']['f1_macro'] for r in self.phase_results]
        test_f1_micros = [r['test']['f1_micro'] for r in self.phase_results]
        
        # Compute statistics
        self.final_stats = {
            'num_phases': len(self.phase_results),
            'avg_loss': float(np.mean(test_losses)),
            'std_loss': float(np.std(test_losses)),
            'avg_accuracy': float(np.mean(test_accs)),
            'std_accuracy': float(np.std(test_accs)),
            'avg_f1_macro': float(np.mean(test_f1_macros)),
            'std_f1_macro': float(np.std(test_f1_macros)),
            'avg_f1_micro': float(np.mean(test_f1_micros)),
            'std_f1_micro': float(np.std(test_f1_micros))
        }
        
        # Print summary
        print("\n" + "="*80)
        print("FINAL RESULTS ACROSS ALL PHASES")
        print("="*80)
        print(f"\n{'Phase':<8} {'Accuracy':<12} {'F1 Macro':<12} {'F1 Micro':<12} {'Loss':<12}")
        print("-"*80)
        
        for result in self.phase_results:
            p = result['phase']
            acc = result['test']['accuracy']
            f1m = result['test']['f1_macro']
            f1mi = result['test']['f1_micro']
            loss = result['test']['loss']
            print(f"{p:<8} {acc:<12.4f} {f1m:<12.4f} {f1mi:<12.4f} {loss:<12.4f}")
        
        print("-"*80)
        print(f"{'MEAN':<8} {self.final_stats['avg_accuracy']:<12.4f} "
              f"{self.final_stats['avg_f1_macro']:<12.4f} "
              f"{self.final_stats['avg_f1_micro']:<12.4f} "
              f"{self.final_stats['avg_loss']:<12.4f}")
        print(f"{'STD':<8} {self.final_stats['std_accuracy']:<12.4f} "
              f"{self.final_stats['std_f1_macro']:<12.4f} "
              f"{self.final_stats['std_f1_micro']:<12.4f} "
              f"{self.final_stats['std_loss']:<12.4f}")
        print("="*80)
    
    def save_results(self):
        """Save all results to JSON"""
        overall_results = {
            'num_phases': len(self.phase_results),
            'phase_results': self.phase_results,
            'config': {
                'n_epochs': self.config.n_epochs,
                'lr': self.config.lr,
                'dropout': self.config.dropout,
                'node_feat_size': self.config.node_feat_size,
                'neighbors_sample': self.config.neighbors_sample
            }
        }
        
        if hasattr(self, 'final_stats'):
            overall_results['final_statistics'] = self.final_stats
        
        results_file = os.path.join(self.output_dir, 'overall_results.json')
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2)
    
    def plot_results(self):
        """Create visualization of results across phases"""
        if len(self.phase_results) == 0:
            return
        
        phases = [r['phase'] for r in self.phase_results]
        accs = [r['test']['accuracy'] for r in self.phase_results]
        f1s = [r['test']['f1_macro'] for r in self.phase_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(phases, accs, marker='o', linewidth=2, markersize=8, label='Accuracy')
        axes[0].axhline(self.final_stats['avg_accuracy'], color='red', linestyle='--',
                       label=f"Mean: {self.final_stats['avg_accuracy']:.4f}")
        axes[0].set_xlabel('Phase')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Test Accuracy Across Phases')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1].plot(phases, f1s, marker='s', linewidth=2, markersize=8, 
                    label='F1 Macro', color='green')
        axes[1].axhline(self.final_stats['avg_f1_macro'], color='red', linestyle='--',
                       label=f"Mean: {self.final_stats['avg_f1_macro']:.4f}")
        axes[1].set_xlabel('Phase')
        axes[1].set_ylabel('F1 Macro')
        axes[1].set_title('Test F1 Score Across Phases')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'phase_results.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {plot_file}")
        plt.close()
        
def train_hats_multiphase(config, dataset):
    """
    Main entry point for multi-phase training
    
    Usage:
        from config import get_args
        from dataset import StockDataset
        from model import train_hats_multiphase
        
        config = get_args()
        dataset = StockDataset(config)
        train_hats_multiphase(config, dataset)
    """
    multi_trainer = MultiPhaseTrainer(config, dataset)
    multi_trainer.train_all_phases()
    return multi_trainer