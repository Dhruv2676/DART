import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class StockDataset:
    def __init__(self, config):
        self.config = config
        
        # company info
        self.company_tickers = []
        self.num_companies = 0
        self.num_relations = 0
        
        # graph structure
        self.static_rel_mat = None
        self.neighbors = None
        self.rel_num = None
        
        # phase-wise train/dev/test set data
        self.phases = []
        self.current_phase = 0
        
        # feature extraction and scaling
        self.scalers = {}
        self.label_thresholds = []
        
        # initializing the dataset
        self.build_static_adjacency_matrices()
        self.load_all_price_data()
        self.create_phases()
        self.set_phase(config.test_phase if hasattr(config, 'test_phase') else 0)
        print("\n")
    
    def build_static_adjacency_matrices(self):
        """ 
        Builds the static relations and their adjacency matrices from raw data or cache.
        
        This method processes raw relation files to create adjacency matrices 
        representing relationships between companies (e.g., sector, supply chain).
        """
        print("Building Static Relations... ")
        
        with open(self.config.tickers_path, 'r') as f:
            self.company_tickers = f.read().split()
        self.num_companies = len(self.company_tickers)
        
        adj_mat_file = os.path.join(self.config.processed_dir, 'adj_mat.pkl')
        rel_num_file = os.path.join(self.config.processed_dir, 'rel_num.pkl')
        
        # Check if processed matrices already exist to save time
        if os.path.exists(adj_mat_file) and os.path.exists(rel_num_file):
            with open(adj_mat_file, 'rb') as f:
                self.static_rel_mat = pickle.load(f)
            with open(rel_num_file, 'rb') as f:
                self.rel_num = pickle.load(f)
        else:
            df_raw_relations = pd.read_csv(self.config.raw_relations_path)
            
            with open(self.config.properties_path, 'r') as f:
                properties = f.read().split()
            property_to_idx = {ticker: idx for idx, ticker in enumerate(properties)}
            relation_types = df_raw_relations['Relation_Type'].unique()
            num_rel_types = len(relation_types)
            
            # Initialize 3D adjacency matrix: [relation_type, company_i, company_j]
            self.static_rel_mat = np.zeros((num_rel_types, self.num_companies, self.num_companies))
            
            for rel_idx, rel_type in enumerate(relation_types):
                rel_df = df_raw_relations[df_raw_relations['Relation_Type'] == rel_type]
                for _, row in rel_df.iterrows():
                    c1, c2 = row['Source'], row['Target']
                    if c1 not in property_to_idx or c2 not in property_to_idx:
                        continue
                    idx1, idx2 = property_to_idx[c1], property_to_idx[c2]
                    
                    # Set bidirectional relationship
                    self.static_rel_mat[rel_idx, idx1, idx2] = 1
                    self.static_rel_mat[rel_idx, idx2, idx1] = 1
            
            self.rel_num = self.static_rel_mat.sum(axis=2)
            
            # Cache the computed matrices
            with open(adj_mat_file, 'wb') as f:
                pickle.dump(self.static_rel_mat, f)
            with open(rel_num_file, 'wb') as f:
                pickle.dump(self.rel_num, f)
        
        self.neighbors = []
        for rel_idx in range(len(self.static_rel_mat)):
            self.neighbors.append(self.build_neighbor_list(self.static_rel_mat[rel_idx]))
        
        k = self.config.neighbors_sample
        self.rel_num = np.minimum(self.rel_num, k)
        
        # Total relations include static types + 1 dynamic type (added later)
        self.num_relations = len(self.rel_num) + 1
    
    def build_neighbor_list(self, rel_mat):
        """ 
        Builds the neighbors list for the static adjacency matrices.
        
        Parameters:
        - rel_mat: np.array, 2D adjacency matrix for a specific relation.
        
        Returns:
        - rel_neighbors: list, indices of neighbors for each node.
        """
        rel_neighbors = []
        for node_idx in range(self.num_companies):
            neighbor_indices = np.where(rel_mat[node_idx] > 0)[0]
            rel_neighbors.append(neighbor_indices)
        
        return rel_neighbors
    
    def load_all_price_data(self):
        """ 
        Loads the entire price and technical indicator dataset for all companies.
        Populates self.all_price_data and self.all_returns.
        """
        print("Loading complete price history for all companies...")
        
        self.all_price_data = []
        self.all_returns = []
        self.all_dates = []
        dates_loaded = False
        
        for ticker in self.company_tickers:
            file_path = os.path.join(self.config.price_data_dir, f'{ticker}.csv')
            df = pd.read_csv(file_path)
            
            if len(self.all_dates) == 0:
                self.all_dates = df['Date'].values
            
            feature_data = df[self.config.price_feature_list].values
            returns_data = df['Returns'].values
        
            self.all_price_data.append(feature_data)
            self.all_returns.append(returns_data)
        
        # Shape: [num_companies, total_days, num_features]
        self.all_price_data = np.array(self.all_price_data)
        self.all_returns = np.array(self.all_returns)
    
    def create_phases(self):
        """ 
        Splits the dataset into N_PHASES sequential phases using a sliding window approach.
        Each phase contains its own Train/Dev/Test splits.
        """
        N_PHASES = self.config.n_phases
        LOOKBACK = self.config.lookback
        # Hardcoded split lengths for the sliding window
        TRAIN_DAYS = 350
        DEV_DAYS = 70
        TEST_DAYS = 140
        SLIDE_DAYS = 140

        FIRST_TEST_START_DAY = LOOKBACK + TRAIN_DAYS + DEV_DAYS 
        
        total_days = self.all_price_data.shape[1]
        print(f"Creating {N_PHASES} market phases (Sliding Window)...")

        for phase_idx in range(N_PHASES):
            # Calculate indices for the sliding window
            test_start = FIRST_TEST_START_DAY + (phase_idx * SLIDE_DAYS)
            test_end = test_start + TEST_DAYS
            dev_start = test_start - DEV_DAYS
            dev_end = test_start
            train_start = dev_start - TRAIN_DAYS
            train_end = dev_start
            
            data_start = train_start - LOOKBACK
            data_end = test_end
            
            # Validation checks for data boundaries
            if data_start < 0:
                print(f"  ⚠️  Phase {phase_idx} skipped: requires data before day 0.")
                continue
                
            if data_end > total_days:
                print(f"  ⚠️  Phase {phase_idx} (days {data_start} to {data_end}) skipped: exceeds available data ({total_days} days).")
                break

            phase_price_data = self.all_price_data[:, data_start:data_end, :]
            phase_returns = self.all_returns[:, data_start:data_end]
            
            phase_dates = self.all_dates[data_start:data_end]
            
            # Adjust indices relative to the phase start
            phase_train_start = train_start - data_start
            phase_train_end = train_end - data_start
            phase_dev_start = dev_start - data_start
            phase_dev_end = dev_end - data_start
            phase_test_start = test_start - data_start
            phase_test_end = test_end - data_start
            
            phase_data = self.create_phase_data(phase_price_data, phase_returns, phase_dates, phase_train_start, phase_train_end, phase_dev_start, phase_dev_end, phase_test_start, phase_test_end, phase_idx)
            self.phases.append(phase_data)
        
    def create_phase_data(self, price_data, returns, dates, train_start, train_end, dev_start, dev_end, test_start, test_end, phase_idx):
        """ 
        Creates windows, labels, and neighbor structures for a specific phase.

        Parameters:
        - price_data: np.array, price data for this phase.
        - returns: np.array, return data for this phase.
        - dates: np.array, date strings.
        - train_start, train_end, etc.: int, indices for splits.
        - phase_idx: int, identifier for the phase.

        Returns:
        - dict, containing all data (windows, labels, neighbors) for the phase.
        """
        train_price = price_data[:, train_start:train_end, :]
        train_returns_flat = returns[:, train_start:train_end].flatten()
        
        # Fit scalers and thresholds only on training data
        phase_scalers = self.fit_phase_scalers(train_price)
        phase_thresholds = self.fit_phase_thresholds(train_returns_flat)
        
        # Generate windows and labels for each split
        train_windows, train_labels, train_indices = self.create_windows_for_split(price_data[:, :train_end, :], returns[:, :train_end], phase_scalers, phase_thresholds, start_idx=train_start)
        dev_windows, dev_labels, dev_indices = self.create_windows_for_split(price_data[:, :dev_end, :], returns[:, :dev_end], phase_scalers, phase_thresholds, start_idx=dev_start)
        test_windows, test_labels, test_indices = self.create_windows_for_split(price_data[:, :test_end, :], returns[:, :test_end], phase_scalers, phase_thresholds, start_idx=test_start)
        
        # Build dynamic relations based on correlation
        train_dynamic_adj = self.build_dynamic_relations_for_windows(train_windows, returns, train_start)
        dev_dynamic_adj = self.build_dynamic_relations_for_windows(dev_windows, returns, dev_start)
        test_dynamic_adj = self.build_dynamic_relations_for_windows(test_windows, returns, test_start)
        
        # Sample neighbors for graph inputs
        train_neighbors = self.sample_neighbors_for_windows(train_dynamic_adj)
        dev_neighbors = self.sample_neighbors_for_windows(dev_dynamic_adj)
        test_neighbors = self.sample_neighbors_for_windows(test_dynamic_adj)
        
        train_dates = dates[train_indices] if len(train_indices) > 0 else []
        test_dates = dates[test_indices] if len(test_indices) > 0 else []
        
        return {
            'phase_idx': phase_idx,
            'train_windows': train_windows,
            'train_labels': train_labels,
            'train_neighbors': train_neighbors,
            'dev_windows': dev_windows,
            'dev_labels': dev_labels,
            'dev_neighbors': dev_neighbors,
            'test_windows': test_windows,
            'test_labels': test_labels,
            'test_neighbors': test_neighbors,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'scalers': phase_scalers,
            'thresholds': phase_thresholds,
            'global_indices': {
                'train': train_start,
                'dev': dev_start,
                'test': test_start
            }
        }
        
    def fit_phase_scalers(self, train_price_data):
        """ 
        Fits a MinMaxScaler on the given phase's training set.
        
        Parameters:
        - train_price_data: np.array, training data to fit scaler.
        
        Returns:
        - phase_scalers: dict, mapping feature names to fitted scalers.
        """
        phase_scalers = {}
        for feat_idx, feat_name in enumerate(self.config.price_feature_list):
            all_values = train_price_data[:, :, feat_idx].flatten().reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(all_values)
            phase_scalers[feat_name] = scaler
        return phase_scalers
    
    def fit_phase_thresholds(self, train_returns_flat):
        """ 
        Determines label thresholds based on label proportions in the training data.
        
        Parameters:
        - train_returns_flat: np.array, flattened return values from training set.
        
        Returns:
        - phase_thresholds: list, calculated threshold values.
        """
        sorted_returns = np.sort(train_returns_flat)
        n_labels = len(self.config.label_proportion)
        th_total = sum(self.config.label_proportion)
        phase_thresholds = []
        cumulative = 0
        for proportion in self.config.label_proportion[:-1]:
            cumulative += proportion
            threshold_idx = int(len(sorted_returns) * cumulative / th_total) - 1
            phase_thresholds.append(sorted_returns[threshold_idx])
        return phase_thresholds
    
    def create_windows_for_split(self, price_data, returns, scalers, thresholds, start_idx):
        """ 
        Generates windowed feature sequences and labels for a specific split.

        Parameters:
        - price_data: np.array, raw price data.
        - returns: np.array, raw return data.
        - scalers: dict, fitted scalers.
        - thresholds: list, fitted thresholds.
        - start_idx: int, starting index for this split.

        Returns:
        - per_timestep_windows: list, windowed data sorted by timestep.
        - per_timestep_labels: list, labels sorted by timestep.
        - valid_indices: list, valid time indices used.
        """
        split_length = price_data.shape[1]
        
        # Scale the data first
        scaled_data = np.zeros_like(price_data)
        for feat_idx, feat_name in enumerate(self.config.price_feature_list):
            for company_idx in range(self.num_companies):
                feat_values = price_data[company_idx, :, feat_idx].reshape(-1, 1)
                scaled_data[company_idx, :, feat_idx] = scalers[feat_name].transform(feat_values).flatten()
        
        per_company_windows = []
        per_company_labels = []
        valid_indices = []
        indices_collected = False
        for company_idx in range(self.num_companies):
            company_windows = []
            company_labels = []
            for t in range(start_idx, split_length):
                if t < self.config.lookback:
                    continue
                
                if not indices_collected:
                    valid_indices.append(t)
                
                # Extract lookback window
                window = scaled_data[company_idx, t-self.config.lookback:t, :]
                future_return = returns[company_idx, t]
                label = self.classify_returns(future_return, thresholds)
                
                company_windows.append(window)
                company_labels.append(label)
                
            indices_collected = True
            
            per_company_windows.append(np.array(company_windows))
            per_company_labels.append(np.array(company_labels))
        
        if len(per_company_windows[0]) == 0:
            return [], [], []
        
        num_timesteps = per_company_windows[0].shape[0]
        per_timestep_windows = []
        per_timestep_labels = []
        
        # Re-organize data to be per-timestep instead of per-company
        for t in range(num_timesteps):
            timestep_windows = np.array([per_company_windows[c][t] for c in range(self.num_companies)])
            timestep_labels = np.array([per_company_labels[c][t] for c in range(self.num_companies)])
            per_timestep_windows.append(timestep_windows)
            per_timestep_labels.append(timestep_labels)      
        
        return per_timestep_windows, per_timestep_labels, valid_indices
    
    def classify_returns(self, return_value, thresholds):
        """
        Classifies a return value into a one-hot encoded label based on thresholds.

        Parameters:
        - return_value: float, the return value.
        - thresholds: list, boundaries for classes.

        Returns:
        - label: np.array, one-hot encoded vector.
        """
        n_labels = len(self.config.label_proportion)
        label = np.zeros(n_labels)
        if n_labels == 2:
            label[0 if return_value < thresholds[0] else 1] = 1
        elif n_labels == 3:
            if return_value < thresholds[0]:
                label[0] = 1 # down
            elif return_value < thresholds[1]:
                label[1] = 1 # neutral
            else:
                label[2] = 1 # up
        return label
    
    def build_dynamic_relations_for_windows(self, windows, returns, start_idx):
        """ 
        Constructs dynamic adjacency matrices based on correlation of returns within the lookback window.
        
        Returns:
        - dynamic_adj_list: list, sequence of adjacency matrices.
        """
        dynamic_adj_list = []
        if len(windows) == 0:
            return dynamic_adj_list
            
        num_timesteps = len(windows)
        for t in range(num_timesteps):
            actual_t = start_idx + t
            if actual_t < self.config.lookback:
                adj_matrix = np.zeros((self.num_companies, self.num_companies))
            else:
                window_returns = returns[:, actual_t-self.config.lookback:actual_t]
                corr_matrix = np.corrcoef(window_returns)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                # Thresholding correlation to create binary adjacency
                adj_matrix = (np.abs(corr_matrix) > self.config.corr_threshold).astype(float)
                np.fill_diagonal(adj_matrix, 0)
            dynamic_adj_list.append(adj_matrix)
                    
        return dynamic_adj_list
    
    def sample_neighbors_for_windows(self, dynamic_adj_list):
        """ 
        Samples k neighbors for each timestep, combining static and dynamic relations.
        
        Parameters:
        - dynamic_adj_list: list, dynamic adjacency matrices per timestep.
        
        Returns:
        - np.array, sampled neighbors structure.
        """
        if len(dynamic_adj_list) == 0:
            return np.array([])
        
        sampled_list = []
        for dynamic_adj in dynamic_adj_list:
            dynamic_adj_expanded = np.expand_dims(dynamic_adj, 0)
            # Combine static relations with the current dynamic relation
            combined_adj = np.concatenate([self.static_rel_mat, dynamic_adj_expanded], axis=0)
            sampled_neighbors = self.sample_neighbors_from_matrix(combined_adj)
            sampled_list.append(sampled_neighbors)
        
        return np.array(sampled_list)
    
    def sample_neighbors_from_matrix(self, adj_matrix):
        """ 
        Performs the actual sampling of neighbors from a given adjacency matrix.
        
        Parameters:
        - adj_matrix: np.array, combined adjacency matrix.
        
        Returns:
        - sampled: np.array, indices of sampled neighbors.
        """
        k = self.config.neighbors_sample
        num_relations, num_companies, _ = adj_matrix.shape
        sampled = np.zeros((num_relations, num_companies, k), dtype=np.int32)
        for rel_idx in range(num_relations):
            for node_idx in range(num_companies):
                neighbor_indices = np.where(adj_matrix[rel_idx, node_idx] > 0)[0]
                # Shift indices by 1
                neighbor_indices = neighbor_indices + 1
                if len(neighbor_indices) == 0:
                    pass
                elif len(neighbor_indices) <= k:
                    sampled[rel_idx, node_idx, :len(neighbor_indices)] = neighbor_indices
                else:
                    sampled[rel_idx, node_idx] = np.random.choice(neighbor_indices, k, replace=False)
                
        return sampled
    
    def set_phase(self, phase_idx):
        """ 
        Sets the current active phase and loads its data into active variables.
        
        Parameters:
        - phase_idx: int, index of the phase to activate.
        """
        if phase_idx >= len(self.phases):
            raise ValueError(f"Phase {phase_idx} doesn't exist. Only {len(self.phases)} phases available.")
        self.current_phase = phase_idx
        
        phase = self.phases[phase_idx]
        self.train_windows = phase['train_windows']
        self.train_labels = phase['train_labels']
        self.train_sampled_neighbors = phase['train_neighbors']
        self.dev_windows = phase['dev_windows']
        self.dev_labels = phase['dev_labels']
        self.dev_sampled_neighbors = phase['dev_neighbors']
        self.test_windows = phase['test_windows']
        self.test_labels = phase['test_labels']
        self.test_sampled_neighbors = phase['test_neighbors']
        
        self.scalers = phase['scalers']
        self.label_thresholds = phase['thresholds']
    
    def get_batch(self, split='train', batch_size=None, shuffle=False):
        """ 
        Generator that yields batches of data for training or evaluation.
        
        Parameters:
        - split: str, 'train', 'dev', or 'test'.
        - batch_size: int, (Unused in current logic but kept for interface).
        - shuffle: bool, whether to shuffle the order of windows.
        
        Yields:
        - batch_windows, batch_labels, batch_neighbors
        """
        if split == 'train':
            windows = self.train_windows
            labels = self.train_labels
            sampled_neighbors = self.train_sampled_neighbors
        elif split == 'dev':
            windows = self.dev_windows
            labels = self.dev_labels
            sampled_neighbors = self.dev_sampled_neighbors
        elif split == 'test':
            windows = self.test_windows
            labels = self.test_labels
            sampled_neighbors = self.test_sampled_neighbors
        
        if len(windows) == 0:
            return
        
        indices = np.arange(len(windows))
        if shuffle:
            np.random.shuffle(indices)
        
        for t in indices:
            batch_windows = windows[t]
            batch_labels = labels[t]
            batch_neighbors = sampled_neighbors[t]
        
            yield batch_windows, batch_labels, batch_neighbors