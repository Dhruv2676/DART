import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class BaselineStockDataset:
    def __init__(self):
        """
        Initializes the BaselineStockDataset.
        Sets up paths, configuration parameters, and starts data loading.
        """
        # directories and paths
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.cur_dir, '..', 'data')
        
        # configuration parameters
        self.neighbors_sample = 15
        self.price_feature_list = ['Returns']
        self.label_proportion = [3, 4, 3]
        self.lookback = 50
        
        self.company_tickers = []
        self.num_companies = 0
        self.num_relations = 0
        
        # graph structure
        self.static_rel_mat = None
        self.static_sampled_neighbors = None
        self.rel_num = None
        
        # phase-wise train/dev/test set data
        self.phases = []
        self.current_phase = 0
        
        # feature extraction and scaling
        self.scalers = {}
        self.label_thresholds = []
        
        self.train_log_returns = np.array([])
        self.dev_log_returns = np.array([])
        self.test_log_returns = np.array([])
        
        self.all_dates = None
        self.snp500_returns = None
        self.load_dates_and_snp500()
        
        # initializing the dataset
        self.build_static_adjacency_matrices()
        self.load_required_price_data()
        self.create_phases()
        print("\n")
        
    def load_dates_and_snp500(self):
        """
        Loads the list of dates from the first available ticker file and 
        loads S&P 500 returns for benchmark comparison.
        """
        print("Loading dates and S&P 500 benchmark returns...")
        
        first_ticker_file = os.path.join(self.data_dir, 'price', os.listdir(os.path.join(self.data_dir, 'price'))[0])
        df_dates = pd.read_csv(first_ticker_file)
        if 'Date' in df_dates.columns:
            self.all_dates = df_dates['Date'].values
        else:
            print("  Warning: No 'Date' column found in price files. Dates will not be available.")
            self.all_dates = None
        
        snp500_file = os.path.join(self.data_dir, 'snp500_returns.csv')
        if os.path.exists(snp500_file):
            df_snp = pd.read_csv(snp500_file)
            if 'Returns' in df_snp.columns:
                self.snp500_returns = df_snp['Returns'].values
                print(f"  Loaded {len(self.snp500_returns)} S&P 500 return values")
            else:
                print("  Warning: No 'Returns' column in snp500_returns.csv")
                self.snp500_returns = None
        else:
            print(f"  Warning: S&P 500 returns file not found at {snp500_file}")
            self.snp500_returns = None
    
    def build_static_adjacency_matrices(self):
        """ 
        Builds or loads the static relation adjacency matrices.
        
        Reads raw relations from CSV, maps them to company indices, 
        and creates a 3D adjacency matrix [relation_type, source, target].
        Results are cached as pickles for faster loading next time.
        """
        print("Building Static Relations... ")
        
        with open(os.path.join(self.data_dir, 'tickers.txt'), 'r') as f:
            self.company_tickers = f.read().split()
        self.num_companies = len(self.company_tickers)
        
        adj_mat_file = os.path.join(self.cur_dir, 'adj_mat.pkl')
        rel_num_file = os.path.join(self.cur_dir, 'rel_num.pkl')
        
        # Load from cache if available
        if os.path.exists(adj_mat_file) and os.path.exists(rel_num_file):
            with open(adj_mat_file, 'rb') as f:
                self.static_rel_mat = pickle.load(f)
            with open(rel_num_file, 'rb') as f:
                self.rel_num = pickle.load(f)
        else:
            # Construct from raw data
            df_raw_relations = pd.read_csv(os.path.join(self.data_dir, 'relation/raw_relations.csv'))
            
            with open(os.path.join(self.data_dir, 'properties.txt'), 'r') as f:
                properties = f.read().split()
            property_to_idx = {ticker: idx for idx, ticker in enumerate(properties)}
            relation_types = df_raw_relations['Relation_Type'].unique()
            num_rel_types = len(relation_types)
            
            self.static_rel_mat = np.zeros((num_rel_types, self.num_companies, self.num_companies))
            
            for rel_idx, rel_type in enumerate(relation_types):
                rel_df = df_raw_relations[df_raw_relations['Relation_Type'] == rel_type]
                for _, row in rel_df.iterrows():
                    c1, c2 = row['Source'], row['Target']
                    if c1 not in property_to_idx or c2 not in property_to_idx:
                        continue
                    idx1, idx2 = property_to_idx[c1], property_to_idx[c2]
                    
                    # Undirected/Bidirectional graph assumption
                    self.static_rel_mat[rel_idx, idx1, idx2] = 1
                    self.static_rel_mat[rel_idx, idx2, idx1] = 1
            
            self.rel_num = self.static_rel_mat.sum(axis=2)
            
            # Save to cache
            with open(adj_mat_file, 'wb') as f:
                pickle.dump(self.static_rel_mat, f)
            with open(rel_num_file, 'wb') as f:
                pickle.dump(self.rel_num, f)
        
        self.static_sampled_neighbors = self.build_static_sampled_neighbors(self.static_rel_mat)
        
        k = self.neighbors_sample
        self.rel_num = np.minimum(self.rel_num, k)
        
        self.num_relations = len(self.rel_num)
    
    def build_static_sampled_neighbors(self, adj_matrix):
        """
        Samples a fixed number (k) of neighbors for each node and relation type.
        
        Parameters:
        - adj_matrix: 3D numpy array [num_relations, num_companies, num_companies]
        
        Returns:
        - sampled: 3D numpy array [num_relations, num_companies, k] containing neighbor indices.
        """
        k = self.neighbors_sample
        num_relations, num_companies, _ = adj_matrix.shape
        sampled = np.zeros((num_relations, num_companies, k), dtype=np.int32)
        
        for rel_idx in range(num_relations):
            for node_idx in range(num_companies):
                neighbor_indices = np.where(adj_matrix[rel_idx, node_idx] > 0)[0]
                neighbor_indices = neighbor_indices + 1 
                
                if len(neighbor_indices) == 0:
                    pass
                elif len(neighbor_indices) <= k:
                    sampled[rel_idx, node_idx, :len(neighbor_indices)] = neighbor_indices
                else:
                    sampled[rel_idx, node_idx] = np.random.choice(neighbor_indices, k, replace=False)
        return sampled
    
    def load_required_price_data(self):
        """ 
        Loads price history (features and returns) for all companies from CSV files.
        Populates self.all_price_data and self.all_returns.
        """
        print("Loading complete price history for all companies...")
        
        self.all_price_data = []
        self.all_returns = []
        
        for ticker in self.company_tickers:
            file_path = os.path.join(self.data_dir, f'price/{ticker}.csv')
            df = pd.read_csv(file_path)
            
            feature_data = df[self.price_feature_list].values
            returns_data = df['Returns'].values
        
            self.all_price_data.append(feature_data)
            self.all_returns.append(returns_data)
        
        self.all_price_data = np.array(self.all_price_data)
        self.all_returns = np.array(self.all_returns)
    
    def create_phases(self):
        """ 
        Splits the dataset into N_PHASES sequential phases using a sliding window approach.
        Each phase contains defined Train, Dev, and Test periods.
        """
        
        N_PHASES = 12
        LOOKBACK = self.lookback
        TRAIN_DAYS = 350
        DEV_DAYS = 70
        TEST_DAYS = 140
        SLIDE_DAYS = 140

        FIRST_TEST_START_DAY = LOOKBACK + TRAIN_DAYS + DEV_DAYS 
        
        total_days = self.all_price_data.shape[1]
        print(f"Creating {N_PHASES} market phases (Sliding Window)...")

        for phase_idx in range(N_PHASES):
            # Calculate indices
            test_start = FIRST_TEST_START_DAY + (phase_idx * SLIDE_DAYS)
            test_end = test_start + TEST_DAYS
            dev_start = test_start - DEV_DAYS
            dev_end = test_start
            train_start = dev_start - TRAIN_DAYS
            train_end = dev_start
            
            data_start = train_start - LOOKBACK
            data_end = test_end
            
            # Validation checks
            if data_start < 0:
                print(f"  ⚠️  Phase {phase_idx} skipped: requires data before day 0.")
                continue
                
            if data_end > total_days:
                print(f"  ⚠️  Phase {phase_idx} (days {data_start} to {data_end}) skipped: exceeds available data ({total_days} days).")
                break

            phase_price_data = self.all_price_data[:, data_start:data_end, :]
            phase_returns = self.all_returns[:, data_start:data_end]
            
            phase_train_start = train_start - data_start
            phase_train_end = train_end - data_start
            phase_dev_start = dev_start - data_start
            phase_dev_end = dev_end - data_start
            phase_test_start = test_start - data_start
            phase_test_end = test_end - data_start
            
            phase_data = self.create_phase_data(phase_price_data, phase_returns, phase_train_start, phase_train_end, phase_dev_start, phase_dev_end, phase_test_start, phase_test_end, phase_idx, data_start, data_end)
            self.phases.append(phase_data)
    
    def create_phase_data(self, price_data, returns, train_start, train_end, dev_start, dev_end, test_start, test_end, phase_idx, data_start, data_end):
        """ 
        Generates windowed data, labels, and static neighbor structures for a specific phase.
        
        Parameters:
        - price_data: np.array, subset of price data for this phase.
        - returns: np.array, subset of returns.
        - train_start, train_end, dev_start, dev_end, test_start, test_end: indices for splits.
        - phase_idx: int, phase index.
        - data_start, data_end: global indices for date mapping.
        
        Returns:
        - dict: Complete data package for the phase.
        """
        
        train_price = price_data[:, train_start:train_end, :]
        train_returns_flat = returns[:, train_start:train_end].flatten()
        
        # Fit scalers and thresholds on training data only
        phase_scalers = self.fit_phase_scalers(train_price)
        phase_thresholds = self.fit_phase_thresholds(train_returns_flat)
        
        # Generate windows
        train_windows, train_labels = self.create_windows_for_split(price_data[:, :train_end, :], returns[:, :train_end], phase_scalers, phase_thresholds, start_idx=train_start)
        dev_windows, dev_labels = self.create_windows_for_split(price_data[:, :dev_end, :], returns[:, :dev_end], phase_scalers, phase_thresholds, start_idx=dev_start)
        test_windows, test_labels = self.create_windows_for_split(price_data[:, :test_end, :], returns[:, :test_end], phase_scalers, phase_thresholds, start_idx=test_start)
        
        # Use static neighbors for all timesteps (Baseline implementation)
        train_neighbors = np.array([self.static_sampled_neighbors] * len(train_windows))
        dev_neighbors = np.array([self.static_sampled_neighbors] * len(dev_windows))
        test_neighbors = np.array([self.static_sampled_neighbors] * len(test_windows))
        
        # Raw log returns for evaluation
        train_log_returns = returns[:, train_start : train_end - 1]
        dev_log_returns = returns[:, dev_start : dev_end - 1]
        test_log_returns = returns[:, test_start : test_end - 1]
        
        # Map dates if available
        phase_dates = None
        phase_snp500_returns = None
        if self.all_dates is not None:
            phase_dates = {
                'train': self.all_dates[data_start + train_start : data_start + train_end - 1] if data_start + train_end - 1 <= len(self.all_dates) else None,
                'dev': self.all_dates[data_start + dev_start : data_start + dev_end - 1] if data_start + dev_end - 1 <= len(self.all_dates) else None,
                'test': self.all_dates[data_start + test_start : data_start + test_end - 1] if data_start + test_end - 1 <= len(self.all_dates) else None
            }
        
        if self.snp500_returns is not None:
            phase_snp500_returns = {
                'train': self.snp500_returns[data_start + train_start : data_start + train_end - 1] if data_start + train_end - 1 <= len(self.snp500_returns) else None,
                'dev': self.snp500_returns[data_start + dev_start : data_start + dev_end - 1] if data_start + dev_end - 1 <= len(self.snp500_returns) else None,
                'test': self.snp500_returns[data_start + test_start : data_start + test_end - 1] if data_start + test_end - 1 <= len(self.snp500_returns) else None
            }
        
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
            'scalers': phase_scalers,
            'thresholds': phase_thresholds,
            'train_log_returns': train_log_returns,
            'dev_log_returns': dev_log_returns,
            'test_log_returns': test_log_returns,
            'dates': phase_dates,
            'snp500_returns': phase_snp500_returns,
        }
        
    def fit_phase_scalers(self, train_price_data):
        """ 
        Fits a MinMaxScaler for each feature based on the training data.
        
        Parameters:
        - train_price_data: np.array, training feature data.
        
        Returns:
        - dict: Mapping of feature names to fitted scaler objects.
        """
        phase_scalers = {}
        for feat_idx, feat_name in enumerate(self.price_feature_list):
            all_values = train_price_data[:, :, feat_idx].flatten().reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(all_values)
            phase_scalers[feat_name] = scaler
        return phase_scalers
    
    def fit_phase_thresholds(self, train_returns_flat):
        """ 
        Determines classification thresholds based on training return quantiles.
        
        Parameters:
        - train_returns_flat: flattened array of training returns.
        
        Returns:
        - list: Threshold values separating classes.
        """
        sorted_returns = np.sort(train_returns_flat)
        n_labels = len(self.label_proportion)
        th_total = sum(self.label_proportion)
        phase_thresholds = []
        cumulative = 0
        for proportion in self.label_proportion[:-1]:
            cumulative += proportion
            threshold_idx = int(len(sorted_returns) * cumulative / th_total) - 1
            phase_thresholds.append(sorted_returns[threshold_idx])
        return phase_thresholds
    
    def create_windows_for_split(self, price_data, returns, scalers, thresholds, start_idx):
        """ 
        Generates temporal windows of features and corresponding labels for a split.
        
        Parameters:
        - price_data: raw price data.
        - returns: raw returns data.
        - scalers: fitted scalers.
        - thresholds: fitted label thresholds.
        - start_idx: starting index for generation.
        
        Returns:
        - per_timestep_windows, per_timestep_labels: Data organized by timestep.
        """
    
        split_length = price_data.shape[1]
        
        scaled_data = np.zeros_like(price_data)
        for feat_idx, feat_name in enumerate(self.price_feature_list):
            for company_idx in range(self.num_companies):
                feat_values = price_data[company_idx, :, feat_idx].reshape(-1, 1)
                scaled_data[company_idx, :, feat_idx] = scalers[feat_name].transform(feat_values).flatten()
        
        per_company_windows = []
        per_company_labels = []
        for company_idx in range(self.num_companies):
            company_windows = []
            company_labels = []
            for t in range(start_idx, split_length - 1):
                if t < self.lookback:
                    continue
                
                window = scaled_data[company_idx, t-self.lookback:t, :]
                future_return = returns[company_idx, t]
                label = self.classify_returns(future_return, thresholds)
                
                company_windows.append(window)
                company_labels.append(label)
            
            per_company_windows.append(np.array(company_windows))
            per_company_labels.append(np.array(company_labels))
        
        if len(per_company_windows) == 0 or len(per_company_windows[0]) == 0:
            return [], []
        
        num_timesteps = per_company_windows[0].shape[0]
        per_timestep_windows = []
        per_timestep_labels = []
        
        for t in range(num_timesteps):
            timestep_windows = np.array([per_company_windows[c][t] for c in range(self.num_companies)])
            timestep_labels = np.array([per_company_labels[c][t] for c in range(self.num_companies)])
            per_timestep_windows.append(timestep_windows)
            per_timestep_labels.append(timestep_labels)      
        
        return per_timestep_windows, per_timestep_labels  
    
    def classify_returns(self, return_value, thresholds):
        """
        Classifies a return value into one-hot encoded label buckets.
        """
        n_labels = len(self.label_proportion)
        label = np.zeros(n_labels, dtype=np.float32)
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
    
    def sample_neighbors_for_windows(self, dynamic_adj_list):
        """ 
        Samples neighbors for each timestep based on dynamic adjacency matrices.
        (Note: Used for dynamic graphs, whereas this class primarily focuses on static).
        """
        if len(dynamic_adj_list) == 0:
            return np.array([])
        
        sampled_list = []
        for dynamic_adj in dynamic_adj_list:
            dynamic_adj_expanded = np.expand_dims(dynamic_adj, 0)
            combined_adj = np.concatenate([self.static_rel_mat, dynamic_adj_expanded], axis=0)
            sampled_neighbors = self.sample_neighbors_from_matrix(combined_adj)
            sampled_list.append(sampled_neighbors)
        
        return np.array(sampled_list)
    
    def sample_neighbors_from_matrix(self, adj_matrix):
        """
        Samples k neighbors from a combined adjacency matrix.
        
        Parameters:
        - adj_matrix: Combined static+dynamic adjacency.
        
        Returns:
        - sampled: indices of sampled neighbors.
        """
        k = self.neighbors_sample
        num_relations, num_companies, _ = adj_matrix.shape
        sampled = np.zeros((num_relations, num_companies, k), dtype=np.int32)
        for rel_idx in range(num_relations):
            for node_idx in range(num_companies):
                neighbor_indices = np.where(adj_matrix[rel_idx, node_idx] > 0)[0]
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
        Sets the active phase for the dataset, updating pointers to train/dev/test sets.
        
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
        
        self.train_log_returns = phase.get('train_log_returns', np.array([]))
        self.dev_log_returns = phase.get('dev_log_returns', np.array([]))
        self.test_log_returns = phase.get('test_log_returns', np.array([]))
        
        self.current_dates = phase.get('dates', None)
        self.current_snp500_returns = phase.get('snp500_returns', None)
    
    def get_batch(self, split='train', batch_size=None, shuffle=False):
        """
        Generator yielding batches of data for training/evaluation.
        
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
    
    def get_log_returns_for_split(self, split='test'):
        """ Returns flattened log returns for a specific split. """
        if split == 'train':
            returns_slice = self.train_log_returns
        elif split == 'dev':
            returns_slice = self.dev_log_returns
        elif split == 'test':
            returns_slice = self.test_log_returns
        else:
            raise ValueError("Split must be 'train', 'dev', or 'test'")
        
        if returns_slice is None or returns_slice.size == 0:
            return np.array([])

        returns_transposed = returns_slice.T
        return returns_transposed.reshape(-1)
    
    def get_dates_for_split(self, split='test'):
        """Get dates for specified split"""
        if self.current_dates is None:
            return None
        
        if split == 'train':
            return self.current_dates.get('train', None)
        elif split == 'dev':
            return self.current_dates.get('dev', None)
        elif split == 'test':
            return self.current_dates.get('test', None)
        else:
            raise ValueError("Split must be 'train', 'dev', or 'test'")
    
    def get_snp500_returns_for_split(self, split='test'):
        """Get S&P 500 returns for specified split"""
        if self.current_snp500_returns is None:
            return None
        
        if split == 'train':
            returns = self.current_snp500_returns.get('train', None)
        elif split == 'dev':
            returns = self.current_snp500_returns.get('dev', None)
        elif split == 'test':
            returns = self.current_snp500_returns.get('test', None)
        else:
            raise ValueError("Split must be 'train', 'dev', or 'test'")
        
        if returns is None or returns.size == 0:
            return np.array([])
        
        num_companies = self.num_companies
        num_timesteps = len(returns)
        repeated_returns = np.repeat(returns, num_companies)
        
        return repeated_returns

    def check_dimensions(self):
        """Check dimensions for current phase"""
        print("\n" + "=" * 80)
        print(f"DIMENSION CHECK - PHASE {self.current_phase}")
        print("=" * 80)
        
        def check_split(windows, labels, sampled_neighbors, split_name):
            print(f"\n{split_name} Split:")
            if len(windows) == 0:
                print("  ⚠️  No Data!")
                return
            
            num_timesteps = len(windows)
            print(f"  Number of timesteps (batches): {num_timesteps}")
            
            # Check first timestep
            win_shape = windows[0].shape
            label_shape = labels[0].shape
            
            print(f"  Batch 0:")
            print(f"    Windows shape: {win_shape}")
            print(f"    Labels shape: {label_shape}")
            
            assert win_shape[0] == label_shape[0]
            assert win_shape[0] == self.num_companies
            assert win_shape[1] == self.lookback
            
            if len(sampled_neighbors) > 0:
                sampled_shape = sampled_neighbors.shape
                print(f"  Sampled neighbors (full split): {sampled_shape}")
                assert sampled_shape[0] == num_timesteps
                assert sampled_shape[1] == self.num_relations
                assert sampled_shape[2] == self.num_companies
                assert sampled_shape[3] == self.neighbors_sample
            
            print(f"  ✅ All dimensions consistent!")
        
        check_split(self.train_windows, self.train_labels, 
                    self.train_sampled_neighbors, "TRAIN")
        check_split(self.dev_windows, self.dev_labels,
                    self.dev_sampled_neighbors, "DEV")
        check_split(self.test_windows, self.test_labels,
                    self.test_sampled_neighbors, "TEST")
        
        print("\n" + "=" * 80)
        print("✅ ALL DIMENSION CHECKS PASSED")
        print("=" * 80)

    def print_dataset_stats(self):
        """Print stats for all phases"""
        print("\n" + "=" * 80)
        print("MULTI-PHASE DATASET STATISTICS")
        print("=" * 80)
        
        print(f"\nNumber of companies: {self.num_companies}")
        print(f"Number of static relations: {self.num_relations}")
        print(f"Total phases: {len(self.phases)}")
        print(f"Current active phase: {self.current_phase}")
        
        print("\nPer-phase breakdown (timesteps/batches):")
        print("-" * 80)
        print(f"{'Phase':<8} {'Train':<10} {'Dev':<10} {'Test':<10}")
        print("-" * 80)
        
        for i, phase in enumerate(self.phases):
            n_train = len(phase['train_windows'])
            n_dev = len(phase['dev_windows'])
            n_test = len(phase['test_windows'])
            marker = " <--" if i == self.current_phase else ""
            print(f"{i:<8} {n_train:<10} {n_dev:<10} {n_test:<10}{marker}")
        
        print("-" * 80)
        print("\n✓ Use dataset.set_phase(i) to switch between phases")
        print("=" * 80)