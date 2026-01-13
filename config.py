import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='DART',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # important file/folder paths
    dir_group = parser.add_argument_group('Directories and Paths')
    dir_group.add_argument('--data_dir', type=str, default="./data", help="Root directory for data storage.")
    dir_group.add_argument('--processed_dir', type=str, default="./data/processed", help="Directory to store processed pickle files and adjacency matrices.")
    dir_group.add_argument('--price_data_dir', type=str, default="./data/price", help="Directory containing historical price data and technical indicators.")
    dir_group.add_argument('--relation_data_dir', type=str, default="./data/relation", help="Directory containing relational data between stocks.")
    dir_group.add_argument('--tickers_path', type=str, default="./data/tickers.txt", help="Path to the text file listing all company tickers.")
    dir_group.add_argument('--properties_path', type=str, default="./data/properties.txt", help="Path to the text file storing Wikidata properties for companies.")
    dir_group.add_argument('--raw_relations_path', type=str, default="./data/relation/raw_relations.csv", help="Path to the CSV file containing raw relation triples [Source, Target, Relation_Type].")
    dir_group.add_argument('--returns_path', type=str, default="./data/returns.csv", help="Path to the CSV file containing log returns of all stocks.")
    
    dir_group.add_argument('--checkpoint_dir', type=str, default="./logs/checkpoints", help="Directory to save model checkpoints during training.")
    dir_group.add_argument('--log_dir', type=str, default="./logs", help="Directory to save execution logs and tensorboard data.")
    dir_group.add_argument('--max_to_keep', type=int, default=10, help="Maximum number of recent checkpoints to keep.")
    
    # data definition and pre-processing parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--price_feature_list', nargs='+', type=str, 
                            default=['Returns', 
                                     'MACD', 'MACD_Hist', 'MACD_Signal', 
                                     'RSI', 
                                     'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent',
                                     'ADX', 'ADXR', 'DI_Pos', "DI_Neg",
                                     'OBV'], help='List of price and technical indicators to be used as node features.')
    data_group.add_argument('--relational_feature_list', nargs='+', type=str, default=['Source', 'Target', 'Relation_Type'], help='List of column names in the raw relations CSV file.')
    data_group.add_argument('--n_phases', type=int, default=12, help="Number of temporal phases to split the data into for training and testing.")
    data_group.add_argument('--train_days', type=int, default=350, help='Number of days used for training in each phase.')
    data_group.add_argument('--dev_days', type=int, default=70, help='Number of days used for validation/dev in each phase.')
    data_group.add_argument('--test_days', type=int, default=140, help='Number of days used for testing in each phase.')
    data_group.add_argument('--slide_days', type=int, default=140, help="Number of days to shift the window forward for the next phase.")
    data_group.add_argument('--label_proportion', nargs='+', type=int, default=[3, 4, 3], help='Proportion of labels (Up, Neutral, Down) for threshold calculation.')
    data_group.add_argument('--lookback', type=int, default=50, help='Length of the historical window (lookback period) for GNN input.')
    data_group.add_argument('--corr_threshold', type=int, default=0.6, help="Correlation coefficient threshold for establishing dynamic edges between stocks.")
    
    # GNN model parameters
    model_group = parser.add_argument_group('GNN Model Parameters')
    model_group.add_argument('--node_feat_size', type=int, default=128, help="Dimensionality of the node feature embedding after LSTM processing.")
    model_group.add_argument('--use_feat_attention', action='store_true', default=True, help="Flag to enable the State Attention mechanism (Layer 1).")
    model_group.add_argument('--use_rel_attention', action='store_true', default=True, help="Flag to enable the Relational Attention mechanism (Layer 2).")
    model_group.add_argument('--dropout', type=float, default=0.4, help="Dropout probability applied to model layers.")
    model_group.add_argument('--neighbors_sample', type=int, default=15, help="Maximum number of neighbors to sample per relation type (k).")
    model_group.add_argument('--early_stop_patience', type=int, default=15, help="Number of epochs with no improvement on dev set before stopping.")
    model_group.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength applied to model weights.")

    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="Execution mode: 'train' for model training, 'test' for evaluation.")
    train_group.add_argument('--n_epochs', type=int, default=150, help="Maximum number of training epochs.")
    train_group.add_argument('--lr', type=float, default=5e-4, help="Learning rate for the Adam optimizer.")
    train_group.add_argument('--grad_max_norm', type=float, default=3.0, help="Maximum norm for gradient clipping.")
    
    # RL model parameters
    rl_group = parser.add_argument_group('RL Simulator Parameters')
    rl_group.add_argument('--num_episodes', type=int, default=5000, help="Total number of episodes to train the RL agent.")
    rl_group.add_argument('--rl_lr', type=float, default=3e-4, help="Learning rate for the RL agent's optimizer.")
    rl_group.add_argument('--n_steps', type=int, default=2048, help="Number of steps to run in the environment per update batch.")
    rl_group.add_argument('--batch_size', type=int, default=64, help="Minibatch size for PPO network updates.")
    rl_group.add_argument('--rl_epochs', type=int, default=10, help="Number of epochs to optimize the surrogate loss per update.")
    rl_group.add_argument('--gamma', type=float, default=0.95, help="Discount factor for future rewards.")
    rl_group.add_argument('--gae_lambda', type=float, default=0.95, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator.")
    rl_group.add_argument('--clip_range', type=float, default=0.2, help="Clipping parameter for the PPO surrogate loss function.")
    rl_group.add_argument('--ent_coef', type=float, default=0.01, help="Entropy coefficient to encourage exploration.")
    rl_group.add_argument('--vf_coef', type=float, default=0.5, help="Coefficient for the value function loss component.")
    rl_group.add_argument('--max_grad_norm', type=float, default=0.5, help="Maximum allowed gradient norm for RL gradient clipping.")
    rl_group.add_argument('--net_arch', nargs='+', type=int, default=[1024, 512, 128], help="List defining the number of units in each hidden layer of the policy/value networks.")
    rl_group.add_argument('--general_vol_penalty', type=float, default=0.1, help="Weight for the general volatility penalty (Sharpe-like) in the reward function.")
    rl_group.add_argument('--downside_vol_penalty', type=float, default=0.1, help="Weight for the downside volatility penalty (Sortino-like) in the reward function.")
    rl_group.add_argument('--drawdown_penalty_weight', type=float, default=0.1, help="Weight for the maximum drawdown penalty in the reward function.")
    rl_group.add_argument('--cost_penalty_weight', type=float, default=0.05, help="Weight for the transaction cost penalty in the reward function.")
    rl_group.add_argument('--reward_window_size', type=int, default=20, help="Rolling window size for calculating recent volatility metrics.")
    
    rl_group.add_argument('--initial_balance', type=float, default=1000000.0, help="Initial capital available for the trading agent.")
    rl_group.add_argument('--transaction_cost', type=float, default=0.001, help="Transaction cost rate per trade (e.g., 0.001 represents 0.1%%).")
    rl_group.add_argument('--turbulence_threshold', type=float, default=25.0, help="Threshold value for the market turbulence index to trigger defensive actions.")
    
    return parser.parse_args()

if __name__ == "__main__":
    get_args()