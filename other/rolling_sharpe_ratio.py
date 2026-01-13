import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import os

LOG_FILES = [
    ("../baseline models/logs/MLPModel_overall_result.json", "MLP"),
    ("../baseline models/logs/LSTMModel_overall_result.json", "LSTM"),
    ("../baseline models/logs/CNNModel_overall_result.json", "CNN"),
    ("../baseline models/logs/CNN_LSTMModel_overall_result.json", "CNN-LSTM"),
    ("../baseline models/logs/GCNModel_overall_result.json", "GCN"),
    ("../baseline models/logs/GATModel_overall_result.json", "GAT"),
    ("../baseline models/logs/HATSModel_overall_result.json", "HATS"),
]

MY_MODEL_FOLDER = "../logs/rl_results"
MY_MODEL_NAME = "DART"

MODEL_COLOR_LIST = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#b15928', '#6a3d9a']
COLORS = {
    MY_MODEL_NAME: '#084594', 
    "MLP": MODEL_COLOR_LIST[0],
    "LSTM": MODEL_COLOR_LIST[1],
    "CNN": MODEL_COLOR_LIST[2],
    "CNN-LSTM": MODEL_COLOR_LIST[3],
    "GCN": MODEL_COLOR_LIST[4],
    "GAT": MODEL_COLOR_LIST[5],
    "HATS": MODEL_COLOR_LIST[6],
}

RISK_FREE_RATE = 0.02 / 252 
ROLLING_WINDOW_DAYS = 60    
SMOOTHING_INTERVAL = '10D'  

def load_all_returns(filepath, display_name, is_rl_agent=False):
    """
    Parses log files to extract daily returns.
    """
    all_returns = []
    phase_start_dates = {}
    
    if is_rl_agent:
        # Iterate through phase 0-11 for the RL agent folder structure
        for i in range(12): 
            filepath_full = os.path.join(filepath, f"phase_{i}_results.json")
            if not os.path.exists(filepath_full): continue
            
            try:
                with open(filepath_full, 'r') as f:
                    data = json.load(f)
                daily_data = data.get('daily_returns', [])
                
                if daily_data:
                    # Capture the start date of each phase
                    if i not in phase_start_dates:
                        phase_start_dates[i] = daily_data[0]['date']
                    for entry in daily_data:
                        all_returns.append({'date': entry['date'], 'return': entry['return']})
            except Exception:
                continue
    else:
        # Standard processing for baseline models
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            return None, {}
            
        phases = sorted(data.get('phase_results', []), key=lambda x: x['phase'])
        
        for phase in phases:
            daily_data = phase.get('daily_returns', [])
            if daily_data:
                if phase['phase'] not in phase_start_dates:
                    phase_start_dates[phase['phase']] = daily_data[0]['date']
                for entry in daily_data:
                    all_returns.append({'date': entry['date'], 'return': entry['return']})
    
    if not all_returns:
        return None, {}

    df = pd.DataFrame(all_returns)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset='date', keep='last').sort_values('date')
    df.set_index('date', inplace=True)
    return df['return'].rename(display_name), phase_start_dates

def calculate_rolling_sharpe(returns_series, window, risk_free_rate, smoothing_interval):
    """
    Computes the annualized Rolling Sharpe Ratio.
    1. Calculate excess returns (Return - RiskFree).
    2. Compute rolling mean and std dev.
    3. Annualize the ratio.
    4. Resample (smooth) the curve to reduce visual noise.
    """
    excess_returns = returns_series - risk_free_rate
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = returns_series.rolling(window=window).std()
    
    sharpe_ratio_raw = (rolling_mean / rolling_std) * np.sqrt(252)
    sharpe_ratio_raw = sharpe_ratio_raw.dropna()

    # Downsample to smooth the line plot (e.g., 1 point every 10 days)
    sharpe_ratio_smoothed = sharpe_ratio_raw.resample(smoothing_interval).mean().ffill().dropna()
    
    return sharpe_ratio_smoothed.rename(returns_series.name)

def plot_rolling_sharpe(data_dict, phase_dates, output_filename='rolling_sharpe_ratio_closed.png'):
    """
    Plots the rolling Sharpe Ratio comparison.
    Ensures the plot frame is fully closed (all 4 spines visible) and aligns ticks with phase starts.
    """
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Generate X-axis ticks based on phase boundaries + end date
    start_dates_list = [pd.to_datetime(date) for date in phase_dates.values()]
    start_dates_list = sorted(list(set(start_dates_list)))
    end_date = data_dict[MY_MODEL_NAME].index.max()
    all_dates = sorted(list(set(start_dates_list + [end_date])))
    
    x_positions = all_dates
    x_labels = [date.strftime('%Y-%m-%d') for date in all_dates]

    for name, series in data_dict.items():
        color = COLORS.get(name, '#666666')
        
        linewidth = 1.0 
        alpha = 0.8
        zorder = 3

        # Highlight the custom model
        if name == MY_MODEL_NAME:
            linewidth = 1.5 
            alpha = 1.0
            zorder = 4

        ax.plot(series.index, series.values, label=name, 
                color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

    # Ensure a "box" style frame by making all spines visible
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # Add inward ticks on all sides for a cleaner scientific look
    ax.tick_params(axis='both', which='both', length=4)
    ax.tick_params(axis='y', labelleft=True, labelright=False)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    
    # Add reference line at Sharpe=0
    ax.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.8, zorder=5) 
    ax.set_ylabel("Sharpe Ratio", fontsize=12) 
    
    # Vertical lines indicating phase changes
    for date in start_dates_list:
        ax.axvline(date, color='#eeeeee', linestyle='-', linewidth=0.8, zorder=0)

    ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=9, color='#333333', ha='center')
    ax.set_xlabel("Date", fontsize=12)
    
    ax.legend(
        loc='upper left', 
        ncol=1, 
        frameon=True, 
        fontsize=9, 
        edgecolor='#cccccc', 
        fancybox=False
    )
    
    plt.tight_layout(pad=0.2)
    plt.savefig(output_filename, dpi=400)
    print(f"Graph saved as '{output_filename}'")
    plt.close()


def main():
    """
    Main execution pipeline:
    1. Loads returns for baselines and the RL agent.
    2. Aligns all data to a common timeframe.
    3. Calculates rolling sharpe ratio.
    4. Generates the final plot.
    """
    all_returns_data = {}
    all_phase_dates = {}
    
    # Load baselines
    for filepath, display_name in LOG_FILES:
        returns_series, phase_dates = load_all_returns(filepath, display_name, is_rl_agent=False)
        if returns_series is not None:
            all_returns_data[display_name] = returns_series
            all_phase_dates.update(phase_dates)

    # Load RL agent
    rl_returns_series, rl_phase_dates = load_all_returns(MY_MODEL_FOLDER, MY_MODEL_NAME, is_rl_agent=True)
    if rl_returns_series is not None:
        all_returns_data[MY_MODEL_NAME] = rl_returns_series
        all_phase_dates.update(rl_phase_dates)

    if not all_returns_data or not all_phase_dates:
        print("\nFATAL ERROR: No valid return or phase date data loaded. Cannot generate plot.")
        return

    # Align dates to ensure fair comparison
    df_returns = pd.concat(all_returns_data, axis=1).dropna()

    if df_returns.empty:
        print("\nERROR: Dataframes could not be aligned due to non-overlapping dates or empty input.")
        return
        
    # Calculate Metrics
    rolling_sharpe_data = {}
    for name, returns in df_returns.items():
        sharpe = calculate_rolling_sharpe(returns, ROLLING_WINDOW_DAYS, RISK_FREE_RATE, SMOOTHING_INTERVAL)
        rolling_sharpe_data[name] = sharpe
    
    # Plots the graph
    plot_rolling_sharpe(rolling_sharpe_data, all_phase_dates, 'rolling_sharpe_ratio.png')


if __name__ == "__main__":
    main()