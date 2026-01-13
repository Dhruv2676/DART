import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

SNP_FILE = "../data/snp500_returns.csv"

MY_MODEL_FOLDER = "../logs/rl_results"
MY_MODEL_NAME = "DART"

COLORS = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#b15928', '#6a3d9a']
BASELINE_COLOR = '#333333'
MY_MODEL_COLOR = '#000000'

def process_log_file(filepath, display_name):
    """
    Reads a single JSON log file, extracts daily returns across all phases, 
    and calculates the cumulative asset value.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        all_returns = []
        phases = sorted(data.get('phase_results', []), key=lambda x: x['phase'])
        
        # Aggregate returns from all phases into a single list
        for phase in phases:
            daily_data = phase.get('daily_returns', [])
            for entry in daily_data:
                all_returns.append({'date': entry['date'], 'return': entry['return']})
                
        if not all_returns:
            print(f"Warning: No return data found in {filepath}")
            return None

        # Convert to DataFrame and handle date overlaps between phases
        df = pd.DataFrame(all_returns)
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset='date', keep='last').sort_values('date')
        df.set_index('date', inplace=True)
        
        # Calculate cumulative asset value starting at 100
        df['asset_value'] = 100 * (1 + df['return']).cumprod()
        
        return display_name, df['asset_value']

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def process_csv_file(filepath, display_name, start_date=None, end_date=None):
    """
    Reads the S&P 500 CSV file, trims it to the specified date range,
    and calculates the cumulative asset value.
    """
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').drop_duplicates(subset='Date', keep='last')
        df.set_index('Date', inplace=True)
        
        # Filter data to match the timeline of the trained models
        if start_date and end_date:
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
        if df.empty:
            return None

        df['asset_value'] = 100 * (1 + df['Returns']).cumprod()
        return display_name, df['asset_value']
        
    except Exception as e:
        print(f"Error processing CSV {filepath}: {e}")
        return None

def process_phase_folder(folder_path, display_name):
    """
    Reads individual phase result files (phase_0 to phase_11) from a folder,
    aggregates them into a continuous timeline, and captures phase start dates for plotting.
    """
    try:
        all_returns = []
        phase_start_dates = []
        
        # Iterate through all 12 phases
        for i in range(12):
            filename = f"phase_{i}_results.json"
            filepath = os.path.join(folder_path, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: {filename} not found in {folder_path}. Skipping.")
                continue
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            daily_data = data.get('daily_returns', [])
            
            # Store the start date of the phase for X-axis ticks
            if daily_data:
                first_date = pd.to_datetime(daily_data[0]['date'])
                phase_start_dates.append(first_date)

            for entry in daily_data:
                all_returns.append({
                    'date': entry['date'],
                    'return': entry['return']
                })
        
        if not all_returns:
            print(f"Warning: No valid phase data found in {folder_path}")
            return None, None

        df = pd.DataFrame(all_returns)
        df['date'] = pd.to_datetime(df['date'])
        
        # Add the final date to the ticks
        last_date = df['date'].max()
        phase_start_dates.append(last_date)
        
        # Handle overlapping dates from sliding windows
        df = df.sort_values('date').drop_duplicates(subset='date', keep='last')
        df.set_index('date', inplace=True)
        
        df['asset_value'] = 100 * (1 + df['return']).cumprod()
        
        return display_name, df['asset_value'], sorted(list(set(phase_start_dates)))

    except Exception as e:
        print(f"Error processing phase folder {folder_path}: {e}")
        return None, None

def plot_performance(data_dict, my_model_name, specific_ticks=None):
    """
    Generates a comparative line chart of asset values over time.
    Highlights the custom model and S&P 500, assigns colors to baselines,
    and sets X-axis ticks based on phase boundaries.
    """
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    color_idx = 0
    
    # Iterate through all loaded data series to plot
    for name, series in data_dict.items():
        if name == "S&P 500":
            color = BASELINE_COLOR
            linewidth = 1.5
            alpha = 0.8
            zorder = 1
        elif name == my_model_name:
            color = MY_MODEL_COLOR 
            linewidth = 2.0        
            alpha = 1.0            
            zorder = 3             
        else:
            # Cycle through colors for baseline models
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1
            linewidth = 1.5
            alpha = 0.9
            zorder = 2

        ax.plot(series.index, series.values, label=name, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        
    # Format X-axis ticks to align with phase start dates if available
    if specific_ticks:
        ax.set_xticks(specific_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#d9d9d9')
    
    # Stylize borders
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('#555555')
        ax.spines[spine].set_linewidth(1.0)
    
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#555555', fancybox=False, borderpad=1)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Asset Value", fontsize=12)

    plt.tight_layout()
    plt.savefig('asset_value.png', dpi=300)
    print("Graph saved as 'asset_value.png'")
    plt.show()

def main():
    """
    Main execution flow:
    1. Loads baseline model logs.
    2. Loads custom RL model logs (folder based).
    3. Calculates common date range.
    4. Loads and trims S&P 500 data.
    5. plots the comparison graph.
    """
    model_data = {}
    custom_model_ticks = [] 
    
    # 1. Load Baseline Models
    print("Processing baseline models...")
    for filepath, display_name in LOG_FILES:
        result = process_log_file(filepath, display_name)
        if result:
            name, series = result
            model_data[name] = series
            print(f"Loaded {name}")

    # 2. Load Custom RL Model (Phase Folder)
    print("Processing custom model folder...")
    if os.path.exists(MY_MODEL_FOLDER):
        result_tuple = process_phase_folder(MY_MODEL_FOLDER, MY_MODEL_NAME)
        # result_tuple contains: (Name, Series, PhaseDates)
        if result_tuple[0] is not None:
            name, series, phase_dates = result_tuple
            model_data[name] = series
            custom_model_ticks = phase_dates 
            print(f"Loaded {name}")
            print(f"Final value of DART is {series[-1]}")
    else:
        print(f"Custom model folder not found: {MY_MODEL_FOLDER}")

    # 3. Determine Date Range to trim S&P 500 data
    overall_min_date = None
    overall_max_date = None
    
    for series in model_data.values():
        min_d = series.index.min()
        max_d = series.index.max()
        if overall_min_date is None or min_d < overall_min_date:
            overall_min_date = min_d
        if overall_max_date is None or max_d > overall_max_date:
            overall_max_date = max_d

    # 4. Load and Process S&P 500 Benchmark
    if overall_min_date and overall_max_date:
        snp_result = process_csv_file(SNP_FILE, "S&P 500", start_date=overall_min_date, end_date=overall_max_date)
        if snp_result:
            name, series = snp_result
            model_data[name] = series
            print(f"Loaded {name} (trimmed)")

    # 5. Plot Results
    if model_data:
        plot_performance(model_data, MY_MODEL_NAME, specific_ticks=custom_model_ticks)
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    main()