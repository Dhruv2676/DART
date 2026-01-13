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

PRIMARY_MODEL_COLOR = '#cc0000' 
PRIMARY_MODEL_FILL_COLOR = '#880000'
COLORS = {
    MY_MODEL_NAME: PRIMARY_MODEL_COLOR, 
    "MLP": '#a6cee3',
    "LSTM": '#b2df8a',
    "CNN": '#fb9a99',
    "CNN-LSTM": '#fdbf6f',
    "GCN": '#cab2d6',
    "GAT": '#b15928',
    "HATS": '#6a3d9a',
}

# Resampling Interval: Smoothes the curve by taking data every 10 days to reduce noise
RESAMPLING_INTERVAL = '10D'

def load_all_returns(filepath, display_name, is_rl_agent=False):
    """
    Loads daily returns from logs.
    Returns: (pd.Series of returns, dictionary of phase start dates)
    """
    all_returns = []
    phase_start_dates = {}
    
    if is_rl_agent:
        # RL Agent logs are split into separate files per phase (0-11)
        for i in range(12): 
            filepath_full = os.path.join(filepath, f"phase_{i}_results.json")
            if not os.path.exists(filepath_full): continue
            
            try:
                with open(filepath_full, 'r') as f:
                    data = json.load(f)
                daily_data = data.get('daily_returns', [])
                
                if daily_data:
                    # Store phase start date to mark on the X-axis later
                    if i not in phase_start_dates:
                        phase_start_dates[i] = daily_data[0]['date']
                    for entry in daily_data:
                        all_returns.append({'date': entry['date'], 'return': entry['return']})
            except Exception:
                continue
    else:
        # Baseline models store all phases in one large JSON file
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            return None, []
            
        phases = sorted(data.get('phase_results', []), key=lambda x: x['phase'])
        
        for phase in phases:
            daily_data = phase.get('daily_returns', [])
            if daily_data:
                if phase['phase'] not in phase_start_dates:
                    phase_start_dates[phase['phase']] = daily_data[0]['date']
                for entry in daily_data:
                    all_returns.append({'date': entry['date'], 'return': entry['return']})
    
    if not all_returns:
        return None, []

    df = pd.DataFrame(all_returns)
    df['date'] = pd.to_datetime(df['date'])
    # Remove overlaps from sliding window (keep the latest data point for any date)
    df = df.drop_duplicates(subset='date', keep='last').sort_values('date')
    df.set_index('date', inplace=True)
    return df['return'].rename(display_name), phase_start_dates

def calculate_drawdown(returns_series, interval):
    """
    Calculates the Drawdown percentage curve.
    Includes resampling step for smoother visualization.
    """
    # 1. Calculate Cumulative Returns
    cumulative_return = (1 + returns_series).cumprod()
    
    # 2. Resample (Downsample) to reduce high-frequency noise
    if interval:
        cumulative_return = cumulative_return.resample(interval).last().dropna()
        
    # 3. Compute Drawdown
    max_value = cumulative_return.cummax()
    drawdown = (cumulative_return / max_value) - 1.0
    
    return drawdown.rename(returns_series.name)

def plot_drawdown_curve(data_dict, phase_dates, output_filename='drawdown_final_closed.png'):
    """
    Generates a comparative Drawdown plot with filled areas for the custom model.
    Ensures the plot frame is fully closed and axes are aligned with phase dates.
    """
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 1. Prepare X-Axis Ticks (Phase Starts + End Date)
    start_dates = [pd.to_datetime(date) for date in phase_dates.values()]
    start_dates = sorted(list(set(start_dates)))
    end_date = data_dict[MY_MODEL_NAME].index.max()
    all_dates = sorted(list(set(start_dates + [end_date])))
    
    x_positions = all_dates
    x_labels = [date.strftime('%Y-%m-%d') for date in all_dates]

    # 2. Plot Curves
    for name, series in data_dict.items():
        drawdown_pct = series.values * 100
        color = COLORS.get(name, '#666666')
        
        linewidth = 1.0 
        alpha = 0.8
        zorder = 3

        if name == MY_MODEL_NAME:
            linewidth = 1.0 
            alpha = 1.0
            zorder = 4
            
            # Highlight model with filled area under the curve
            ax.fill_between(series.index, drawdown_pct, 0, 
                            where=(drawdown_pct < 0), 
                            color=PRIMARY_MODEL_FILL_COLOR, alpha=0.3, interpolate=True)

        ax.plot(series.index, drawdown_pct, label=name, 
                color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

    # 3. Y-Axis Formatting
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.8, zorder=5)
    ax.set_yticklabels([f'{int(y)}%' for y in ax.get_yticks()], color='#333333')
    ax.set_ylabel("Drawdown (%)", fontsize=12) 
    
    # 4. Vertical Phase Dividers
    for date in start_dates:
        ax.axvline(date, color='#eeeeee', linestyle='-', linewidth=0.8, zorder=0)

    ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    # 5. Frame Styling (Closed Box)
    # Explicitly enable all four spines
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # Enable ticks on all sides
    ax.tick_params(axis='both', which='both', length=4)
    ax.tick_params(axis='y', labelleft=True, labelright=False)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    
    # 6. X-Axis Labeling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=8, color='#333333', ha='center')
    ax.set_xlabel("Date", fontsize=12)
    
    # 7. Legend
    ax.legend(
        loc='lower left', 
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
    all_returns_data = {}
    all_phase_dates = {}
    
    # 1. Load Baseline Model Data
    for filepath, display_name in LOG_FILES:
        returns_series, phase_dates = load_all_returns(filepath, display_name, is_rl_agent=False)
        if returns_series is not None:
            all_returns_data[display_name] = returns_series
            if not all_phase_dates:
                all_phase_dates.update(phase_dates)

    # 2. Load RL Agent Data
    rl_returns_series, rl_phase_dates = load_all_returns(MY_MODEL_FOLDER, MY_MODEL_NAME, is_rl_agent=True)
    if rl_returns_series is not None:
        all_returns_data[MY_MODEL_NAME] = rl_returns_series
        all_phase_dates.update(rl_phase_dates)

    if not all_returns_data or not all_phase_dates:
        print("\nFATAL ERROR: No valid return or phase date data loaded. Cannot generate plot.")
        return

    # 3. Align Data (Intersection of dates)
    df_returns = pd.concat(all_returns_data, axis=1).dropna()

    if df_returns.empty:
        print("\nERROR: Dataframes could not be aligned due to non-overlapping dates or empty input.")
        return
        
    # 4. Calculate Drawdowns
    drawdown_data = {}
    for name, returns in df_returns.items():
        drawdown = calculate_drawdown(returns, RESAMPLING_INTERVAL)
        drawdown_data[name] = drawdown
    
    # 5. Plot
    plot_drawdown_curve(drawdown_data, all_phase_dates, 'drawdown.png')


if __name__ == "__main__":
    main()