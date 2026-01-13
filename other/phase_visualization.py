import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TICKERS_PATH = '../data/tickers.txt'
PRICE_DATA_DIR = '../data/price/'
OUTPUT_FILENAME = 'phase_visualization.png'

TRAIN_DAYS = 350
DEV_DAYS = 70
TEST_DAYS = 140
SLIDE_DAYS = 140
N_PHASES = 12
LOOKBACK = 0  

def create_equal_weighted_index():
    """
    Aggregates returns from all available stocks to create a synthetic 
    equal-weighted market index for visualization purposes.
    """
    print(f"Loading tickers from: {TICKERS_PATH}")
    if not os.path.exists(TICKERS_PATH):
        print(f"Error: {TICKERS_PATH} not found.")
        return None, None

    with open(TICKERS_PATH, 'r') as f:
        tickers = f.read().split()
    
    print(f"Processing {len(tickers)} tickers from: {PRICE_DATA_DIR}...")
    
    all_series = {}
    
    # Load returns for every ticker found in the price directory
    for ticker in tickers:
        file_path = os.path.join(PRICE_DATA_DIR, f'{ticker}.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, usecols=['Date', 'Returns'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                df = df[~df.index.duplicated(keep='first')]
                all_series[ticker] = df['Returns']
            except Exception as e:
                print(f"Error reading {ticker}: {e}")
        else:
            pass

    if not all_series:
        print("Error: No valid data found.")
        return None, None
    
    print("Aligning dates across all stocks...")
    wide_df = pd.DataFrame(all_series)
    wide_df.sort_index(inplace=True)

    # Calculate mean daily return across all stocks (Equal Weighting)
    mean_daily_return = wide_df.mean(axis=1, skipna=True)
    
    # Convert returns to cumulative asset value
    custom_index_values = (1 + mean_daily_return).cumprod()
    
    dates = custom_index_values.index.values
    values = custom_index_values.values
    
    print(f"Index created successfully. Length: {len(values)} trading days.")
    return dates, values

def plot_improved_visualization(dates, values):
    """
    Plots the market index and overlays colored regions representing 
    the sliding window phases (Test periods) and their training/dev history.
    """
    if dates is None or values is None:
        return

    total_days = len(values)
    FIRST_TEST_START_DAY = LOOKBACK + TRAIN_DAYS + DEV_DAYS 
    
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Plot the main market index line
    ax.plot(dates, values, color='black', linewidth=1.2, zorder=10)
    
    # Determine Y-axis limits with padding for top bars
    y_min = np.min(values)
    y_max = np.max(values)
    y_range = y_max - y_min
    
    pad_bottom = y_range * 0.10
    pad_top = y_range * 0.35
    
    ax.set_ylim(y_min - pad_bottom, y_max + pad_top)
    
    cmap = plt.get_cmap('Pastel1') 
    phase_boundaries = []
    
    tick_dates = [dates[0]] 
    last_phase_end_date = None
    
    # Loop through phases to draw Test period highlights
    for i in range(N_PHASES):
        test_start_idx = FIRST_TEST_START_DAY + (i * SLIDE_DAYS)
        test_end_idx = test_start_idx + TEST_DAYS
        
        if test_end_idx >= total_days:
            break
            
        d_test_start = dates[test_start_idx]
        d_test_end = dates[test_end_idx]
        
        tick_dates.append(d_test_start)
        
        last_phase_end_date = d_test_end
        
        # Draw colored vertical span for the Test period
        color = cmap(i % 9) 
        ax.axvspan(d_test_start, d_test_end, color=color, alpha=0.6, ec='gray', lw=0.5)
        
        # Label the phase
        mid_point_idx = test_start_idx + (TEST_DAYS // 2)
        label_y_pos = (y_min - pad_bottom) + (pad_bottom * 0.4)
        
        ax.text(dates[mid_point_idx], label_y_pos, 
                f"Phase {i+1}", 
                rotation=0, 
                ha='center', va='center', fontsize=10, color='#333333')

        # Store indices to draw Training/Dev bars later
        dev_start_idx = test_start_idx - DEV_DAYS
        train_start_idx = dev_start_idx - TRAIN_DAYS
        
        phase_boundaries.append({
            'train_idxs': (train_start_idx, dev_start_idx),
            'dev_idxs': (dev_start_idx, test_start_idx)
        })
        
    if last_phase_end_date is not None:
        tick_dates.append(last_phase_end_date)
    
    # Handle the final date tick
    last_date = dates[-1]
    if last_phase_end_date is not None:
        ts_last_phase = pd.to_datetime(last_phase_end_date)
        ts_last_date = pd.to_datetime(last_date)
        
        if ts_last_date != ts_last_phase:
            tick_dates.append(last_date)
    else:
        tick_dates.append(last_date)
        
    tick_dates = pd.to_datetime(tick_dates).unique()
    tick_dates = sorted(tick_dates)

    # Draw horizontal bars at the top to visualize Train/Dev splits for the first few phases
    bar_height = y_range * 0.05
    current_y_top = ax.get_ylim()[1] - (y_range * 0.02)
    
    for i in range(min(3, len(phase_boundaries))):
        p = phase_boundaries[i]
        
        d_train_start = dates[p['train_idxs'][0]]
        d_train_end   = dates[p['train_idxs'][1]]
        d_dev_start   = dates[p['dev_idxs'][0]]
        d_dev_end     = dates[p['dev_idxs'][1]]
        
        # Draw Training bar (Grey)
        ax.fill_between([d_train_start, d_train_end], 
                        current_y_top, current_y_top - bar_height, 
                        color='#D3D3D3', ec='black', lw=1)
        
        # Draw Dev bar (Yellow)
        ax.fill_between([d_dev_start, d_dev_end], 
                        current_y_top, current_y_top - bar_height, 
                        color='#FFC107', ec='black', lw=1)
        
        mid_train = dates[p['train_idxs'][0] + (TRAIN_DAYS // 2)]
        mid_dev   = dates[p['dev_idxs'][0] + (DEV_DAYS // 2)]
        
        ax.text(mid_train, current_y_top - (bar_height/2), "Training", 
                ha='center', va='center', fontsize=9, fontweight='normal', color='black')
        
        ax.text(mid_dev, current_y_top - (bar_height/2), "Dev", 
                ha='center', va='center', fontsize=8, fontweight='bold', color='black')

        current_y_top -= bar_height
    
    ax.set_xticks(tick_dates)
    ax.set_xticklabels([d.strftime('%d/%m/%y') for d in tick_dates], rotation=0, ha='center', fontsize=8)
    
    ax.set_title("")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Index Value", fontsize=12)
    
    plt.tight_layout()
    
    print(f"Saving to {OUTPUT_FILENAME}...")
    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    dates, index_vals = create_equal_weighted_index()
    
    if dates is not None:
        plot_improved_visualization(dates, index_vals)