import matplotlib.pyplot as plt

def plot_zone_trends(final_report):

    # Get a sorted list of unique zones to ensure they plot in order (1 to 20)
    zones = sorted(final_report['Zone'].dropna().unique())

    # Create a 5x4 grid of subplots. 
    # figsize=(20, 15) gives a nice widescreen aspect ratio.
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 15), sharex=True, sharey=True)

    # Flatten the 5x4 matrix into a 1D list of 20 axes so we can loop over it easily
    axes = axes.flatten()

    for i, zone in enumerate(zones):
        ax = axes[i]
        zone_data = final_report[final_report['Zone'] == zone].copy()
        
        # 1. Plot raw data
        ax.plot(zone_data['Sample_ID'], zone_data['Median_Val'], 
                marker='o', linestyle='', markersize=3, alpha=0.4, 
                color='#1f77b4', label='Raw Median')
        
        # 2. Calculate and plot the 50-sample rolling average
        zone_data['Rolling_Mean'] = zone_data['Median_Val'].rolling(window=50, min_periods=1).mean()
        ax.plot(zone_data['Sample_ID'], zone_data['Rolling_Mean'], 
                color='red', linewidth=2, label='50-Sample Trend')

        ax.set_title(f'Zone {zone}')
        ax.grid(True, alpha=0.3)

    # Add shared X and Y labels for the entire figure
    fig.supxlabel('Sample ID', fontsize=14)
    fig.supylabel('Median Value (°C)', fontsize=14)

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout()
    plt.show()