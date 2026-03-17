import matplotlib.pyplot as plt
import math
import numpy as np

def plot_zone_trends(final_report):
    """Plots a grid of 20 subplots (5 rows x 4 columns) for each zone's median temperature trends."""
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

def plot_combined_zone_trends(final_report):
    """
    Plots a grid for the combined pH/Temp simulation with 6 plots per row 
    (3 zones per row: pH and Temp for each).
    """
    if 'Median_pH' not in final_report.columns or 'Median_Temp' not in final_report.columns:
        print("Error: The report does not contain combined 'Median_pH' and 'Median_Temp' columns.")
        return

    # Get sorted zones
    zones = sorted(final_report['Zone'].dropna().unique())
    num_zones = len(zones)

    # Calculate Grid Layout (3 zones per row = 6 plots per row)
    zones_per_row = 3
    ncols = zones_per_row * 2
    nrows = math.ceil(num_zones / zones_per_row)

    # Adjust figsize: Width of 24 comfortably fits 6 plots. 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 3.5 * nrows))

    # Flatten the axes array to easily iterate and turn off unused plots later
    # (Handling the edge case where there's only 1 row)
    if nrows == 1:
        axes = [axes]
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes

    for i, zone in enumerate(zones):
        zone_data = final_report[final_report['Zone'] == zone].copy()
        
        # Calculate exactly which row and columns this zone belongs to
        row = i // zones_per_row
        col_ph = (i % zones_per_row) * 2
        col_temp = col_ph + 1
        
        # Access the specific axes
        if nrows == 1:
            ax_ph = axes_flat[col_ph]
            ax_temp = axes_flat[col_temp]
        else:
            ax_ph = axes[row, col_ph]
            ax_temp = axes[row, col_temp]
        
        # --- Plot pH (Left Plot of the Pair) ---
        ax_ph.plot(zone_data['Sample_ID'], zone_data['Median_pH'], 
                   marker='o', linestyle='', markersize=3, alpha=0.4, 
                   color='#2ca02c', label='Raw pH')
        
        zone_data['Rolling_pH'] = zone_data['Median_pH'].rolling(window=50, min_periods=1).mean()
        ax_ph.plot(zone_data['Sample_ID'], zone_data['Rolling_pH'], 
                   color='darkgreen', linewidth=2, label='Trend')
        
        ax_ph.set_title(f'Zone {zone} - pH')
        ax_ph.set_ylabel('pH')
        ax_ph.grid(True, alpha=0.3)
        ax_ph.legend(loc='best', fontsize=8)
        
        # --- Plot Temperature (Right Plot of the Pair) ---
        ax_temp.plot(zone_data['Sample_ID'], zone_data['Median_Temp'], 
                     marker='o', linestyle='', markersize=3, alpha=0.4, 
                     color='#1f77b4', label='Raw Temp')
        
        zone_data['Rolling_Temp'] = zone_data['Median_Temp'].rolling(window=50, min_periods=1).mean()
        ax_temp.plot(zone_data['Sample_ID'], zone_data['Rolling_Temp'], 
                     color='red', linewidth=2, label='Trend')
        
        ax_temp.set_title(f'Zone {zone} - Temp')
        ax_temp.set_ylabel('Temp (°C)')
        ax_temp.grid(True, alpha=0.3)
        ax_temp.legend(loc='best', fontsize=8)

    # Hide any unused subplots at the end of the grid (e.g., if 20 zones, spots 41 and 42 are empty)
    total_plots_needed = num_zones * 2
    for j in range(total_plots_needed, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.supxlabel('Sample ID (Time)', fontsize=18)
    
    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    plt.show()