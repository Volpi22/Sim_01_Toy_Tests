import matplotlib.pyplot as plt
import math
import numpy as np

def plot_zone_trends(final_report, target_zones=None):
    """
    Plots a dynamic grid for single-parameter zone trends.
    If target_zones is provided (e.g., [7, 16, 19]), it will only plot those zones.
    """
    # 1. Filter Zones
    available_zones = sorted(final_report['Zone'].dropna().unique())
    if target_zones is not None:
        zones = [z for z in target_zones if z in available_zones]
    else:
        zones = available_zones
        
    num_zones = len(zones)
    if num_zones == 0:
        print("Error: No valid zones to plot.")
        return

    # 2. Dynamic Grid Layout (Max 4 columns wide)
    ncols = min(num_zones, 4)
    nrows = math.ceil(num_zones / ncols)

    # Scale figure size dynamically based on rows and columns
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True, sharey=True)

    # np.array ensures we can consistently flatten whether axes is a single object, 1D array, or 2D array
    axes_flat = np.array(axes).flatten()

    for i, zone in enumerate(zones):
        ax = axes_flat[i]
        zone_data = final_report[final_report['Zone'] == zone].copy()
        
        # Plot raw data
        ax.plot(zone_data['Sample_ID'], zone_data['Median_Val'], 
                marker='o', linestyle='', markersize=3, alpha=0.4, 
                color='#1f77b4', label='Raw Median')
        
        # Calculate and plot the 8-sample rolling average
        zone_data['Rolling_Mean'] = zone_data['Median_Val'].rolling(window=8, min_periods=1).mean()
        ax.plot(zone_data['Sample_ID'], zone_data['Rolling_Mean'], 
                color='red', linewidth=2, label='8-Sample Trend')

        ax.set_title(f'Zone {zone}')
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots at the end of the grid
    for j in range(num_zones, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Add shared X and Y labels for the entire figure
    fig.supxlabel('Sample ID', fontsize=14)
    fig.supylabel('Median Value', fontsize=14)

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout()
    plt.show()


def plot_combined_zone_trends(final_report, target_zones=None):
    """
    Plots a dynamic grid for the combined pH/Temp simulation.
    If target_zones is provided, it will only plot those specific zones.
    """
    if 'Median_pH' not in final_report.columns or 'Median_Temp' not in final_report.columns:
        print("Error: The report does not contain combined 'Median_pH' and 'Median_Temp' columns.")
        return

    # 1. Filter Zones
    available_zones = sorted(final_report['Zone'].dropna().unique())
    if target_zones is not None:
        zones = [z for z in target_zones if z in available_zones]
    else:
        zones = available_zones
        
    num_zones = len(zones)
    if num_zones == 0:
        print("Error: No valid zones to plot.")
        return

    # 2. Dynamic Grid Layout (Max 3 zones per row -> 6 plots per row)
    zones_per_row = min(num_zones, 3)
    ncols = zones_per_row * 2
    nrows = math.ceil(num_zones / zones_per_row)

    # Scale figure size dynamically 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * zones_per_row, 3.5 * nrows))

    # Safely flatten the axes array
    axes_flat = np.array(axes).flatten()

    for i, zone in enumerate(zones):
        zone_data = final_report[final_report['Zone'] == zone].copy()
        
        # Calculate index positions: every zone takes 2 consecutive spots in the flat array
        idx_ph = i * 2
        idx_temp = i * 2 + 1
        
        ax_ph = axes_flat[idx_ph]
        ax_temp = axes_flat[idx_temp]
        
        # --- Plot pH (Left Plot of the Pair) ---
        ax_ph.plot(zone_data['Sample_ID'], zone_data['Median_pH'], 
                   marker='o', linestyle='', markersize=3, alpha=0.4, 
                   color='#2ca02c', label='Raw pH')
        
        zone_data['Rolling_pH'] = zone_data['Median_pH'].rolling(window=8, min_periods=1).mean()
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
        
        zone_data['Rolling_Temp'] = zone_data['Median_Temp'].rolling(window=8, min_periods=1).mean()
        ax_temp.plot(zone_data['Sample_ID'], zone_data['Rolling_Temp'], 
                     color='red', linewidth=2, label='Trend')
        
        ax_temp.set_title(f'Zone {zone} - Temp')
        ax_temp.set_ylabel('Temp (°C)')
        ax_temp.grid(True, alpha=0.3)
        ax_temp.legend(loc='best', fontsize=8)

    # Hide any unused subplots at the end of the grid
    total_plots_needed = num_zones * 2
    for j in range(total_plots_needed, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.supxlabel('Sample ID (Time)', fontsize=16)
    
    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    plt.show()