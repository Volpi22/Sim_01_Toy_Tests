import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

class SimulationVisualizer:
    """
    Handles data visualization for Nanobot simulation results, 
    supporting both single-parameter and multi-parameter (combined) reports.
    """
    
    def __init__(self, style="seaborn-v0_8-muted"):
        self.style = style
        # Default colors for consistency
        self.colors = {
            'raw': '#1f77b4',     # Muted Blue
            'trend': '#d62728',   # Strong Red
            'ph_raw': '#2ca02c',  # Green
            'ph_trend': '#1a591a' # Dark Green
        }

    def _prepare_zones(self, final_report, target_zones):
        """Filters and sorts available zones for plotting."""
        available = sorted(final_report['Zone'].dropna().unique())
        if target_zones:
            return [z for z in target_zones if z in available]
        return available

    def plot_zone_trends(self, final_report, target_zones=None, window=8):
        """
        Plots a grid of trends for a single parameter (e.g., Pressure or SO2).
        """
        zones = self._prepare_zones(final_report, target_zones)
        num_zones = len(zones)
        if num_zones == 0:
            print("No valid zones found for plotting."); return

        # Calculate grid dimensions (Max 4 columns)
        ncols = min(num_zones, 4)
        nrows = math.ceil(num_zones / ncols)

        with plt.style.context(self.style):
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, 
                figsize=(5 * ncols, 3.8 * nrows), 
                sharex=True
            )
            axes_flat = np.array(axes).flatten()

            for i, zone in enumerate(zones):
                ax = axes_flat[i]
                data = final_report[final_report['Zone'] == zone].copy()
                
                # Rolling calculation
                data['Trend'] = data['Median_Val'].rolling(window=window, min_periods=1).mean()

                # Plotting
                ax.scatter(data['Sample_ID'], data['Median_Val'], s=10, alpha=0.3, 
                           color=self.colors['raw'], label='Raw Medians')
                ax.plot(data['Sample_ID'], data['Trend'], color=self.colors['trend'], 
                        linewidth=2, label=f'{window}-Sample Trend')

                ax.set_title(f"Zone {zone}", fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.6)
                if i == 0: ax.legend(fontsize=8)

            # Cleanup empty subplots
            for j in range(num_zones, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.supxlabel('Sequential Sample ID', fontsize=12)
            fig.supylabel('Parameter Value (Median)', fontsize=12)
            fig.tight_layout()
            plt.show()

    def plot_combined_trends(self, final_report, target_zones=None, window=8):
        """
        Plots paired trends for pH and Temperature in a side-by-side grid.
        """
        if not {'Median_pH', 'Median_Temp'}.issubset(final_report.columns):
            print("Error: Report missing 'Median_pH' or 'Median_Temp' columns."); return

        zones = self._prepare_zones(final_report, target_zones)
        num_zones = len(zones)
        
        # Grid: Every zone gets 2 plots (pH & Temp) side-by-side
        zones_per_row = min(num_zones, 3)
        ncols = zones_per_row * 2
        nrows = math.ceil(num_zones / zones_per_row)

        with plt.style.context(self.style):
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, 
                figsize=(4.5 * ncols, 4 * nrows)
            )
            axes_flat = np.array(axes).flatten()

            for i, zone in enumerate(zones):
                data = final_report[final_report['Zone'] == zone].copy()
                
                # Setup axes for the pair
                ax_ph = axes_flat[i * 2]
                ax_temp = axes_flat[i * 2 + 1]

                # pH Plot
                ph_trend = data['Median_pH'].rolling(window=window, min_periods=1).mean()
                ax_ph.scatter(data['Sample_ID'], data['Median_pH'], s=8, alpha=0.3, color=self.colors['ph_raw'])
                ax_ph.plot(data['Sample_ID'], ph_trend, color=self.colors['ph_trend'], linewidth=1.8)
                ax_ph.set_title(f"Z{zone}: pH", fontsize=10, loc='left')
                ax_ph.set_ylabel("pH")

                # Temp Plot
                temp_trend = data['Median_Temp'].rolling(window=window, min_periods=1).mean()
                ax_temp.scatter(data['Sample_ID'], data['Median_Temp'], s=8, alpha=0.3, color=self.colors['raw'])
                ax_temp.plot(data['Sample_ID'], temp_trend, color=self.colors['trend'], linewidth=1.8)
                ax_temp.set_title(f"Z{zone}: Temp", fontsize=10, loc='left')
                ax_temp.set_ylabel("°C")

                for ax in [ax_ph, ax_temp]:
                    ax.grid(True, alpha=0.4, linestyle=':')
                    ax.tick_params(labelsize=9)

            # Cleanup and formatting
            for j in range(num_zones * 2, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.supxlabel('Time Progression (Sample IDs)', fontsize=13)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.show()