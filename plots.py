import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour

# ==========================================
# OPTUNA TUNING VISUALIZER
# ==========================================

class OptunaVisualizer:
    """
    Handles the generation of interactive Plotly graphs for Optuna hyperparameter tuning.
    These plots analyze the search space, cost history, and parameter importance.
    """
    def __init__(self):
        pass

    def plot_1d_raw_tuning(self, study, title_suffix="Raw Filter"):
        """
        Visualizes a 1D tuning study (Single Parameter Raw Filter).
        Since it's 1D, we only need the history to see how the cost dropped over time.
        """
        print(f"--- Visualization: {title_suffix} ---")
        history_fig = plot_optimization_history(study)
        history_fig.update_layout(title=f"Optimization History ({title_suffix})")
        history_fig.show()

    def plot_2d_agg_tuning(self, study, title_suffix="Aggregated Filter"):
        """
        Visualizes a 2D tuning study (Single Parameter Aggregated Filter).
        Shows cost history, parameter importance, and a 2D contour map of the search space.
        """
        print(f"--- Visualization: {title_suffix} ---")
        
        history_fig = plot_optimization_history(study)
        history_fig.update_layout(title=f"Optimization History ({title_suffix})")
        history_fig.show()

        importance_fig = plot_param_importances(study)
        importance_fig.update_layout(title=f"Hyperparameter Importance ({title_suffix})")
        importance_fig.show()

        contour_fig = plot_contour(study, params=['agg_persistence', 'agg_abs_drift'])
        contour_fig.update_layout(title=f"2D Search Space Contour ({title_suffix})")
        contour_fig.show()

    def plot_combined_raw_tuning(self, study, title_suffix="Combined Raw Filter"):
        """
        Visualizes a 2D tuning study (Combined Parameter Raw Filter).
        Shows the interplay between the pH drift threshold and Temp drift threshold.
        """
        print(f"--- Visualization: {title_suffix} ---")
        
        history_fig = plot_optimization_history(study)
        history_fig.update_layout(title=f"Optimization History ({title_suffix})")
        history_fig.show()

        importance_fig = plot_param_importances(study)
        importance_fig.update_layout(title=f"Hyperparameter Importance ({title_suffix})")
        importance_fig.show()

        contour_fig = plot_contour(study, params=['raw_drift_ph', 'raw_drift_temp'])
        contour_fig.update_layout(title=f"2D Search Space Contour ({title_suffix})")
        contour_fig.show()

    def plot_combined_agg_tuning(self, study, title_suffix="Combined Aggregated Filter"):
        """
        Visualizes a 3D tuning study (Combined Parameter Aggregated Filter).
        Generates a 3x3 matrix contour plot comparing persistence vs pH vs Temp drift.
        """
        print(f"--- Visualization: {title_suffix} ---")
        
        history_fig = plot_optimization_history(study)
        history_fig.update_layout(title=f"Optimization History ({title_suffix})")
        history_fig.show()

        importance_fig = plot_param_importances(study)
        importance_fig.update_layout(title=f"Hyperparameter Importance ({title_suffix})")
        importance_fig.show()

        contour_fig = plot_contour(study, params=['agg_persistence', 'agg_abs_drift_ph', 'agg_abs_drift_temp'])
        contour_fig.update_layout(title=f"3D Search Space Matrix ({title_suffix})", height=800, width=800)
        contour_fig.show()

# ==========================================
# BASELINE SIMULATION VISUALIZER
# ==========================================

class SimulationVisualizer:
    """
    Handles data visualization for general Nanobot simulation results.
    Useful for visualizing raw baseline data and simple moving averages 
    before any complex anomaly detection is applied.
    """
    
    def __init__(self, style="seaborn-v0_8-muted"):
        self.style = style
        # Default color palettes for consistent plotting across different methods
        self.colors = {
            'raw': '#1f77b4',     # Muted Blue (Default raw data)
            'trend': '#d62728',   # Strong Red (Default trend line)
            'ph_raw': '#2ca02c',  # Green (pH specific raw data)
            'ph_trend': '#1a591a' # Dark Green (pH specific trend line)
        }

    def _prepare_zones(self, final_report, target_zones):
        """
        Helper method to filter and sort available zones for plotting.
        If specific zones are requested, it validates them against the dataset.
        """
        available = sorted(final_report['Zone'].dropna().unique())
        if target_zones:
            return [z for z in target_zones if z in available]
        return available

    def plot_zone_trends(self, final_report, target_zones=None, window=8):
        """
        Plots a grid of temporal trends for a single parameter (e.g., Pressure or SO2).
        Calculates and overlays a moving average to smooth out the noise.
        
        Args:
            final_report (DataFrame): The output dataset from the simulator.
            target_zones (list, optional): Specific zone IDs to plot.
            window (int): The number of samples to include in the rolling average.
        """
        zones = self._prepare_zones(final_report, target_zones)
        num_zones = len(zones)
        if num_zones == 0:
            print("No valid zones found for plotting."); return

        # Calculate grid dimensions dynamically (Max 4 columns wide)
        ncols = min(num_zones, 4)
        nrows = math.ceil(num_zones / ncols)

        with plt.style.context(self.style):
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, 
                figsize=(5 * ncols, 3.8 * nrows), 
                sharex=True # Share the X-axis across all subplots to align timeframes
            )
            # Flatten axes array to easily iterate over it, even if it's 2D
            axes_flat = np.array(axes).flatten()

            for i, zone in enumerate(zones):
                ax = axes_flat[i]
                data = final_report[final_report['Zone'] == zone].copy()
                
                # Calculate the rolling average trendline
                data['Trend'] = data['Median_Val'].rolling(window=window, min_periods=1).mean()

                # Plot raw medians as faint scattered dots
                ax.scatter(data['Sample_ID'], data['Median_Val'], s=10, alpha=0.3, 
                           color=self.colors['raw'], label='Raw Medians')
                # Plot the smoothed trendline over top
                ax.plot(data['Sample_ID'], data['Trend'], color=self.colors['trend'], 
                        linewidth=2, label=f'{window}-Sample Trend')

                ax.set_title(f"Zone {zone}", fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.6)
                if i == 0: ax.legend(fontsize=8)

            # Cleanup: Hide any empty subplots if the number of zones doesn't perfectly fill the grid
            for j in range(num_zones, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.supxlabel('Sequential Sample ID', fontsize=12)
            fig.supylabel('Parameter Value (Median)', fontsize=12)
            fig.tight_layout()
            plt.show()

    def plot_combined_trends(self, final_report, target_zones=None, window=8):
        """
        Plots paired trends for pH and Temperature in a side-by-side grid.
        Every requested zone gets two adjacent plots.
        """
        if not {'Median_pH', 'Median_Temp'}.issubset(final_report.columns):
            print("Error: Report missing 'Median_pH' or 'Median_Temp' columns."); return

        zones = self._prepare_zones(final_report, target_zones)
        num_zones = len(zones)
        
        # Grid logic: Every zone takes up 2 columns (one for pH, one for Temp)
        # Max 3 zones per row (which means 6 columns total)
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
                
                # Grab the two adjacent axes for this specific zone
                ax_ph = axes_flat[i * 2]
                ax_temp = axes_flat[i * 2 + 1]

                # Setup pH Plot
                ph_trend = data['Median_pH'].rolling(window=window, min_periods=1).mean()
                ax_ph.scatter(data['Sample_ID'], data['Median_pH'], s=8, alpha=0.3, color=self.colors['ph_raw'])
                ax_ph.plot(data['Sample_ID'], ph_trend, color=self.colors['ph_trend'], linewidth=1.8)
                ax_ph.set_title(f"Z{zone}: pH", fontsize=10, loc='left')
                ax_ph.set_ylabel("pH")

                # Setup Temp Plot
                temp_trend = data['Median_Temp'].rolling(window=window, min_periods=1).mean()
                ax_temp.scatter(data['Sample_ID'], data['Median_Temp'], s=8, alpha=0.3, color=self.colors['raw'])
                ax_temp.plot(data['Sample_ID'], temp_trend, color=self.colors['trend'], linewidth=1.8)
                ax_temp.set_title(f"Z{zone}: Temp", fontsize=10, loc='left')
                ax_temp.set_ylabel("°C")

                for ax in [ax_ph, ax_temp]:
                    ax.grid(True, alpha=0.4, linestyle=':')
                    ax.tick_params(labelsize=9)

            # Cleanup empty subplots
            for j in range(num_zones * 2, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.supxlabel('Time Progression (Sample IDs)', fontsize=13)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.show()

# ==========================================
# ANOMALY DETECTION VISUALIZER
# ==========================================

class AnomalyVisualizer:
    """
    Handles plotting for the AnomalyDetector pipeline.
    Visualizes raw noisy data, Kalman filter state lines, and precisely
    highlights the moment the system flags a drift/anomaly.
    """
    
    def __init__(self, style="seaborn-v0_8-muted"):
        self.style = style

    def plot_single_raw(self, df, detection_time, title="Raw Stream Anomaly Detection"):
        """Plots a single panel showing raw pings vs the CUSUM Kalman state."""
        with plt.style.context(self.style):
            plt.figure(figsize=(10, 5))
            
            # Plot the highly noisy raw data points faintly in the background
            plt.scatter(df['Time_Seconds'], df['Value'], alpha=0.3, color='gray', s=10, label='Raw Readings')
            # Plot the smoothed Kalman filter state
            plt.plot(df['Time_Seconds'], df['Filtered_Value'], color='#1f77b4', linewidth=2, label='Kalman State')
            
            # If an anomaly was detected, draw a red vertical line at that exact second
            if detection_time:
                plt.axvline(x=detection_time, color='#d62728', linestyle='--', linewidth=2, 
                            label=f'Anomaly Detected: Hour {detection_time/3600:.2f}')
            
            plt.title(title, fontweight='bold')
            plt.xlabel('Time (Seconds)', fontsize=12)
            plt.ylabel('Sensor Value', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

    def plot_single_aggregated(self, df, detection_time, title="Aggregated Stream Anomaly Detection"):
        """Plots a single panel showing aggregated packets vs the Adaptive Kalman state."""
        with plt.style.context(self.style):
            plt.figure(figsize=(10, 5))
            
            # Plot aggregated medians as squares
            plt.scatter(df['Time_Seconds_End'], df['Median_Val'], alpha=0.7, color='#ff7f0e', marker='s', label='Aggregated Median')
            # Plot the adaptive Kalman state tracking the medians
            plt.plot(df['Time_Seconds_End'], df['Filtered_Value'], color='#2ca02c', linewidth=2, label='Adaptive Kalman State')
            
            if detection_time:
                plt.axvline(x=detection_time, color='#d62728', linestyle='--', linewidth=2, 
                            label=f'Anomaly Detected: Hour {detection_time/3600:.2f}')
            
            plt.title(title, fontweight='bold')
            plt.xlabel('Time (Seconds)', fontsize=12)
            plt.ylabel('Aggregated Median Value', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

    def plot_comparison(self, raw_df, agg_df, det_raw, det_agg, title="Detection Time Comparison"):
        """
        Creates a 2x1 stacked plot to compare the speed and stability of 
        Raw stream detection vs. Aggregated stream detection.
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # --- TOP PANEL: RAW STREAM ---
            ax[0].scatter(raw_df['Time_Seconds'], raw_df['Value'], alpha=0.2, color='gray', s=10, label='Raw Readings')
            ax[0].plot(raw_df['Time_Seconds'], raw_df['Filtered_Value'], color='#1f77b4', linewidth=2, label='Kalman State')
            if det_raw:
                ax[0].axvline(x=det_raw, color='#d62728', linestyle='--', linewidth=2, label=f'Raw Detection: Hour {det_raw/3600:.2f}')
            ax[0].set_title("Raw Stream Detection", fontweight='bold')
            ax[0].legend(loc='upper left')
            ax[0].grid(True, linestyle='--', alpha=0.6)

            # --- BOTTOM PANEL: AGGREGATED STREAM ---
            ax[1].scatter(agg_df['Time_Seconds_End'], agg_df['Median_Val'], alpha=0.8, color='#ff7f0e', marker='s', label='Agg Median')
            ax[1].plot(agg_df['Time_Seconds_End'], agg_df['Filtered_Value'], color='#2ca02c', linewidth=2, label='Adaptive State')
            if det_agg:
                ax[1].axvline(x=det_agg, color='#d62728', linestyle='--', linewidth=2, label=f'Agg Detection: Hour {det_agg/3600:.2f}')
            ax[1].set_title("Aggregated Stream Detection", fontweight='bold')
            ax[1].legend(loc='upper left')
            ax[1].grid(True, linestyle='--', alpha=0.6)

            # Format the X-axis to display Hours instead of seconds
            ticks = ax[1].get_xticks()
            ax[1].set_xticks(ticks) # Lock the tick positions
            ax[1].set_xticklabels([f"{t/3600:.1f}h" for t in ticks]) # Apply formatted labels

            fig.supxlabel('Time Progression (Hours)', fontsize=12)
            fig.supylabel('Sensor Value', fontsize=12)
            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
    def plot_combined_comparison(self, raw_df, agg_df, det_raw, det_agg, title="Combined Detection Comparison"):
        """
        Creates a 2x2 grid to compare Raw vs Aggregated detection for a dual-parameter 
        disease (pH and Temperature simultaneously).
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

            # ========================================
            # ROW 0: RAW STREAM
            # ========================================
            
            # Top-Left: Raw pH
            ax[0, 0].scatter(raw_df['Time_Seconds'], raw_df['Value_pH'], alpha=0.2, color='gray', s=10)
            ax[0, 0].plot(raw_df['Time_Seconds'], raw_df['Filtered_pH'], color='#1f77b4', linewidth=2)
            if det_raw['ph']: ax[0, 0].axvline(x=det_raw['ph'], color='#d62728', linestyle='--')
            ax[0, 0].set_title("Raw Stream: pH", fontweight='bold')
            ax[0, 0].grid(True, linestyle='--', alpha=0.6)

            # Top-Right: Raw Temperature
            ax[0, 1].scatter(raw_df['Time_Seconds'], raw_df['Value_Temp'], alpha=0.2, color='gray', s=10)
            ax[0, 1].plot(raw_df['Time_Seconds'], raw_df['Filtered_Temp'], color='#9467bd', linewidth=2)
            if det_raw['temp']: ax[0, 1].axvline(x=det_raw['temp'], color='#d62728', linestyle='--')
            ax[0, 1].set_title("Raw Stream: Temperature", fontweight='bold')
            ax[0, 1].grid(True, linestyle='--', alpha=0.6)

            # ========================================
            # ROW 1: AGGREGATED STREAM
            # ========================================
            
            # Bottom-Left: Aggregated pH
            ax[1, 0].scatter(agg_df['Time_Seconds_End'], agg_df['Median_pH'], alpha=0.8, color='#ff7f0e', marker='s')
            ax[1, 0].plot(agg_df['Time_Seconds_End'], agg_df['Filtered_pH'], color='#2ca02c', linewidth=2)
            if det_agg['ph']: ax[1, 0].axvline(x=det_agg['ph'], color='#d62728', linestyle='--')
            ax[1, 0].set_title("Aggregated Stream: pH", fontweight='bold')
            ax[1, 0].grid(True, linestyle='--', alpha=0.6)

            # Bottom-Right: Aggregated Temperature
            ax[1, 1].scatter(agg_df['Time_Seconds_End'], agg_df['Median_Temp'], alpha=0.8, color='#ff7f0e', marker='s')
            ax[1, 1].plot(agg_df['Time_Seconds_End'], agg_df['Filtered_Temp'], color='#8c564b', linewidth=2)
            if det_agg['temp']: ax[1, 1].axvline(x=det_agg['temp'], color='#d62728', linestyle='--')
            ax[1, 1].set_title("Aggregated Stream: Temperature", fontweight='bold')
            ax[1, 1].grid(True, linestyle='--', alpha=0.6)

            # Format the X-axis for the bottom row to display Hours
            ticks = ax[1, 0].get_xticks()
            for col in range(2):
                ax[1, col].set_xticks(ticks) # Lock positions to prevent warnings
                ax[1, col].set_xticklabels([f"{t/3600:.1f}h" for t in ticks]) # Set Hour labels

            fig.supxlabel('Time Progression (Hours)', fontsize=12)
            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()