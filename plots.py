"""
Data Visualization Module

This module provides a suite of visualization tools to analyze the nanonetwork's
performance. It is divided into three primary components:
1. OptunaVisualizer: Generates interactive Plotly charts to analyze Bayesian optimization.
2. SimulationVisualizer: Generates Matplotlib grid plots to observe baseline physiological trends.
3. AnomalyVisualizer: Generates detailed Matplotlib subplots comparing raw vs. aggregated 
   Kalman filter tracking and highlights the exact moment of anomaly detection.
"""

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
        """Initializes the OptunaVisualizer."""
        pass

    def plot_1d_raw_tuning(self, study, title_suffix="Raw Filter"):
        """
        Visualizes a 1D tuning study (Single Parameter Raw Filter).
        Since it is 1D, it exclusively plots the history to illustrate cost reduction over time.
        
        @param study (optuna.study.Study): The completed Optuna study object.
        @param title_suffix (str): Contextual string appended to the plot title.
        """
        print(f"--- Visualization: {title_suffix} ---")
        history_fig = plot_optimization_history(study)
        history_fig.update_layout(title=f"Optimization History ({title_suffix})")
        history_fig.show()

    def plot_2d_agg_tuning(self, study, title_suffix="Aggregated Filter"):
        """
        Visualizes a 2D tuning study (Single Parameter Aggregated Filter).
        Displays cost history, parameter importance, and a 2D contour map of the search space.
        
        @param study (optuna.study.Study): The completed Optuna study object.
        @param title_suffix (str): Contextual string appended to the plot titles.
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
        Illustrates the interplay between the pH drift threshold and Temperature drift threshold.
        
        @param study (optuna.study.Study): The completed Optuna study object.
        @param title_suffix (str): Contextual string appended to the plot titles.
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
        Generates a 3x3 matrix contour plot comparing persistence vs. pH vs. Temp drift thresholds.
        
        @param study (optuna.study.Study): The completed Optuna study object.
        @param title_suffix (str): Contextual string appended to the plot titles.
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
    before any complex Kalman anomaly detection is applied.
    """
    
    def __init__(self, style="seaborn-v0_8-muted"):
        """
        Initializes the simulation visualizer with a specific Matplotlib style.
        
        @param style (str): The Matplotlib style string to format the plots.
        """
        self.style = style
        # Default color palettes for consistent plotting across different methods
        self.colors = {
            'raw': '#1f77b4',      # Muted Blue (Default raw data)
            'trend': '#d62728',    # Strong Red (Default trend line)
            'ph_raw': '#2ca02c',   # Green (pH specific raw data)
            'ph_trend': '#1a591a'  # Dark Green (pH specific trend line)
        }

    def _prepare_zones(self, final_report, target_zones):
        """
        Helper method to filter and sort available zones for plotting.
        If specific zones are requested, it validates them against the dataset.
        
        @param final_report (pd.DataFrame): The simulation dataset.
        @param target_zones (list): List of requested zone IDs, or None for all.
        @return list: A validated, sorted list of zone IDs to plot.
        """
        available = sorted(final_report['Zone'].dropna().unique())
        if target_zones:
            return [z for z in target_zones if z in available]
        return available

    def plot_zone_trends(self, final_report, target_zones=None, window=8):
        """
        Plots a grid of temporal trends for a single parameter (e.g., pH or Temp).
        Calculates and overlays a moving average to smooth out high-frequency noise.
        
        @param final_report (pd.DataFrame): The output dataset from the simulator.
        @param target_zones (list, optional): Specific zone IDs to plot. Plots all if None.
        @param window (int): The number of samples to include in the rolling average.
        """
        zones = self._prepare_zones(final_report, target_zones)
        num_zones = len(zones)
        if num_zones == 0:
            print("No valid zones found for plotting.")
            return

        # Calculate grid dimensions dynamically (Max 4 columns wide)
        ncols = min(num_zones, 4)
        nrows = math.ceil(num_zones / ncols)

        with plt.style.context(self.style):
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, 
                figsize=(5 * ncols, 3.8 * nrows), 
                sharex=True # Share the X-axis across all subplots to align timeframes
            )
            # Flatten axes array to easily iterate over it, even if it is 2D
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
                if i == 0: 
                    ax.legend(fontsize=8)

            # Cleanup: Hide any empty subplots if the number of zones does not perfectly fill the grid
            for j in range(num_zones, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.supxlabel('Sequential Sample ID', fontsize=12)
            fig.supylabel('Parameter Value (Median)', fontsize=12)
            fig.tight_layout()
            plt.show()

    def plot_combined_trends(self, final_report, target_zones=None, window=8):
        """
        Plots paired trends for pH and Temperature in a side-by-side grid.
        Every requested zone is allocated two adjacent plots.
        
        @param final_report (pd.DataFrame): The combined multimodal dataset.
        @param target_zones (list, optional): Specific zone IDs to plot.
        @param window (int): The number of samples to include in the rolling average.
        """
        if not {'Median_pH', 'Median_Temp'}.issubset(final_report.columns):
            print("Error: Report missing 'Median_pH' or 'Median_Temp' columns.")
            return

        zones = self._prepare_zones(final_report, target_zones)
        num_zones = len(zones)
        
        # Grid logic: Every zone takes up 2 columns (one for pH, one for Temp)
        # Max 3 zones per row (which results in 6 columns total)
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
    Handles advanced plotting for the AnomalyDetector pipeline.
    Visualizes raw noisy data, Kalman filter state lines, and precisely highlights 
    the temporal moment the system triggers a diagnostic anomaly.
    """
    
    def __init__(self, style="seaborn-v0_8-muted"):
        """
        Initializes the anomaly visualizer.
        
        @param style (str): The Matplotlib style string to format the plots.
        """
        self.style = style

    def plot_comparison(self, raw_df, agg_df, det_raw, det_agg, target_zones, title="Detection Time Comparison"):
        """
        Creates a dynamic grid to compare Raw vs. Aggregated detection performance.
        Automatically isolates and plots every targeted diseased zone, plus any healthy 
        zone that incorrectly triggered a False Positive.
        
        @param raw_df (pd.DataFrame): The processed raw data stream containing Kalman states.
        @param agg_df (pd.DataFrame): The processed aggregated data stream containing Kalman states.
        @param det_raw (dict): Dictionary mapping zones to their raw detection timestamps.
        @param det_agg (dict): Dictionary mapping zones to their aggregated detection timestamps.
        @param target_zones (list): List of ground-truth diseased zones.
        @param title (str): Title for the entire figure grid.
        """
        # Create copies and convert time to hours for clean X-axis scaling
        r_df = raw_df.copy()
        a_df = agg_df.copy()
        r_df['Time_Hours'] = r_df['Time_Seconds'] / 3600
        a_df['Time_Hours'] = a_df['Time_Seconds_End'] / 3600

        # Compile unique zones to plot (Targets + Any False Alarms)
        zones_to_plot = sorted(list(set(target_zones) | set(det_raw.keys()) | set(det_agg.keys())))
        nrows = len(zones_to_plot)

        with plt.style.context(self.style):
            # 2 Columns: Col 0 = Raw Stream, Col 1 = Aggregated Stream
            # squeeze=False ensures axes is always a 2D array even if nrows=1
            fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows), sharex=True, squeeze=False)

            for i, zone in enumerate(zones_to_plot):
                is_target = zone in target_zones
                zone_label = f"Zone {zone} (TARGET)" if is_target else f"Zone {zone} (FALSE ALARM)"
                
                # --- LEFT COLUMN: RAW STREAM ---
                ax_raw = axes[i, 0]
                z_raw = r_df[r_df['Zone'] == zone]
                
                ax_raw.scatter(z_raw['Time_Hours'], z_raw['Value'], alpha=0.2, color='gray', s=10, label='Raw Readings')
                ax_raw.plot(z_raw['Time_Hours'], z_raw['Filtered_Value'], color='#1f77b4', linewidth=2, label='Kalman State')
                
                if zone in det_raw:
                    t_det = det_raw[zone] / 3600
                    ax_raw.axvline(x=t_det, color='#d62728', linestyle='--', linewidth=2, label=f'Alert: {t_det:.2f}h')
                
                ax_raw.set_title(f"RAW - {zone_label}", fontweight='bold', color='black' if is_target else '#d62728')
                ax_raw.grid(True, linestyle='--', alpha=0.6)
                ax_raw.legend(loc='upper left')

                # --- RIGHT COLUMN: AGGREGATED STREAM ---
                ax_agg = axes[i, 1]
                z_agg = a_df[a_df['Zone'] == zone]
                
                ax_agg.scatter(z_agg['Time_Hours'], z_agg['Median_Val'], alpha=0.8, color='#ff7f0e', marker='s', label='Agg Median')
                ax_agg.plot(z_agg['Time_Hours'], z_agg['Filtered_Value'], color='#2ca02c', linewidth=2, label='Adaptive State')
                
                if zone in det_agg:
                    t_det = det_agg[zone] / 3600
                    ax_agg.axvline(x=t_det, color='#d62728', linestyle='--', linewidth=2, label=f'Alert: {t_det:.2f}h')
                
                ax_agg.set_title(f"AGGREGATED - {zone_label}", fontweight='bold', color='black' if is_target else '#d62728')
                ax_agg.grid(True, linestyle='--', alpha=0.6)
                ax_agg.legend(loc='upper left')

            fig.supxlabel('Time Progression (Hours)', fontsize=14, fontweight='bold')
            fig.supylabel('Sensor Value', fontsize=14, fontweight='bold')
            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
    def plot_combined_comparison(self, raw_df, agg_df, det_raw, det_agg, target_zones, title="Combined Detection Comparison"):
        """
        Creates a dynamic 4-column grid for dual-parameter diseases.
        Visualizes the independent Kalman tracking for both pH and Temperature across both data streams.
        Columns: [Raw pH] | [Raw Temp] | [Agg pH] | [Agg Temp]
        
        @param raw_df (pd.DataFrame): The processed raw multimodal data stream.
        @param agg_df (pd.DataFrame): The processed aggregated multimodal data stream.
        @param det_raw (dict): Dictionary mapping zones to their multimodal raw detection timestamps.
        @param det_agg (dict): Dictionary mapping zones to their multimodal aggregated detection timestamps.
        @param target_zones (list): List of ground-truth diseased zones.
        @param title (str): Title for the entire figure grid.
        """
        r_df = raw_df.copy()
        a_df = agg_df.copy()
        r_df['Time_Hours'] = r_df['Time_Seconds'] / 3600
        a_df['Time_Hours'] = a_df['Time_Seconds_End'] / 3600

        zones_to_plot = sorted(list(set(target_zones) | set(det_raw.keys()) | set(det_agg.keys())))
        nrows = len(zones_to_plot)

        with plt.style.context(self.style):
            # 4 Columns for combined plots, squeeze=False ensures 2D indexing
            fig, axes = plt.subplots(nrows, 4, figsize=(20, 4 * nrows), sharex=True, squeeze=False)

            for i, zone in enumerate(zones_to_plot):
                is_target = zone in target_zones
                z_label = f"Z{zone} (TARGET)" if is_target else f"Z{zone} (FALSE ALARM)"
                
                z_raw = r_df[r_df['Zone'] == zone]
                z_agg = a_df[a_df['Zone'] == zone]
                
                # --- Column 0: Raw pH ---
                ax = axes[i, 0]
                ax.scatter(z_raw['Time_Hours'], z_raw['Value_pH'], alpha=0.2, color='gray', s=10)
                ax.plot(z_raw['Time_Hours'], z_raw['Filtered_pH'], color='#1f77b4', linewidth=2)
                if zone in det_raw and det_raw[zone].get('ph'):
                    ax.axvline(x=det_raw[zone]['ph']/3600, color='#d62728', linestyle='--')
                ax.set_title(f"RAW pH - {z_label}", fontweight='bold', color='black' if is_target else '#d62728')
                ax.grid(True, linestyle='--', alpha=0.6)

                # --- Column 1: Raw Temperature ---
                ax = axes[i, 1]
                ax.scatter(z_raw['Time_Hours'], z_raw['Value_Temp'], alpha=0.2, color='gray', s=10)
                ax.plot(z_raw['Time_Hours'], z_raw['Filtered_Temp'], color='#9467bd', linewidth=2)
                if zone in det_raw and det_raw[zone].get('temp'):
                    ax.axvline(x=det_raw[zone]['temp']/3600, color='#d62728', linestyle='--')
                ax.set_title(f"RAW Temp - {z_label}", fontweight='bold', color='black' if is_target else '#d62728')
                ax.grid(True, linestyle='--', alpha=0.6)

                # --- Column 2: Aggregated pH ---
                ax = axes[i, 2]
                ax.scatter(z_agg['Time_Hours'], z_agg['Median_pH'], alpha=0.8, color='#ff7f0e', marker='s')
                ax.plot(z_agg['Time_Hours'], z_agg['Filtered_pH'], color='#2ca02c', linewidth=2)
                if zone in det_agg and det_agg[zone].get('ph'):
                    ax.axvline(x=det_agg[zone]['ph']/3600, color='#d62728', linestyle='--')
                ax.set_title(f"AGG pH - {z_label}", fontweight='bold', color='black' if is_target else '#d62728')
                ax.grid(True, linestyle='--', alpha=0.6)

                # --- Column 3: Aggregated Temperature ---
                ax = axes[i, 3]
                ax.scatter(z_agg['Time_Hours'], z_agg['Median_Temp'], alpha=0.8, color='#ff7f0e', marker='s')
                ax.plot(z_agg['Time_Hours'], z_agg['Filtered_Temp'], color='#8c564b', linewidth=2)
                if zone in det_agg and det_agg[zone].get('temp'):
                    ax.axvline(x=det_agg[zone]['temp']/3600, color='#d62728', linestyle='--')
                ax.set_title(f"AGG Temp - {z_label}", fontweight='bold', color='black' if is_target else '#d62728')
                ax.grid(True, linestyle='--', alpha=0.6)

            fig.supxlabel('Time Progression (Hours)', fontsize=14, fontweight='bold')
            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()