import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches

def generate_gymnast_style_radar():
    print("Generating Gymnast-Style Stylized Radar Chart...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Define models
    models = {
        "Baseline": os.path.join(root_dir, "Baseline", "simulation_results_base_2.xlsx"),
        "Baseline with Penalty": os.path.join(root_dir, "Baseline", "simulation_results_base_penalty_2.xlsx"),
        "Dijkstra Pure": os.path.join(root_dir, "Dijkstra_Pure", "simulation_results_dijkstra_2.xlsx"),
        "ReverseDijkstra": os.path.join(root_dir, "Dijkstra_SafeZones", "simulation_results_dijkstra_safe_2.xlsx")
    }
    
    extracted_data = {}
    
    for model_name, path in models.items():
        if not os.path.exists(path):
            print(f"[{model_name}] WARNING: File not found ({path})")
            continue
            
        try:
            df = pd.read_excel(path, sheet_name='Run Comparisons')
            rename_map = {
                'Avg Density (p/m2)': 'Avg Target Density (p/m2)',
                'Average Density': 'Avg Target Density (p/m2)',
                'Peak Density': 'Peak Density (p/m2)',
                'Average Velocity': 'Avg Velocity (m/s)'
            }
            df.rename(columns=rename_map, inplace=True)
            
            df_runs = df[pd.to_numeric(df['Run'], errors='coerce').notnull()].copy()
            for col in df_runs.columns:
                df_runs[col] = pd.to_numeric(df_runs[col], errors='coerce')
            df_runs = df_runs.dropna(subset=['Total Evacuation Time (s)'])
            
            if len(df_runs) == 0:
                continue
                
            avg_vals = df_runs.mean(numeric_only=True)
            extracted_data[model_name] = avg_vals
            print(f"[{model_name}] Loaded {len(df_runs)} iterations.")
        except Exception as e:
            print(f"[{model_name}] Error: {e}")
            
    if not extracted_data:
        print("No valid data loaded.")
        return
        
    df_compare = pd.DataFrame(extracted_data).T

    # --- FIXED THEORETICAL METRIC RANGES ---
    # Why fixed ranges instead of dynamic Min-Max:
    # When using Min-Max normalization across models, if all models score similarly on a metric,
    # the lowest scorer collapses to 0 and the highest collapses to 10 — even if the actual
    # difference is tiny. This creates a highly misleading radar shape that exaggerates contrast.
    # With fixed ranges, each metric is always normalized against its physically meaningful scale,
    # so the radar chart accurately represents how each model performs in an absolute sense.
    metric_ranges = {
        'Total Evacuation Time (s)': (0, 1200),
        'Total Casualties':          (0, 1000),
        'Total Evacuated':           (0, 10000),
        'Avg Target Density (p/m2)': (0, 5),
        'Peak Density (p/m2)':       (0, 5),
        'Avg Exit Flow Rate':        (0, 200),
        'Avg Velocity (m/s)':        (0, 2),
        'Remaining Agents':          (0, 5000)
    }
    
    # These are the metrics where LOWER values represent BETTER performance.
    # Why inversion is necessary:
    # On a radar chart, a bigger polygon = better model.
    # Without inverting "lower is better" metrics like Casualties, a model with 0 deaths
    # would score 0 and visually shrink inward — the exact opposite of the intended representation.
    invert_metrics = {
        'Total Evacuation Time (s)',
        'Total Casualties',
        'Remaining Agents',
        'Avg Target Density (p/m2)',
        'Peak Density (p/m2)'
    }

    potential_metrics = [
        'Total Evacuation Time (s)',
        'Total Casualties', 
        'Total Evacuated',
        'Avg Target Density (p/m2)', 
        'Peak Density (p/m2)',
        'Avg Exit Flow Rate',
        'Avg Velocity (m/s)',
        'Remaining Agents'
    ]
    
    # Only include metrics that exist in the loaded data AND have defined fixed ranges
    actual_metrics = [m for m in potential_metrics if m in df_compare.columns and m in metric_ranges]
    if len(actual_metrics) < 3:
        print("Not enough shared metrics. Available:", list(df_compare.columns))
        return
    
    def normalize(row):
        """
        Normalize metric values to 0-10 using fixed theoretical ranges.
        Inverted metrics flip the scale so that lower raw values produce higher scores.
        Values are clamped to [0, 10] to avoid overflow if a model exceeds the expected range.
        """
        norm_row = {}
        for metric in actual_metrics:
            val = row[metric]
            lo, hi = metric_ranges[metric]
            span = hi - lo
            
            if span == 0:
                norm_row[metric] = 10.0
                continue
            
            if metric in invert_metrics:
                # Inverted: best possible (lo) → 10, worst possible (hi) → 0
                norm_val = ((hi - val) / span) * 10.0
            else:
                # Standard: best possible (hi) → 10, worst possible (lo) → 0
                norm_val = ((val - lo) / span) * 10.0
            
            # Clamp: if a model exceeds the defined theoretical range, cap at 0 or 10
            norm_row[metric] = max(0.0, min(10.0, norm_val))
        return norm_row

    # Colors defined to match the Gymnast Style closely
    colors = {
        "Baseline": "#f25c54",               # Stylized Red
        "Baseline with Penalty": "#f4a261",  # Stylized Orange
        "Dijkstra Pure": "#3a86ff",          # Stylized Blue
        "ReverseDijkstra": "#60d394"         # Stylized Green
    }
    
    num_vars = len(actual_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Initialize polar plot
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_axes([0.3, 0.1, 0.65, 0.75], polar=True)
    
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)  # Clockwise
    
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('#cccccc')
    ax.spines['polar'].set_linewidth(1.2)
    ax.grid(color='#e0e0e0', linestyle='-', linewidth=1.2)
    
    ax.set_xticks(angles[:-1])
    short_labels = [m.replace(' (', '\n(') for m in actual_metrics]
    ax.set_xticklabels(short_labels, size=11, family='serif', color='black')
    ax.tick_params(pad=15)
    
    ax.set_rlabel_position(0)
    ax.set_rticks([2, 4, 6, 8, 10])
    ax.set_yticklabels([], color="black", size=10, family='serif')
    plt.ylim(0, 10)
    
    # Plot each model as a transparent polygon with vertex dots
    for model_name, row in df_compare.iterrows():
        norm_dict = normalize(row[actual_metrics])
        norm_vals = [norm_dict[m] for m in actual_metrics]
        norm_vals += norm_vals[:1]  # Close polygon
        
        col = colors.get(model_name, "#333333")
        
        # The thicker line at 2.5 ensures the polygon outline is clearly distinguishable
        ax.plot(angles, norm_vals, linewidth=2.5, linestyle='solid', color=col)
        # Vertex dots at size 6 mark where the exact normalized score falls on each axis
        ax.plot(angles, norm_vals, marker='o', markersize=6, color=col, linestyle='None')
        # Alpha of 0.35 gives strong translucent fill while still showing overlap between models
        ax.fill(angles, norm_vals, col, alpha=0.35)
        
    # Top-left title and legend, matching the Gymnast Scoring chart layout exactly
    fig.text(0.02, 0.95, "Simulation Metrics Radar Chart", 
             fontsize=18, fontweight='bold', family='serif', ha='left', va='top')
             
    patches = [
        mpatches.Patch(color=colors.get(m, 'blue'), label=m) 
        for m in df_compare.index
    ]
    
    fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(0.02, 0.90), 
               fontsize=12, frameon=False, borderaxespad=0., prop={'family': 'serif', 'size': 12})
    
    output_img = os.path.join(script_dir, "the_gymnast_radar_comparison.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {output_img}")
    
    plt.show(block=False)

if __name__ == "__main__":
    generate_gymnast_style_radar()
