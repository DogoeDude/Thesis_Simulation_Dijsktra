import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_raw_bar_chart():
    print("Generating Raw Metric Grouped Bar Chart...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
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
            extracted_data[model_name] = df_runs.mean(numeric_only=True)
            print(f"[{model_name}] Avg Casualties: {df_runs['Total Casualties'].mean():.1f}")
        except Exception as e:
            print(f"[{model_name}] Error: {e}")
    
    if not extracted_data:
        print("No valid data loaded.")
        return
    
    df_compare = pd.DataFrame(extracted_data).T

    # Metrics: (column_name, short_label, unit, lower_is_better)
    metrics_config = [
        ('Total Evacuation Time (s)', 'Evacuation\nTime',   'seconds',      True),
        ('Total Casualties',           'Casualties',         'agents',        True),
        ('Total Evacuated',            'Evacuated',          'agents',        False),
        ('Avg Target Density (p/m2)',  'Avg Density',        'p/m²',          True),
        ('Peak Density (p/m2)',        'Peak Density',       'p/m²',          True),
        ('Avg Exit Flow Rate',         'Exit Flow Rate',     'agents/step',   False),
        ('Avg Velocity (m/s)',         'Avg Velocity',       'm/s',           False),
    ]
    
    actual_configs = [(m, lbl, unit, inv) for (m, lbl, unit, inv) in metrics_config if m in df_compare.columns]
    
    colors = {
        "Baseline": "#f25c54",
        "Baseline with Penalty": "#f4a261",
        "Dijkstra Pure": "#3a86ff",
        "ReverseDijkstra": "#60d394"
    }
    
    model_names = list(extracted_data.keys())
    n_metrics = len(actual_configs)
    n_models = len(model_names)
    
    cols = 4
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), facecolor='white')
    axes = axes.flatten()
    
    bar_positions = np.arange(n_models)
    bar_width = 0.6

    for i, (metric, label, unit, lower_is_better) in enumerate(actual_configs):
        ax = axes[i]
        
        raw_vals = [df_compare.loc[m, metric] if m in df_compare.index else 0 for m in model_names]
        bar_colors = [colors.get(m, "#888888") for m in model_names]
        
        # Determine which model wins this metric
        if lower_is_better:
            best_idx = int(np.argmin(raw_vals))
            direction_label = "↓ Lower is Better"
        else:
            best_idx = int(np.argmax(raw_vals))
            direction_label = "↑ Higher is Better"

        bars = ax.bar(bar_positions, raw_vals, width=bar_width,
                      color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.7)

        # Compute rank order (1st best → lowest rank index)
        sorted_indices = np.argsort(raw_vals) if lower_is_better else np.argsort(raw_vals)[::-1]
        rank_of = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        rank_labels = ['1st', '2nd', '3rd', '4th']
        rank_colors = ['#DAA520', '#888888', '#cd7f32', '#999999']  # gold, silver, bronze, grey

        # Highlight best bar with a gold border
        best_idx = int(sorted_indices[0])
        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(3.5)

        # Value labels and rank badges above every bar
        y_max = max(raw_vals) if max(raw_vals) > 0 else 1
        for j, (bar, val) in enumerate(zip(bars, raw_vals)):
            rank = rank_of[j]
            rc = rank_colors[rank - 1]
            # Numeric value just above the bar
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.01,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold' if rank == 1 else 'normal',
                    color=rc)
            # Rank badge above the value
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.09,
                    rank_labels[rank - 1],
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color=rc)


        # Subtitle showing direction
        ax.set_title(f"{label}\n({unit})", fontsize=11, fontweight='bold', family='serif')
        ax.text(0.5, -0.16, direction_label,
                transform=ax.transAxes, ha='center', fontsize=8.5,
                color='#c0392b' if lower_is_better else '#27ae60',
                fontstyle='italic')

        ax.set_xticks(bar_positions)
        ax.set_xticklabels([m.replace(' with ', '\nwith ').replace('Dijkstra Pure', 'Dijkstra\nPure')
                            for m in model_names], fontsize=8)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_ylim(0, y_max * 1.25)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    # Shared model color legend at the top
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors.get(m, "#888"), alpha=0.85)
               for m in model_names]
    fig.legend(handles, model_names, loc='upper center', ncol=n_models,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.03),
               prop={'family': 'serif', 'size': 11})

    fig.suptitle("Model Performance Comparison — Raw Average Metric Values\n"
                 "(* gold border = best performer per metric)",
                 fontsize=14, fontweight='bold', family='serif', y=1.06)

    plt.tight_layout(pad=2.5)

    output_img = os.path.join(script_dir, "model_comparison_bar.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {output_img}")
    plt.show(block=False)

if __name__ == "__main__":
    generate_raw_bar_chart()
