import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("husl")

def load_data(name, path):
    """Loads simulation data from an Excel file."""
    if not os.path.exists(path):
        print(f"Warning: {name} file not found at {path}")
        return None
    try:
        data = {}
        # 1. Scalars (Run Comparisons / Summary)
        try:
            df_runs = pd.read_excel(path, sheet_name='Run Comparisons')
            # Look for AVERAGE row
            df_runs['Run'] = df_runs['Run'].astype(str)
            avg_row = df_runs[df_runs['Run'] == 'AVERAGE']
            
            if not avg_row.empty:
                data['scalars'] = avg_row.iloc[0].to_dict()
            else:
                # Fallback: mean of numeric columns
                data['scalars'] = df_runs.mean(numeric_only=True).to_dict()
        except Exception as e:
             # Fallback for single run files
             try:
                 data['scalars'] = pd.read_excel(path, sheet_name='Summary').iloc[0].to_dict()
             except:
                 data['scalars'] = {}

        # 2. Time Series
        try:
            data['time'] = pd.read_excel(path, sheet_name='Avg Time Series')
        except:
            data['time'] = None
        
        # 3. Exits
        try:
            data['exits'] = pd.read_excel(path, sheet_name='Avg Exit Usage')
        except:
            data['exits'] = None
            
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def plot_scalar_comparison(models, metric_key, title, ylabel, output_dir, file_name):
    """Plots a bar chart for a scalar metric across models."""
    data = []
    for name, model_data in models.items():
        val = model_data['scalars'].get(metric_key, 0)
        # Handle cases where value might be non-numeric or missing
        try:
            val = float(val)
        except:
            val = 0
        data.append({'Model': name, ylabel: val})
    
    if not data:
        return

    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Model', y=ylabel, data=df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Simulation Model')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

def plot_exit_usage(models, output_dir):
    """Plots grouped bar chart for exit usage."""
    # Collect all data
    records = []
    for m_name, m_data in models.items():
        if m_data['exits'] is not None:
            df = m_data['exits']
            # Standardize column names if needed
            if 'Exit ID' in df.columns:
                # Find usage column
                usage_col = next((c for c in df.columns if 'Usage' in c), None)
                if usage_col:
                    for _, row in df.iterrows():
                        records.append({
                            'Model': m_name,
                            'Exit ID': str(int(row['Exit ID'])), # Keep as string for categorical plotting
                            'Usage': row[usage_col]
                        })

    if not records:
        print("No exit usage data found to plot.")
        return

    df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Exit ID', y='Usage', hue='Model', data=df)
    plt.title('Exit Usage Distribution by Model')
    plt.ylabel('Usage Count')
    plt.xlabel('Exit ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exit_usage_comparison.png'))
    plt.close()

def run_visualization(scenario_name, files):
    """Runs the full visualization pipeline for a given set of files."""
    print(f"--- Processing Scenario: {scenario_name} ---")
    
    # 1. Setup Output Directory
    out_dir = f"Results_Graphs_{scenario_name.replace(' ', '_')}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # 2. Load Data
    models = {}
    base_dir = os.getcwd() # Assume script is run from project root
    
    for name, rel_path in files.items():
        # Handle absolute vs relative paths
        if os.path.isabs(rel_path):
             path = rel_path
        else:
             path = os.path.join(base_dir, rel_path)
             
        print(f"Loading {name} from {path}...")
        res = load_data(name, path)
        if res and res.get('scalars'):
            models[name] = res
        else:
            print(f"Skipping {name} (no data)")

    if not models:
        print("No valid models loaded. Exiting.")
        return

    # 3. Plot Scalar Metrics
    metrics_to_plot = [
        ('Total Evacuation Time (s)', 'Total Evacuation Time (s)', 'total_time_comparison.png'),
        ('Total Casualties', 'Total Casualties', 'total_casualties_comparison.png'),
        ('Avg Exit Flow Rate', 'Avg Flow Rate (agents/s)', 'flow_rate_comparison.png'),
        ('Remaining Agents', 'Remaining Agents', 'remaining_agents_comparison.png'),
        ('Avg Velocity (m/s)', 'Avg Velocity (m/s)', 'velocity_comparison.png'),
        ('Avg Density (p/m2)', 'Avg Density (p/m2)', 'density_comparison.png')
    ]

    for key, label, fname in metrics_to_plot:
        # Check if key exists in at least one model
        has_data = any(key in m['scalars'] for m in models.values())
        if has_data:
            plot_scalar_comparison(models, key, f'{scenario_name}: {label}', label, out_dir, fname)
        else:
            print(f"Metric '{key}' not found in data.")

    # 4. Plot Exit Usage
    plot_exit_usage(models, out_dir)
    
    print(f"Graphs saved to {out_dir}\n")

if __name__ == "__main__":
    # --- SCENARIO 1: Full Comparison ---
    files_full = {
        'Baseline': 'Baseline/simulation_results_base_2.xlsx',
        'ACO': 'Baseline_ACO/simulation_results_aco_2.xlsx',
        'Dijkstra': 'Dijkstra_Pure/simulation_results_dijkstra_2.xlsx',
        'Safe Zones': 'Dijkstra_SafeZones/simulation_results_dijkstra_safe_2.xlsx'
    }
    run_visualization("Full Comparison", files_full)

    # --- SCENARIO 2: Safe vs Baseline vs Base(Pen) ---
    files_safe = {
        'Baseline': 'Baseline/simulation_results_base_2.xlsx',
        'Base + Penalty': 'Baseline/simulation_results_base_penalty_2.xlsx',
        'Safe Zones': 'Dijkstra_SafeZones/simulation_results_dijkstra_safe_2.xlsx'
    }
    run_visualization("Safe vs Baseline", files_safe)
