import pandas as pd
import os

file_path = "Baseline/simulation_results_base_2.xlsx"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    # Try another one
    file_path = "Dijkstra_SafeZones/simulation_results_dijkstra_safe_2.xlsx"

if os.path.exists(file_path):
    print(f"Inspecting: {file_path}")
    try:
        xls = pd.ExcelFile(file_path)
        print("Sheet names:", xls.sheet_names)
        for sheet in xls.sheet_names:
            print(f"\n--- Sheet: {sheet} ---")
            df = pd.read_excel(xls, sheet_name=sheet, nrows=5)
            print(df.to_string())
    except Exception as e:
        print(f"Error reading excel: {e}")
else:
    print("No simulation files found to inspect.")
