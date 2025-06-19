import pandas as pd


# -------------------
# EPANET file parsing
# -------------------

# Pressure data file parser (for P.txt)
def read_pressure_timeseries(filepath: str) -> pd.DataFrame:
    records = []
    current_node = None

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # 3 parts indicate a line starting with node name
            if len(parts) == 3 and not line.startswith(' '):
                current_node, time_str, value_str = parts
            # 2 parts indicate a line with just time and pressure value
            elif len(parts) == 2:
                time_str, value_str = parts
            else:
                continue

            time_index = pd.to_datetime(time_str, format='%H:%M').time()
            pressure = float(value_str)

            records.append({
                'node': current_node,
                'time': time_index,
                'pressure': pressure
            })

    # Convert to DataFrame
    df = pd.DataFrame.from_records(records)
    
    # Pivot table: nodes as files, time as an index
    df_pivot = df.pivot(index='time', columns='node', values='pressure')
    
    return df_pivot