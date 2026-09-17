import pandas as pd

df = pd.read_csv('hydraulic_preprocessed_dataset.csv')
target_cols = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage', 'Accumulator_Pressure', 'Stable_Flag']
feat_cols = [c for c in df.columns if c not in target_cols + [t+'_enc' for t in target_cols] + ['Cycle_ID']]

# Pick 10 representative cycles with diverse conditions
sample_indices = [1664, 0, 211, 210, 599, 100, 350, 750, 1200, 1800]
sample_df = df.iloc[sample_indices][['Cycle_ID'] + feat_cols].copy()
sample_df.to_csv('sample_test_instances.csv', index=False)
print(f"Created sample_test_instances.csv with {len(sample_df)} rows and {len(sample_df.columns)} columns.")
