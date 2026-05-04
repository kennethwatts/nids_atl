import pandas as pd

def inspect_data(filename):
    df = pd.read_csv(filename)
    print(f"--- Inspecting {filename} ---")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique Labels: {df['Label'].unique()}")
    print("-" * 30)

#inspect_data('robust_2017_final.csv')
#inspect_data('robust_2018_final.csv')
inspect_data('master_stratified_dataset_2017.csv')
inspect_data('master_stratified_dataset_2018.csv')
