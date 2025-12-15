import pandas as pd

DATA_PATH = "data/raw/dataset_v1.csv"

df = pd.read_csv(DATA_PATH)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)
