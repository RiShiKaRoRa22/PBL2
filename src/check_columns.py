import pandas as pd

df = pd.read_csv("data/processed/final_train.csv")

print("\nColumns in final dataset:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head())
