import pandas as pd

df = pd.read_parquet("cleaned_traffic.parquet")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nTop 15 Violation Groups:")
print(df['violation_group'].value_counts().head(15))

print("\nPercentage of top groups:")
print((df['violation_group'].value_counts(normalize=True).head(10) * 100).round(2))

print("\nSample rows (description + group):")
print(df[['description', 'violation_group']].head(8).to_string(index=False))

print("\nTop 10 Makes:")
print(df['make'].value_counts().head(10))

print("\nTop 10 Colors:")
print(df['color'].value_counts().head(10))