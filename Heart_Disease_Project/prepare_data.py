import pandas as pd
import numpy as np

RAW_CSV = "data/heart_disease.csv"
CLEAN_CSV = "data/heart_disease_clean.csv"

df = pd.read_csv(RAW_CSV)
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='ignore')
df.drop_duplicates(inplace=True)
if 'target' in df.columns:
    df['target'] = df['target'].astype(int)
elif 'num' in df.columns:
    df.rename(columns={'num':'target'}, inplace=True)
    df['target'] = df['target'].astype(int)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

df.to_csv(CLEAN_CSV, index=False)
print("Saved cleaned data to", CLEAN_CSV)
print(df.head())