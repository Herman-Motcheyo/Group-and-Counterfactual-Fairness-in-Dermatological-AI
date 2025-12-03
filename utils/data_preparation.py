
import numpy as np
import pandas as pd

def handle_missing_values(df):
    df = df.copy()

    if 'age' in df.columns:
        median_age = df['age'].median()
        df['age'].fillna(median_age, inplace=True)

    if 'sex' in df.columns:
        df['sex'] = df['sex'].replace(['unknown', 'Unknown'], np.nan)

        # Remplir les valeurs manquantes avec le mode
        if df['sex'].isnull().any():
            mode_sex = df['sex'].mode()[0]
            df['sex'].fillna(mode_sex, inplace=True)

    return df
