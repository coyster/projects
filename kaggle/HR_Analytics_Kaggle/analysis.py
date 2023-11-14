import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.

def get_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), data_path))

    return df


def save_data(df: pd.DataFrame, data_path: str) -> None:
    df.to_csv(os.path.join(os.path.dirname(__file__), data_path))


def describe_data(df: pd.DataFrame) -> None:
    for col in df.columns:
        d = df[col].describe()
        print(f"{col}: {d}")
        save_data(d, f"data/{col}_stats.csv")


def correlate_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    corr = ""
    for c in df.columns:
        corr_temp = df[c].corr()

    return corr


def correlation_matrix(df: pd.DataFrame, show=False, target_col=None) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    if show:
        plt.figure(figsize=(10, 10))
        if target_col is not None:
            sns.heatmap(corr[[target_col]], cmap="YlGnBu", annot=True)
        else:
            sns.heatmap(corr, cmap="YlGnBu", annot=True)
        plt.show()
    save_data(corr, f"data/correlation_matrix.csv")
    return corr


#####################################################

df = get_data("data/archive/HR_Analytics.csv")
corr_df = correlation_matrix(df, show=False, target_col="JobSatisfaction")
# corr_df = correlation_matrix(df, show=True, target_col="JobSatisfaction")
print(corr_df["JobSatisfaction"])



# JobSatisfaction
# corr = df.corr(method="pearson", numeric_only=True)

# plt.plot(corr, label="JobSatisfaction")
# plt.show()
# print(corr)
# plt.plot(df["JobSatisfaction"], label="JobSatisfaction")
# plt.hist(df["JobSatisfaction"], bins=8, label="JobSatisfaction" )
# plt.show()
