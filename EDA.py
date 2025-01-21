
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# 1. Load data
df = pd.read_csv('Spotify Most Streamed Songs.csv')

# 2. Basic information
def basic_info(df):
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())

# 3. Missing values analysis
def analyze_missing(df):
    missing = df.isnull().sum()/len(df)*100
    missing = missing[missing > 0]
    plt.figure(figsize=(10,4))
    missing.plot(kind='bar')
    plt.title('Missing Values Percentage')
    plt.show()

# 4. Numerical analysis
def analyze_numerical(df):
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in num_cols:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        df[col].hist()
        plt.title(f'Histogram of {col}')
        plt.subplot(1,2,2)
        df[col].plot(kind='box')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

# 5. Categorical analysis
def analyze_categorical(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(10,4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()

# 6. Correlation analysis
def correlation_analysis(df):
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    plt.figure(figsize=(10,8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Main EDA function
def perform_eda(df):
    basic_info(df)
    analyze_missing(df)
    analyze_numerical(df)
    analyze_categorical(df)
    correlation_analysis(df)
# %%
# Run EDA
perform_eda(df)
# %%
