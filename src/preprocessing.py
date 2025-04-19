# preprocessing.py
#
# This module provides utility functions for loading, exploring, cleaning, and preprocessing a dataset.
# It prepares the data for training and evaluation by applying techniques such as:
# - Exploratory Data Analysis (EDA)
# - Missing value imputation
# - Train-test split
# - Standard scaling
# - Handling class imbalance using SMOTE
#
# Main Functions:
# - load_data: Loads a dataset from a CSV file
# - eda: Performs exploratory data analysis with visualizations
# - handle_missing_data: Fills missing values using median imputation
# - split_features_target: Splits the dataset into features and target
# - split_train_test: Splits the dataset into train and test sets
# - preprocess_data: Loads and fully preprocesses the dataset for model training
#
# Example usage:
# X_train_res, X_test, y_train_res, y_test = preprocess_data('../data/dataset.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def eda(df):
    """
    This function performs basic exploratory data analysis on the dataset.
    It provides the following insights:
    - Shape of the dataset
    - Data types of the columns
    - Summary statistics
    - Missing values count
    - Correlation matrix
    - Visualizations for distribution and pairwise relationships.

    Parameters:
        df (pd.DataFrame): The input dataset to be analyzed.
    """
    # Print the shape of the dataset
    print(f"Shape of the dataset: {df.shape}")

    # Print the data types of the columns
    print("\nData types:\n", df.dtypes)

    # Summary statistics
    print("\nSummary Statistics:\n", df.describe())

    # Missing values
    print("\nMissing values:\n", df.isnull().sum())

    # Correlation matrix - Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    print("\nCorrelation Matrix:\n", numerical_df.corr())

    # Visualizations
    # Distribution of experience years
    plt.figure(figsize=(10, 6))
    sns.histplot(df['experience_years'], kde=True, bins=20)
    plt.title('Distribution of Experience Years')
    plt.xlabel('Experience Years')
    plt.ylabel('Frequency')
    plt.show()

    # Distribution of technical score
    plt.figure(figsize=(10, 6))
    sns.histplot(df['technical_score'], kde=True, bins=20, color='orange')
    plt.title('Distribution of Technical Score')
    plt.xlabel('Technical Score')
    plt.ylabel('Frequency')
    plt.show()

    # Correlation Heatmap for numerical data
    plt.figure(figsize=(8, 6))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

    # Pairplot for experience_years vs technical_score and label
    sns.pairplot(df, vars=["experience_years", "technical_score", "label"], hue="label")
    plt.show()


import numpy as np

def handle_missing_data(df):
    """
    Handle missing values in the dataset.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
    # Select only numeric columns and cast them to float to avoid issues
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Fill missing values with median for numeric columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    return df



def split_features_target(df, target_column):
    """
    Split the dataset into features (X) and target (y).

    Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): The column name of the target variable.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    X = df.drop(columns=[target_column])  # Features: All columns except the target
    y = df[target_column]  # Target variable: The specified column
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: The training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def preprocess_data(file_path):
    """
    Load, clean, and preprocess the dataset for model training.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset ready for training.
    """
    df = load_data(file_path)
    #eda(df)
    df = handle_missing_data(df)

    # Split into features (X) and target (y)
    X = df[['experience_years', 'technical_score']]  # Example features
    y = df['label']  # Target variable

    # Ensure that we only use numeric columns for scaling
    X_numeric = X.select_dtypes(include=[np.number])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    # Standardize the features (important for models like SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test



if __name__ == "__main__":
    # Example usage
    X_train_res, X_test, y_train_res, y_test = preprocess_data('../data/dataset.csv')
    print("Data preprocessed successfully!")
