import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col='class', test_size=0.2, random_state=42):
    """
    Prepare data for modeling by separating features and target, and splitting into train/test sets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The preprocessed dataframe
    target_col : str
        Name of the target column ('class' or 'Class')
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        Split feature and target arrays for training and testing
    feature_names : list
        List of feature names used in the model
    """
    # Identify the target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Remove any non-numeric columns that weren't properly encoded
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns if col != target_col]
    
    print(f"Number of features selected: {len(feature_columns)}")
    print("\nFeatures to be used in model:")
    for col in feature_columns:
        print(f"- {col}")
    
    # Separate features and target
    X = df[feature_columns]
    y = df[target_col]
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensure balanced split for imbalanced datasets
    )
    
    # Print split information
    print(f"\nData split summary:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"\nClass distribution:")
    print("Training set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("\nTest set:")
    print(pd.Series(y_test).value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test, feature_columns 