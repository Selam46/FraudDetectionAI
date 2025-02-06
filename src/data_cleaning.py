import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import ipaddress

def load_datasets():
    """
    Load all datasets from the data/raw directory
    """
    # Get the current file's directory and construct the correct path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data', 'raw')
    
    # Construct full paths to the files
    fraud_data_path = os.path.join(data_dir, 'Fraud_Data.csv')
    ip_country_path = os.path.join(data_dir, 'IpAddress_to_Country.csv')
    credit_card_path = os.path.join(data_dir, 'creditcard.csv')
    
    # Load the datasets
    fraud_data = pd.read_csv(fraud_data_path)
    ip_country_data = pd.read_csv(ip_country_path)
    credit_card_data = pd.read_csv(credit_card_path)
    
    return fraud_data, ip_country_data, credit_card_data

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataframe
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    df_cleaned = df.copy()
    
    # Get numerical and categorical columns
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    else:
        # Handle numerical columns
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                if strategy == 'mean':
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        # Handle categorical columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    return df_cleaned

def remove_duplicates(df):
    """
    Remove duplicate rows from dataframe
    """
    return df.drop_duplicates()

def convert_datatypes(df):
    """
    Convert columns to appropriate data types
    """
    df_converted = df.copy()
    
    # Convert timestamp columns if they exist
    timestamp_columns = ['signup_time', 'purchase_time']
    for col in timestamp_columns:
        if col in df_converted.columns:
            df_converted[col] = pd.to_datetime(df_converted[col])
    
    return df_converted

def ip_to_int(ip):
    """Convert IP address to integer"""
    try:
        # If IP is already a number (float or int), just return it as an integer
        if isinstance(ip, (float, int)):
            return int(float(ip))  # Convert scientific notation to integer
        return None
    except:
        return None

def convert_scientific_to_ip(scientific_notation):
    """Convert scientific notation to IP address string"""
    try:
        # Convert scientific notation to integer
        ip_int = int(float(scientific_notation))
        
        # Convert integer to IP address string
        ip_parts = []
        for _ in range(4):
            ip_parts.append(str(ip_int & 255))
            ip_int = ip_int >> 8
        return '.'.join(reversed(ip_parts))
    except:
        return None

def prepare_ip_data(fraud_df, ip_country_df):
    """
    Prepare and merge IP address data
    
    Parameters:
    fraud_df: DataFrame containing fraud data
    ip_country_df: DataFrame containing IP to country mapping
    
    Returns:
    DataFrame with country information merged
    """
    # Make copies of the dataframes
    fraud_df = fraud_df.copy()
    ip_country_df = ip_country_df.copy()
    
    print("Converting IP addresses to standard format...")
    
    # Convert scientific notation IPs to standard format
    fraud_df['ip_string'] = fraud_df['ip_address'].apply(convert_scientific_to_ip)
    
    # Convert IP strings to integers for comparison
    print("Converting IPs to integers for matching...")
    fraud_df['ip_int'] = fraud_df['ip_address'].astype(np.int64)
    ip_country_df['lower_int'] = ip_country_df['lower_bound_ip_address'].astype(np.int64)
    ip_country_df['upper_int'] = ip_country_df['upper_bound_ip_address'].astype(np.int64)
    
    # Print some statistics
    print(f"\nTotal records in fraud data: {len(fraud_df)}")
    print(f"Valid IP conversions: {fraud_df['ip_string'].notna().sum()}")
    
    # Merge with country data
    print("\nMatching IPs with country ranges...")
    fraud_df['country'] = None
    
    # Vectorized matching operation
    for idx, row in fraud_df.iterrows():
        if pd.notna(row['ip_int']):
            mask = (ip_country_df['lower_int'] <= row['ip_int']) & (ip_country_df['upper_int'] >= row['ip_int'])
            matching_countries = ip_country_df[mask]['country'].values
            if len(matching_countries) > 0:
                fraud_df.loc[idx, 'country'] = matching_countries[0]
        
        if idx % 10000 == 0:
            print(f"Processed {idx} records...")
    
    # Print matching statistics
    matches = fraud_df['country'].notna().sum()
    print(f"\nMatched {matches} IPs with countries ({matches/len(fraud_df)*100:.2f}%)")
    
    return fraud_df

def create_time_features(df):
    """
    Create time-based features from datetime columns
    """
    df = df.copy()
    
    # Convert timestamp columns to datetime if they're not already
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Create time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['purchase_time'].dt.month
    
    # Time difference between signup and purchase (in hours)
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    return df

def create_transaction_features(df):
    """
    Create transaction frequency and velocity features
    """
    df = df.copy()
    
    # Transaction count per user
    user_tx_counts = df.groupby('user_id').agg({
        'purchase_time': 'count',
        'purchase_value': ['mean', 'std', 'max', 'min']
    })
    
    user_tx_counts.columns = [
        'user_tx_count',
        'user_avg_purchase',
        'user_std_purchase',
        'user_max_purchase',
        'user_min_purchase'
    ]
    
    # Merge back to original dataframe
    df = df.merge(user_tx_counts, left_on='user_id', right_index=True, how='left')
    
    # Calculate transaction velocity (transactions per day)
    user_time_range = df.groupby('user_id').agg({
        'purchase_time': lambda x: (x.max() - x.min()).total_seconds() / (24 * 3600)
    }).reset_index()
    user_time_range.columns = ['user_id', 'user_time_range_days']
    
    df = df.merge(user_time_range, on='user_id', how='left')
    df['tx_velocity'] = df['user_tx_count'] / df['user_time_range_days'].clip(lower=1)
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using LabelEncoder
    """
    df = df.copy()
    encoders = {}
    
    categorical_columns = ['source', 'browser', 'sex', 'country']
    
    for column in categorical_columns:
        if column in df.columns:
            le = LabelEncoder()
            df[f'{column}_encoded'] = le.fit_transform(df[column].astype(str))
            encoders[column] = le
    
    return df, encoders

def scale_numerical_features(df):
    """
    Scale numerical features using StandardScaler
    """
    df = df.copy()
    scaler = StandardScaler()
    
    numerical_columns = [
        'age', 'purchase_value', 'user_tx_count', 
        'user_avg_purchase', 'user_std_purchase',
        'user_max_purchase', 'user_min_purchase',
        'tx_velocity', 'time_since_signup'
    ]
    
    # Only scale columns that exist in the dataframe
    columns_to_scale = [col for col in numerical_columns if col in df.columns]
    
    if columns_to_scale:
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    return df, scaler
