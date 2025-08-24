from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

########### Select Columns ###########
def selected_columns(df, numerical_cols, categorical_cols, target_col):
    return df[numerical_cols + categorical_cols + [target_col]]

########### Remove rows without Target ###########
def remove_rows_without_target(df, target_col):
    df = df.dropna(subset=[target_col])
    return df

########### Fit preprocessing on training data ###########
def fit_preprocessing(df, numerical_cols, categorical_cols, encode_categorical_cols, target_col, map_dict):
    """Fit preprocessing parameters on training data and return fitted objects"""
    
    # Select columns and remove rows without target
    df = selected_columns(df, numerical_cols, categorical_cols, target_col)
    df = remove_rows_without_target(df, target_col)
    
    # Learn imputation values
    numerical_medians = df[numerical_cols].median().to_dict()
    categorical_modes = df[categorical_cols].mode().iloc[0].to_dict()
    
    # Apply imputation to fit other transformers
    df[numerical_cols] = df[numerical_cols].fillna(numerical_medians)
    df[categorical_cols] = df[categorical_cols].fillna(categorical_modes)
    
    # Fit log transformer
    log_transformer = FunctionTransformer(func=np.log1p, validate=True)
    df[numerical_cols] = log_transformer.fit_transform(df[numerical_cols])
    
    # Fit scaler
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Apply encoding to see what columns we'll have
    df = pd.get_dummies(df, columns=encode_categorical_cols, drop_first=True)
    
    # Map target
    df[target_col] = df[target_col].map(map_dict)
    
    return {
        'numerical_medians': numerical_medians,
        'categorical_modes': categorical_modes,
        'log_transformer': log_transformer,
        'scaler': scaler,
        'all_columns': df.columns.tolist()
    }, df

########### Transform data using fitted parameters ###########
def transform_with_fitted(df, fitted_params, numerical_cols, categorical_cols, encode_categorical_cols, target_col, map_dict, has_target=True):
    """Transform data using fitted preprocessing parameters"""
    
    if has_target:
        df = selected_columns(df, numerical_cols, categorical_cols, target_col)
        df = remove_rows_without_target(df, target_col)
    else:
        df = df[numerical_cols + categorical_cols].copy()
    
    # Apply learned imputation
    df[numerical_cols] = df[numerical_cols].fillna(fitted_params['numerical_medians'])
    df[categorical_cols] = df[categorical_cols].fillna(fitted_params['categorical_modes'])
    
    # Apply log transform and scaling using fitted objects
    df[numerical_cols] = fitted_params['log_transformer'].transform(df[numerical_cols])
    df[numerical_cols] = fitted_params['scaler'].transform(df[numerical_cols])
    
    # Apply encoding
    df = pd.get_dummies(df, columns=encode_categorical_cols, drop_first=True)
    
    # Ensure same columns as training data
    for col in fitted_params['all_columns']:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    if has_target:
        df = df.reindex(columns=fitted_params['all_columns'], fill_value=0)
        # Map target
        df[target_col] = df[target_col].map(map_dict)
    else:
        # For test data without target, exclude target column
        test_columns = [col for col in fitted_params['all_columns'] if col != target_col]
        df = df.reindex(columns=test_columns, fill_value=0)
    
    return df