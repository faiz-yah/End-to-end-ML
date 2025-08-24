from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

########### Imputation of Missing Values ###########
## Nummerical Features
def impute_numerical(df, numerical_cols):
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    return df

## Categorical Features
def impute_categorical(df, categorical_cols):
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    return df

########### Log Transformation for Long-tailed Distribution ###########
def log_transform(df, numerical_cols):
    log_transformer = FunctionTransformer(func=np.log1p, validate=True)
    df[numerical_cols] = log_transformer.fit_transform(df[numerical_cols])
    return df

########### Standardised Numerical Values ###########
def standardize_numerical(df, numerical_cols):
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


########### Remove rows without Target ###########
def remove_rows_without_target(df, target_col):
    df = df.dropna(subset=[target_col])
    return df

########### One hot encoding - Categorical  ###########
def encode_categorical(df, encode_categorical_cols):
    df = pd.get_dummies(df, columns=encode_categorical_cols, drop_first=True)
    return df


########### Target Mapping ###########
def map_target(df, target_col, map_dict):
    df[target_col] = df[target_col].map(map_dict)
    return df


########### Drop Columns ###########
def drop_columns(df, target_col, map_dict):
    df[target_col] = df[target_col].map(map_dict)
    return df