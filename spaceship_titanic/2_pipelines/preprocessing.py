from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler

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


