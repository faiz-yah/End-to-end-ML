import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import wandb
from preprocessing import pre_processing_integrated
from train_test import train_test_split_data, test_model
import json
import yaml

def prepare_data():
    
    with open("config.json", "r") as f:
        config = json.load(f)
    
    numerical_cols = config["numerical_cols"]
    categorical_cols = config["categorical_cols"]
    encode_categorical_cols = config["encode_categorical_cols"]
    target_col = config["target_col"]
    map_dict = {True: config["map_dict"]["true"], False: config["map_dict"]["false"]}
    
    test_size = config["test_size"]
    random_state = config["random_state"]
    
    data_path_1 = "https://raw.githubusercontent.com/faiz-yah/End-to-end-ML/refs/heads/main/spaceship_titanic/dataset/spaceship-titanic/test.csv"
    data_path_2 = "https://raw.githubusercontent.com/faiz-yah/End-to-end-ML/refs/heads/main/spaceship_titanic/dataset/spaceship-titanic/train.csv"

    df_1 = pd.read_csv(data_path_1)
    df_2 = pd.read_csv(data_path_2)

    df = pd.concat([df_1, df_2])

    df = pre_processing_integrated(df, numerical_cols, categorical_cols, encode_categorical_cols, target_col, map_dict)
    
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_col, test_size, random_state)
    
    return X_train, X_test, y_train, y_test
    
def main():
    with open("wandb_config.yaml") as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(project="spaceship-titanic", config=sweep_config)
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    n_estimators = wandb.config.n_estimators
    max_depth = wandb.config.max_depth
    min_samples_split = wandb.config.min_samples_split
    min_samples_leaf = wandb.config.min_samples_leaf


    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    y_pred, accuracy, precision, recall, class_report, conf_matrix = test_model(model, X_test, y_test) 
    
    wandb.log(
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "classification_report": class_report,
            "conf_matrix": conf_matrix.tolist(),
        }
    )
    
main()

