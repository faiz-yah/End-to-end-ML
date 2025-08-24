import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import wandb
from preprocessing_fixed import fit_preprocessing, transform_with_fitted
from train_test import test_model
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
    
    # Load training data only
    train_path = "https://raw.githubusercontent.com/faiz-yah/End-to-end-ML/refs/heads/main/spaceship_titanic/dataset/spaceship-titanic/train.csv"
    train_df = pd.read_csv(train_path)
    
    # Fit preprocessing on training data and get processed training data
    fitted_params, X_train_processed = fit_preprocessing(train_df, numerical_cols, categorical_cols, encode_categorical_cols, target_col, map_dict)
    
    X_train = X_train_processed.drop(columns=[target_col])
    y_train = X_train_processed[target_col]
    
    # Load test data
    test_path = "https://raw.githubusercontent.com/faiz-yah/End-to-end-ML/refs/heads/main/spaceship_titanic/dataset/spaceship-titanic/test.csv"
    test_df = pd.read_csv(test_path)
    
    # Transform test data using fitted parameters (no target in test data)
    X_test = transform_with_fitted(test_df, fitted_params, numerical_cols, categorical_cols, encode_categorical_cols, target_col, map_dict, has_target=False)
    
    return X_train, X_test, y_train, None
    
def main():
    with open("wandb_config.yaml") as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(project="spaceship-titanic", config=sweep_config)
    
    X_train, X_test, y_train, _ = prepare_data()
    
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
    
    # For competition data without target, you'd just predict:
    # y_pred = model.predict(X_test)
    
    # For now, we'll use a train/validation split from training data for evaluation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model.fit(X_train_split, y_train_split)
    y_pred, accuracy, precision, recall, class_report, conf_matrix = test_model(model, X_val_split, y_val_split)
    
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