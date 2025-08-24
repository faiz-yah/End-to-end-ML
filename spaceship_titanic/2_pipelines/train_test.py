import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix


def train_test_split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose 'random_forest' or 'logistic_regression'.")
    
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return y_pred, accuracy, precision, recall, class_report, conf_matrix

