import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import yaml
import pickle
import mlflow
import mlflow.sklearn

def train_model():
    """Обучение модели с логированием в MLflow"""
    # Загрузка параметров
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    # Настройка MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine_classification")
    
    with mlflow.start_run():
        # Выбор модели
        if params['train']['model_type'] == "random_forest":
            model = RandomForestClassifier(
                n_estimators=params['train']['n_estimators'],
                max_depth=params['train']['max_depth'],
                random_state=params['train']['random_state']
            )
        elif params['train']['model_type'] == "logistic_regression":
            model = LogisticRegression(random_state=params['train']['random_state'])
        else:
            raise ValueError(f"Unknown model type: {params['train']['model_type']}")
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Логирование параметров
        mlflow.log_params(params['train'])
        mlflow.log_param("test_size", params['prepare']['test_size'])
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        
        # Логирование модели
        mlflow.sklearn.log_model(model, "model")
        
        # Сохранение модели локально
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Логирование артефакта
        mlflow.log_artifact("model.pkl")
        mlflow.log_artifact("params.yaml")
        
        print(f"Model training completed! Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()