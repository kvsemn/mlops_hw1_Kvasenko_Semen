import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os

def load_data():
    """Загрузка и подготовка данных"""
    df = pd.read_csv('data/raw/wine.csv', header=None)
    
    # Добавляем названия колонок для датасета Wine
    column_names = [
        'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 
        'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
        'proanthocyanins', 'color_intensity', 'hue', 'od280/od315', 'proline'
    ]
    df.columns = column_names
    
    return df

def prepare_data():
    """Подготовка данных для обучения"""
    # Загрузка параметров
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    df = load_data()
    
    # Разделение на признаки и целевую переменную
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['prepare']['test_size'],
        random_state=params['prepare']['random_state'],
        stratify=y
    )
    
    # Сохранение обработанных данных
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Data preparation completed!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

if __name__ == "__main__":
    prepare_data()