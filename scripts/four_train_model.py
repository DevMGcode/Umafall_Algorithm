import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import numpy as np

def main():
    # Cargar datos optimizados
    df = pd.read_csv("data/wrist_filtered/features/wrist_features_optimized.csv")

    # Variables y etiqueta
    X = df.drop(columns=['label', 'activity_type', 'file'])
    y = df['label']

    # Dividir datos (70% train, 30% test) con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Aplicar SMOTE para balancear clases en el conjunto de entrenamiento
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Definir modelo base
    rf = RandomForestClassifier(random_state=42)

    # Definir espacio de búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Búsqueda aleatoria con validación cruzada (3 folds)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='f1'  # optimizar F1-score
    )

    # Entrenar búsqueda
    random_search.fit(X_train_res, y_train_res)

    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    # Mejor modelo
    best_rf = random_search.best_estimator_

    # Evaluar en test
    y_pred = best_rf.predict(X_test)
    print("Reporte de clasificación en test:")
    print(classification_report(y_test, y_pred, target_names=['ADL', 'Fall']))

    # Guardar modelo y datos de test
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_rf, 'models/random_forest_fall_detection_optimized.pkl')
    X_test.to_csv('models/X_test.csv', index=False)
    y_test.to_csv('models/y_test.csv', index=False)

    print("Modelo optimizado guardado en 'models/random_forest_fall_detection_optimized.pkl'")

if __name__ == "__main__":
    main()