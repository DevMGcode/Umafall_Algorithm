import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from imblearn.combine import SMOTEENN
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("data/wrist_filtered/features/wrist_features_optimized.csv")

    X = df.drop(columns=['label', 'activity_type', 'file'])
    y = df['label']

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # SMOTEENN para balancear clases (sobremuestreo + submuestreo)
    smote_enn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

    # RandomForest con pesos balanceados para clases
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='f1'
    )

    random_search.fit(X_train_res, y_train_res)

    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    best_rf = random_search.best_estimator_

    # Predecir probabilidades para ajustar umbral
    y_probs = best_rf.predict_proba(X_test)[:, 1]

    # Calcular precisión, recall y umbrales
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

    # Graficar curvas
    plt.plot(thresholds, precisions[:-1], label='Precisión')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.plot(thresholds, f1_scores[:-1], label='F1-score')
    plt.xlabel('Umbral')
    plt.legend()
    plt.title('Curva Precisión, Recall y F1-score vs Umbral')
    plt.show()

    # Mostrar los 5 mejores umbrales según F1-score
    best_indices = f1_scores.argsort()[-5:][::-1]
    print("Top 5 umbrales por F1-score:")
    for i in best_indices:
        print(f"Umbral: {thresholds[i]:.3f}, F1-score: {f1_scores[i]:.3f}, Precisión: {precisions[i]:.3f}, Recall: {recalls[i]:.3f}")

    # Aquí puedes elegir manualmente el umbral que prefieras
    threshold = 0.5  # Cambia este valor según la gráfica y los resultados impresos
    y_pred_adj = (y_probs >= threshold).astype(int)

    print(f"Reporte de clasificación con umbral ajustado = {threshold}:")
    print(classification_report(y_test, y_pred_adj, target_names=['ADL', 'Fall']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_rf, 'models/random_forest_fall_detection_optimized.pkl')
    X_test.to_csv('models/X_test.csv', index=False)
    y_test.to_csv('models/y_test.csv', index=False)

    print("Modelo optimizado guardado en 'models/random_forest_fall_detection_optimized.pkl'")

if __name__ == "__main__":
    main()