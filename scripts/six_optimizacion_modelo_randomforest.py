#six_optimizacion_randomforest.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# Definir ruta base y carpeta 'results'
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Cargar datos preprocesados
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(results_dir, "combined_preprocessed_1.5s.csv")
df = pd.read_csv(file_path)

# Separar características y etiquetas
exclude_cols = ['FileName', 'Fall_ADL', 'Fall_ADL2', 'Type_of_Movement', 'Description_of_movement', 'Gender', 'label']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df['label']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("Optimizando hiperparámetros para Random Forest...")

# Crear pipeline con SMOTE
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

# Definir parámetros para búsqueda
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Configurar búsqueda de hiperparámetros con validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
)

# Ejecutar búsqueda
grid_search.fit(X_train, y_train)

# Mostrar mejores parámetros
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)
print(f"Mejor F1 score: {grid_search.best_score_:.4f}")

# Evaluar modelo optimizado en conjunto de prueba
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nEvaluación del modelo optimizado en conjunto de prueba:")
print(classification_report(y_test, y_pred, target_names=['No Caída', 'Caída']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Caída', 'Caída'],
            yticklabels=['No Caída', 'Caída'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Random Forest Optimizado')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix_optimized.png'))
plt.show()

# Guardar resultados de optimización
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(os.path.join(results_dir, 'optimization_results.csv'), index=False)

# Guardar modelo optimizado
import joblib
joblib.dump(best_model, os.path.join(results_dir, 'optimized_random_forest.pkl'))

print("\nResultados de optimización guardados en 'results/optimization_results.csv'")
print("Modelo optimizado guardado en 'results/optimized_random_forest.pkl'")


# Análisis de importancia de características
feature_importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nCaracterísticas más importantes:")
print(feature_importances.head(10))

# Visualizar importancia de características
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
plt.title('Top 10 Características Más Importantes')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
plt.show()

print("\nOptimización del algoritmo completada.")