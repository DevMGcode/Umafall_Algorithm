#five_implementacion_algoritmos.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Importar varios algoritmos para comparar
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Cargar datos preprocesados
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
file_path = os.path.join(results_dir, "combined_preprocessed_1.5s.csv")
df = pd.read_csv(file_path)



# Separar características y etiquetas
exclude_cols = ['FileName', 'Fall_ADL', 'Fall_ADL2', 'Type_of_Movement', 'Description_of_movement', 'Gender', 'label']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df['label']

# Configurar validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definir modelos a comparar
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
}

# Resultados para cada modelo
results = {}
cv_scores = {}

print("Evaluando modelos con validación cruzada...")

# Evaluar cada modelo con validación cruzada
for name, model in models.items():
    print(f"\nEvaluando: {name}")

    # Crear pipeline con SMOTE
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    # Calcular scores con validación cruzada
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    cv_scores[name] = scores

    print(f"F1 scores en cada fold: {scores}")
    print(f"F1 score promedio: {scores.mean():.4f}")

    results[name] = {
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }

# Visualizar resultados de validación cruzada
plt.figure(figsize=(10, 6))
plt.boxplot([cv_scores[name] for name in models.keys()], labels=models.keys())
plt.title('Comparación de F1 scores entre modelos')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
plt.show()

# Tabla de resultados
results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'F1 Score Promedio': [results[model]['mean_score'] for model in results],
    'Desviación Estándar': [results[model]['std_score'] for model in results]
}).sort_values('F1 Score Promedio', ascending=False)

print("\nResumen de resultados:")
print(results_df)

# Guardar resultados
results_df.to_csv(os.path.join(results_dir, 'model_comparison_results.csv'), index=False)
print("\nResultados guardados en 'results/model_comparison_results.csv'")

# Identificar el mejor modelo
best_model_name = results_df.iloc[0]['Modelo']
print(f"\nEl mejor modelo es: {best_model_name} con F1 score promedio de {results_df.iloc[0]['F1 Score Promedio']:.4f}")

print("\nImplementación de algoritmos completada.")