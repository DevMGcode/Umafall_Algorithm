#seven_validacion_final.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_curve, auc, precision_recall_curve,
                            average_precision_score)
import joblib


# Rutas
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Cargar datos preprocesados
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(results_dir, "combined_preprocessed_1.5s.csv")
df = pd.read_csv(file_path)

# Cargar modelo optimizado
model_path = os.path.join(results_dir, "optimized_random_forest.pkl")
best_model = joblib.load(model_path)

print("Realizando validación final del modelo optimizado...")

# Separar características y etiquetas
exclude_cols = ['FileName', 'Fall_ADL', 'Fall_ADL2', 'Type_of_Movement', 'Description_of_movement', 'Gender', 'label']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df['label']

# Validación cruzada para predicciones
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
y_prob_cv = cross_val_predict(best_model, X, y, cv=cv, method='predict_proba')[:, 1]

# Reporte de clasificación
print("\nReporte de Clasificación con Validación Cruzada:")
print(classification_report(y, y_pred_cv, target_names=['No Caída', 'Caída']))

# Matriz de confusión
cm = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Caída', 'Caída'],
            yticklabels=['No Caída', 'Caída'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Validación Cruzada')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix_final.png'))
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y, y_prob_cv)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Área bajo la curva = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Detección de Caídas')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
plt.show()

# Curva Precisión-Recall
precision, recall, _ = precision_recall_curve(y, y_prob_cv)
avg_precision = average_precision_score(y, y_prob_cv)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precisión promedio = {avg_precision:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.title('Curva Precisión-Recall - Detección de Caídas')
plt.legend(loc="lower left")
plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
plt.show()

# Análisis por tipo de caída (si está disponible)
if 'Type_of_Movement' in df.columns and 'Description_of_movement' in df.columns:
    print("\nAnálisis por tipo de caída:")

    # Filtrar solo caídas
    falls_df = df[df['label'] == 1]

    # Agrupar por tipo de caída
    fall_types = falls_df.groupby('Description_of_movement').size().reset_index(name='count')
    fall_types = fall_types.sort_values('count', ascending=False)

    print(fall_types)

    # Visualizar distribución de tipos de caída
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='Description_of_movement', data=fall_types)
    plt.title('Distribución de Tipos de Caída')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fall_types_distribution.png'))
    plt.show()

    # Evaluar rendimiento por tipo de caída
    fall_performance = {}

    for fall_type in falls_df['Description_of_movement'].unique():
        # Filtrar solo este tipo de caída
        type_indices = df[df['Description_of_movement'] == fall_type].index

        if len(type_indices) < 10:  # Omitir tipos con pocas muestras
            continue

        # Calcular métricas para este tipo
        y_true_type = y.iloc[type_indices]
        y_pred_type = y_pred_cv[type_indices]

        # Calcular precisión y recall
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_true_type, y_pred_type)
        recall = recall_score(y_true_type, y_pred_type)

        fall_performance[fall_type] = {
            'precision': precision,
            'recall': recall,
            'count': len(type_indices)
        }

    # Mostrar rendimiento por tipo
    performance_df = pd.DataFrame.from_dict(fall_performance, orient='index')
    print("\nRendimiento por tipo de caída:")
    print(performance_df)

    # Guardar análisis
    performance_df.to_csv(os.path.join(results_dir, 'fall_type_performance.csv'))

# Guardar resultados finales
final_results = {
    'accuracy': (y == y_pred_cv).mean(),
    'roc_auc': roc_auc,
    'avg_precision': avg_precision
}

pd.DataFrame([final_results]).to_csv(os.path.join(results_dir, 'final_validation_results.csv'), index=False)

print("\nValidación final completada.")
print(f"Exactitud global: {final_results['accuracy']:.4f}")
print(f"Área bajo la curva ROC: {final_results['roc_auc']:.4f}")
print(f"Precisión promedio: {final_results['avg_precision']:.4f}")