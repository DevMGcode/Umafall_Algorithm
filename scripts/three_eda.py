import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import mutual_info_classif

# Ruta al archivo de características optimizadas
FEATURES_FILE = "data/wrist_filtered/features/wrist_features_optimized.csv"
OUTPUT_DIR = "eda/plots_fall_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar datos
df = pd.read_csv(FEATURES_FILE)
df['label'] = df['label'].astype(int)

# 1. Balance de clases
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title("Distribución de clases (0=ADL, 1=Fall)")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"))
plt.close()

# 2. Estadísticas descriptivas básicas
print("Estadísticas descriptivas de edad y características numéricas:")
print(df.describe())

# 3. Importancia de características (mutual information)
X = df.drop(columns=['label', 'activity_type', 'file'])
y = df['label']
mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Guardar top 20 características importantes
mi_series.head(20).to_csv(os.path.join(OUTPUT_DIR, "feature_importance_summary.csv"), header=["mutual_info_score"])

top5 = mi_series.head(5)
print("Top 5 características más informativas:")
print(top5)

# 4. Boxplots para top 5 características por clase
plt.figure(figsize=(12, 8))
for i, feature in enumerate(top5.index):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=df, x='label', y=feature)
    plt.title(f'{feature} por clase')
    plt.xlabel("Clase")
    plt.ylabel(feature)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_features_boxplots.png"))
plt.close()

# 5. Heatmap de correlación entre características
plt.figure(figsize=(14, 12))
sns.heatmap(X.corr(), cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title("Matriz de correlación entre características")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# 6. Gráfico de barras de importancia de características (top 20)
plt.figure(figsize=(10, 6))
sns.barplot(x=mi_series.head(20).values, y=mi_series.head(20).index)
plt.title("Importancia de características (Mutual Information)")
plt.xlabel("Score")
plt.ylabel("Característica")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_mutual_info.png"))
plt.close()

print("EDA completado. Gráficas y reportes guardados en:", OUTPUT_DIR)