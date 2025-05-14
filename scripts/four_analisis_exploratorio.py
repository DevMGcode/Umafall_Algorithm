
#four_analisis_exploratorio.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Obtener ruta absoluta del archivo preprocesado basado en la ubicación del script
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "results", "combined_preprocessed_1.5s.csv")

print("Cargando archivo desde:", file_path)


# Cargar dataset preprocesado
df = pd.read_csv(file_path)

# 1. Información general del dataset
print("Número total de muestras:", len(df))
print("Número de columnas:", len(df.columns))
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# 2. Distribución de clases (caída vs no caída)
print("\nDistribución de clases (0=No caída, 1=Caída):")
print(df['label'].value_counts())
print("\nPorcentaje de cada clase:")
print(df['label'].value_counts(normalize=True) * 100)

# Gráfico de barras para visualizar el balance de clases
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Distribución de clases")
plt.xlabel("Clase (0=No caída, 1=Caída)")
plt.ylabel("Cantidad de muestras")
plt.show()

# 3. Estadísticas descriptivas de las características numéricas
print("\nEstadísticas descriptivas de las características numéricas:")
print(df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

# 4. Correlación entre algunas características seleccionadas
features_to_check = ['var_X', 'var_Y', 'var_Z', 'mean_X', 'mean_Y', 'mean_Z', 'std_X', 'std_Y', 'std_Z']
corr_matrix = df[features_to_check].corr()

print("\nMatriz de correlación entre características seleccionadas:")
print(corr_matrix)

# Mapa de calor para visualizar la correlación
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de calor de correlación")
plt.show()

# 5. Distribución de una característica ejemplo (var_X) por clase
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='var_X', hue='label', bins=30, kde=True, stat="density", common_norm=False)
plt.title("Distribución de var_X por clase")
plt.xlabel("var_X (varianza en eje X)")
plt.ylabel("Densidad")
plt.show()