#three_preprocesamiento_combinado.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Obtener ruta absoluta de la carpeta donde está el script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construir rutas absolutas a los archivos CSV
umafall_path = os.path.join(base_dir, '..', 'unified_UMAFALL_dataset', 'UMAFall_window_1.5s_all.csv')
wedafall_path = os.path.join(base_dir, '..', 'unified_WEDAFALL_dataset', 'WEDAFall_window_1.5s_all.csv')

print("Ruta UMAFall:", umafall_path)
print("Ruta WEDAFall:", wedafall_path)

# Cargar datasets
umafall = pd.read_csv(umafall_path)
wedafall = pd.read_csv(wedafall_path)

# Unir datasets
df = pd.concat([umafall, wedafall], ignore_index=True)

# Mostrar info básica
print("Dimensiones combinadas:", df.shape)
print("Columnas:", df.columns.tolist())
print("Valores faltantes por columna:\n", df.isnull().sum())

# Limpieza: eliminar filas con valores faltantes
df_clean = df.dropna()

# Etiquetado binario: 1 para caída (FALL), 0 para ADL
df_clean['label'] = df_clean['Type_of_Movement'].apply(lambda x: 1 if x.upper() == 'FALL' else 0)

# Seleccionar columnas numéricas para normalizar (excluyendo etiquetas y texto)
exclude_cols = ['FileName', 'Fall_ADL', 'Fall_ADL2', 'Type_of_Movement', 'Description_of_movement', 'Gender', 'label']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

# Normalización
scaler = StandardScaler()
df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

# Verificar balance de clases
print("Distribución de clases:\n", df_clean['label'].value_counts(normalize=True))

# Dividir en train y test para futuros pasos
X = df_clean[feature_cols]
y = df_clean['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Guardar preprocesados para uso posterior

results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, "combined_preprocessed_1.5s.csv")
df_clean.to_csv(output_file, index=False)
print(f"Preprocesamiento completado y archivo guardado: {output_file}")

""" df_clean.to_csv(os.path.join(base_dir, "combined_preprocessed_1.5s.csv"), index=False)
print("Preprocesamiento completado y archivo guardado: combined_preprocessed_1.5s.csv") """