import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def extract_features_from_wrist(zip_file_path):
    """
    Extrae características de los datos del sensor de muñeca de todos los archivos en el dataset UMAFall.

    Args:
        zip_file_path: Ruta al archivo ZIP del dataset

    Returns:
        DataFrame con características y etiquetas
    """
    from fall_detector import FallDetector

    detector = FallDetector()
    features_list = []
    labels = []
    descriptions = []

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_list = [f for f in zip_ref.namelist() if f.endswith('.csv')]

        for file_name in file_list:
            try:
                with zip_ref.open(file_name) as file:
                    content = file.read().decode('utf-8')
                    lines = content.split('\n')

                    # Extraer metadatos
                    is_fall = False
                    description = "Unknown"
                    for line in lines[:15]:
                        if "Type of Movement: TRUE" in line:
                            is_fall = True
                        elif "Description of the movement:" in line:
                            description = line.split("Description of the movement:")[1].strip()

                    # Buscar la línea que contiene información del sensor de muñeca
                    wrist_sensor_id = None
                    for line in lines:
                        if "WRIST" in line:
                            parts = line.strip().split(';')
                            if len(parts) >= 3:
                                wrist_sensor_id = parts[1].strip()
                            break

                    if wrist_sensor_id is None:
                        continue

                    # Encontrar dónde comienzan los datos
                    data_start_idx = 0
                    for i, line in enumerate(lines):
                        if "TimeStamp" in line:
                            data_start_idx = i + 1
                            break

                    # Extraer datos del sensor de muñeca
                    data = []
                    for i in range(data_start_idx, len(lines)):
                        line = lines[i].strip()
                        if not line:
                            continue

                        parts = line.split(';')
                        if len(parts) >= 7 and parts[6].strip() == wrist_sensor_id:
                            try:
                                timestamp = float(parts[0])
                                x = float(parts[2])
                                y = float(parts[3])
                                z = float(parts[4])
                                data.append([timestamp, x, y, z])
                            except:
                                continue

                    if data:
                        data_array = np.array(data)
                        features = detector.extract_features(data_array)[0]
                        features_list.append(features)
                        labels.append(1 if is_fall else 0)
                        descriptions.append(description)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Crear DataFrame con características
    feature_names = [
        'mean_x', 'std_x', 'min_x', 'max_x', 'q1_x', 'median_x', 'q3_x', 'skew_x', 'kurtosis_x',
        'mean_y', 'std_y', 'min_y', 'max_y', 'q1_y', 'median_y', 'q3_y', 'skew_y', 'kurtosis_y',
        'mean_z', 'std_z', 'min_z', 'max_z', 'q1_z', 'median_z', 'q3_z', 'skew_z', 'kurtosis_z',
        'mean_mag', 'std_mag', 'min_mag', 'max_mag', 'q1_mag', 'median_mag', 'q3_mag', 'skew_mag', 'kurtosis_mag',
        'corr_xy', 'corr_xz', 'corr_yz'
    ]

    df = pd.DataFrame(features_list, columns=feature_names)
    df['is_fall'] = labels
    df['description'] = descriptions

    return df

def main():
    # Extraer características
    print("Extrayendo características del sensor de muñeca...")
    df = extract_features_from_wrist("UMAFall_Dataset.zip")

    if df.empty:
        print("No se pudieron extraer características del sensor de muñeca.")
        return

    print(f"Características extraídas: {df.shape[0]} muestras, {df.shape[1]-2} características")

    # Guardar características
    df.to_csv('data/wrist_features.csv', index=False)

    # Visualizar distribución de clases
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_fall', data=df)
    plt.title('Distribución de Clases')
    plt.xlabel('Es Caída')
    plt.ylabel('Frecuencia')
    plt.xticks([0, 1], ['No', 'Sí'])
    plt.grid(True, alpha=0.3)
    plt.savefig('results/class_distribution.png')

    # Visualizar distribución de actividades
    plt.figure(figsize=(12, 8))
    activity_counts = df['description'].value_counts()
    sns.barplot(x=activity_counts.values, y=activity_counts.index)
    plt.title('Distribución de Actividades')
    plt.xlabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('results/activity_distribution.png')

    # Visualizar correlación entre características
    plt.figure(figsize=(16, 14))
    corr_matrix = df.drop(['is_fall', 'description'], axis=1).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Matriz de Correlación de Características')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png')

    # Reducción de dimensionalidad con PCA
    print("Aplicando PCA...")
    X = df.drop(['is_fall', 'description'], axis=1).values
    y = df['is_fall'].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Es Caída')
    plt.title('PCA - Reducción a 2 Dimensiones')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/pca_visualization.png')

    # Reducción de dimensionalidad con t-SNE
    print("Aplicando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Es Caída')
    plt.title('t-SNE - Reducción a 2 Dimensiones')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/tsne_visualization.png')

    # Visualizar características más importantes
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importance = rf.feature_importances_
    feature_names = df.drop(['is_fall', 'description'], axis=1).columns

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title('15 Características Más Importantes')
    plt.tight_layout()
    plt.savefig('results/feature_importance_top15.png')

if __name__ == "__main__":
    main()