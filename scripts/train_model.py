import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Crear directorios para el proyecto
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def extract_wrist_sensor_data(zip_file_path):
    """
    Extrae datos del sensor de muñeca del dataset UMAFall.

    Args:
        zip_file_path: Ruta al archivo ZIP del dataset

    Returns:
        Datos del sensor de muñeca, etiquetas y metadatos
    """
    wrist_data = []
    labels = []
    metadata = []

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_list = zip_ref.namelist()

        for file_name in file_list:
            if file_name.endswith('.csv'):
                try:
                    with zip_ref.open(file_name) as file:
                        content = file.read().decode('utf-8')
                        lines = content.split('\n')

                        # Extraer metadatos
                        file_metadata = {}
                        for i, line in enumerate(lines[:15]):
                            try:
                                if "ID:" in line:
                                    file_metadata['activity_id'] = line.split("ID:")[1].strip()
                                elif "Age:" in line:
                                    age_str = line.split("Age:")[1].strip()
                                    # Limpiar la cadena y extraer solo los dígitos
                                    age_digits = ''.join(c for c in age_str if c.isdigit())
                                    if age_digits:
                                        file_metadata['age'] = int(age_digits)
                                    else:
                                        file_metadata['age'] = 0  # Valor predeterminado
                                elif "Height(cm):" in line:
                                    height_str = line.split("Height(cm):")[1].strip()
                                    # Limpiar y convertir
                                    height_digits = ''.join(c for c in height_str if c.isdigit() or c == '.')
                                    if height_digits:
                                        file_metadata['height'] = float(height_digits)
                                    else:
                                        file_metadata['height'] = 0.0  # Valor predeterminado
                                elif "Weight(Kg):" in line:
                                    weight_str = line.split("Weight(Kg):")[1].strip()
                                    # Limpiar y convertir
                                    weight_digits = ''.join(c for c in weight_str if c.isdigit() or c == '.')
                                    if weight_digits:
                                        file_metadata['weight'] = float(weight_digits)
                                    else:
                                        file_metadata['weight'] = 0.0  # Valor predeterminado
                                elif "Gender:" in line:
                                    file_metadata['gender'] = line.split("Gender:")[1].strip()
                                elif "Type of Movement: ADL" in line:
                                    file_metadata['movement_type'] = "ADL"
                                elif "Type of Movement: FALSE" in line or "Type of Movement: TRUE" in line:
                                    is_fall = "TRUE" in line
                                    file_metadata['is_fall'] = is_fall
                                elif "Description of the movement:" in line:
                                    file_metadata['description'] = line.split("Description of the movement:")[1].strip()
                            except Exception as e:
                                # Si hay un error al procesar una línea de metadatos, lo ignoramos
                                continue

                        # Buscar la línea que contiene información del sensor de muñeca
                        wrist_sensor_id = None
                        for line in lines:
                            try:
                                if "WRIST" in line:
                                    parts = line.strip().split(';')
                                    if len(parts) >= 3:
                                        wrist_sensor_id = parts[1].strip()
                                    break
                            except Exception:
                                continue

                        if wrist_sensor_id is None:
                            continue

                        # Encontrar dónde comienzan los datos
                        data_start_idx = 0
                        for i, line in enumerate(lines):
                            if "TimeStamp" in line:
                                data_start_idx = i + 1
                                break

                        # Extraer datos del sensor de muñeca
                        sensor_data = []
                        for i in range(data_start_idx, len(lines)):
                            try:
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
                                        sensor_data.append([timestamp, x, y, z])
                                    except:
                                        continue
                            except Exception:
                                continue

                        if sensor_data and len(sensor_data) >= 10:  # Asegurarse de que hay suficientes datos
                            sensor_data = np.array(sensor_data)
                            wrist_data.append(sensor_data)
                            labels.append(1 if file_metadata.get('is_fall', False) else 0)
                            metadata.append(file_metadata)

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    return wrist_data, labels, metadata

def extract_features(data):
    """
    Extrae características de los datos del acelerómetro.

    Args:
        data: Lista de arrays de numpy con datos del acelerómetro

    Returns:
        Array de numpy con características extraídas
    """
    features = []

    for sample in data:
        try:
            # Extraer componentes x, y, z
            x = sample[:, 1]
            y = sample[:, 2]
            z = sample[:, 3]

            # Calcular magnitud
            magnitude = np.sqrt(x**2 + y**2 + z**2)

            # Características estadísticas para cada eje
            feature_vector = []

            # Características para el eje X
            feature_vector.extend([
                np.mean(x), np.std(x), np.min(x), np.max(x),
                np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75),
                stats.skew(x), stats.kurtosis(x)
            ])

            # Características para el eje Y
            feature_vector.extend([
                np.mean(y), np.std(y), np.min(y), np.max(y),
                np.percentile(y, 25), np.percentile(y, 50), np.percentile(y, 75),
                stats.skew(y), stats.kurtosis(y)
            ])

            # Características para el eje Z
            feature_vector.extend([
                np.mean(z), np.std(z), np.min(z), np.max(z),
                np.percentile(z, 25), np.percentile(z, 50), np.percentile(z, 75),
                stats.skew(z), stats.kurtosis(z)
            ])

            # Características para la magnitud
            feature_vector.extend([
                np.mean(magnitude), np.std(magnitude), np.min(magnitude), np.max(magnitude),
                np.percentile(magnitude, 25), np.percentile(magnitude, 50), np.percentile(magnitude, 75),
                stats.skew(magnitude), stats.kurtosis(magnitude)
            ])

            # Correlaciones entre ejes
            feature_vector.extend([
                np.corrcoef(x, y)[0, 1],
                np.corrcoef(x, z)[0, 1],
                np.corrcoef(y, z)[0, 1]
            ])

            features.append(feature_vector)
        except Exception as e:
            print(f"Error al extraer características: {e}")
            continue

    return np.array(features)

def main():
    # Extraer datos del sensor de muñeca
    print("Extrayendo datos del sensor de muñeca...")
    wrist_data, labels, metadata = extract_wrist_sensor_data("UMAFall_Dataset.zip")
    print(f"Datos extraídos: {len(wrist_data)} muestras")
    
    if len(wrist_data) == 0:
        print("No se pudieron extraer datos. Verifique el archivo ZIP.")
        return
    
    print(f"Distribución de etiquetas: {sum(labels)} caídas, {len(labels) - sum(labels)} no caídas")

    # Extraer características
    print("Extrayendo características...")
    X = extract_features(wrist_data)
    
    if len(X) == 0:
        print("No se pudieron extraer características. Verifique los datos.")
        return
        
    y = np.array(labels[:len(X)])  # Asegurarse de que X e y tengan la misma longitud

    # Imprimir información sobre las características
    print(f"Número de características extraídas: {X.shape[1]}")
    feature_names = [
        'mean_x', 'std_x', 'min_x', 'max_x', 'q1_x', 'median_x', 'q3_x', 'skew_x', 'kurtosis_x',
        'mean_y', 'std_y', 'min_y', 'max_y', 'q1_y', 'median_y', 'q3_y', 'skew_y', 'kurtosis_y',
        'mean_z', 'std_z', 'min_z', 'max_z', 'q1_z', 'median_z', 'q3_z', 'skew_z', 'kurtosis_z',
        'mean_mag', 'std_mag', 'min_mag', 'max_mag', 'q1_mag', 'median_mag', 'q3_mag', 'skew_mag', 'kurtosis_mag',
        'corr_xy', 'corr_xz', 'corr_yz'
    ]

    # Guardar características y etiquetas
    try:
        features_df = pd.DataFrame(X, columns=feature_names)
        features_df['is_fall'] = y
        features_df.to_csv('data/features.csv', index=False)
        print(f"Características guardadas en 'data/features.csv'")
    except Exception as e:
        print(f"Error al guardar características: {e}")

    try:
        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Entrenar modelo de Random Forest
        print("Entrenando modelo Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluar modelo
        y_pred = rf_model.predict(X_test)
        print("Informe de clasificación:")
        print(classification_report(y_test, y_pred))

        # Visualizar matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusión')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['No Caída', 'Caída'], rotation=45)
        plt.yticks(tick_marks, ['No Caída', 'Caída'])
        plt.tight_layout()
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')

        # Añadir valores a la matriz
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.savefig('results/confusion_matrix.png')
        print("Matriz de confusión guardada en 'results/confusion_matrix.png'")

        # Visualizar importancia de características
        feature_importance = rf_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Importancia de Características')
        plt.title('Importancia de Características en Random Forest')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        print("Importancia de características guardada en 'results/feature_importance.png'")

        # Guardar modelo
        joblib.dump(rf_model, 'models/random_forest_model.pkl')
        print("Modelo guardado en 'models/random_forest_model.pkl'")
    
    except Exception as e:
        print(f"Error durante el entrenamiento o evaluación: {e}")

if __name__ == "__main__":
    main()