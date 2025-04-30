import numpy as np
import matplotlib.pyplot as plt
from fall_detector import FallDetector
import zipfile
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def extract_wrist_test_data(zip_file_path, max_samples=100):
    """
    Extrae datos de prueba del sensor de muñeca del dataset UMAFall.

    Args:
        zip_file_path: Ruta al archivo ZIP del dataset
        max_samples: Número máximo de muestras a extraer

    Returns:
        Lista de arrays de numpy con los datos del acelerómetro y lista de etiquetas
    """
    test_data = []
    labels = []
    file_names = []

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_list = [f for f in zip_ref.namelist() if f.endswith('.csv')]

        # Limitar el número de muestras
        if max_samples < len(file_list):
            file_list = file_list[:max_samples]

        for file_name in file_list:
            try:
                with zip_ref.open(file_name) as file:
                    content = file.read().decode('utf-8')
                    lines = content.split('\n')

                    # Extraer si es caída
                    is_fall = False
                    for line in lines[:15]:
                        if "Type of Movement: TRUE" in line:
                            is_fall = True
                            break

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
                        test_data.append(np.array(data))
                        labels.append(1 if is_fall else 0)
                        file_names.append(file_name)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return test_data, labels, file_names

def main():
    # Inicializar detector de caídas
    detector = FallDetector()

    # Extraer datos de prueba
    print("Extrayendo datos de prueba del sensor de muñeca...")
    test_data, true_labels, file_names = extract_wrist_test_data("UMAFall_Dataset.zip")

    if not test_data:
        print("No se pudieron extraer datos de prueba del sensor de muñeca.")
        return

    print(f"Datos extraídos: {len(test_data)} muestras")

    # Realizar predicciones
    print("Realizando predicciones...")
    predictions = []
    probabilities = []

    for data in test_data:
        is_fall = detector.detect_fall(data)
        probability = detector.get_fall_probability(data)
        predictions.append(1 if is_fall else 0)
        probabilities.append(probability)

    # Evaluar resultados
    print("Informe de clasificación:")
    print(classification_report(true_labels, predictions))

    # Visualizar matriz de confusión
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Caída', 'Caída'],
                yticklabels=['No Caída', 'Caída'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig('results/test_confusion_matrix.png')

    # Visualizar distribución de probabilidades
    plt.figure(figsize=(10, 6))

    fall_probs = [prob for prob, label in zip(probabilities, true_labels) if label == 1]
    no_fall_probs = [prob for prob, label in zip(probabilities, true_labels) if label == 0]

    plt.hist(fall_probs, alpha=0.5, bins=20, label='Caídas', color='red')
    plt.hist(no_fall_probs, alpha=0.5, bins=20, label='No Caídas', color='blue')

    plt.xlabel('Probabilidad de Caída')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades de Caída')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/probability_distribution.png')

    # Mostrar ejemplos de falsos positivos y falsos negativos
    false_positives = [(i, prob) for i, (pred, true, prob) in
                       enumerate(zip(predictions, true_labels, probabilities))
                       if pred == 1 and true == 0]

    false_negatives = [(i, prob) for i, (pred, true, prob) in
                       enumerate(zip(predictions, true_labels, probabilities))
                       if pred == 0 and true == 1]

    print(f"\nFalsos Positivos: {len(false_positives)}")
    for i, (idx, prob) in enumerate(false_positives[:3]):
        print(f"  {i+1}. {file_names[idx]} (Prob: {prob:.2f})")

    print(f"\nFalsos Negativos: {len(false_negatives)}")
    for i, (idx, prob) in enumerate(false_negatives[:3]):
        print(f"  {i+1}. {file_names[idx]} (Prob: {prob:.2f})")

if __name__ == "__main__":
    main()