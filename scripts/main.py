import numpy as np
import matplotlib.pyplot as plt
from fall_detector import FallDetector
import zipfile
import os

def load_wrist_sensor_data(zip_file_path, sample_index=0):
    """
    Carga una muestra de datos del sensor de muñeca del dataset UMAFall.

    Args:
        zip_file_path: Ruta al archivo ZIP del dataset
        sample_index: Índice de la muestra a cargar

    Returns:
        Array de numpy con los datos del acelerómetro y metadatos
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_list = [f for f in zip_ref.namelist() if f.endswith('.csv')]

        if sample_index >= len(file_list):
            sample_index = 0

        file_name = file_list[sample_index]

        with zip_ref.open(file_name) as file:
            content = file.read().decode('utf-8')
            lines = content.split('\n')

            # Extraer metadatos
            metadata = {}
            for i, line in enumerate(lines[:15]):
                if "ID:" in line:
                    metadata['activity_id'] = line.split("ID:")[1].strip()
                elif "Type of Movement: FALSE" in line or "Type of Movement: TRUE" in line:
                    is_fall = "TRUE" in line
                    metadata['is_fall'] = is_fall
                elif "Description of the movement:" in line:
                    metadata['description'] = line.split("Description of the movement:")[1].strip()

            # Buscar la línea que contiene información del sensor de muñeca
            wrist_sensor_id = None
            for line in lines:
                if "WRIST" in line:
                    parts = line.strip().split(';')
                    if len(parts) >= 3:
                        wrist_sensor_id = parts[1].strip()
                    break

            if wrist_sensor_id is None:
                return np.array([]), metadata, file_name

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

            return np.array(data), metadata, file_name

def main():
    # Inicializar detector de caídas
    detector = FallDetector()

    # Cargar una muestra de datos
    data, metadata, file_name = load_wrist_sensor_data("UMAFall_Dataset.zip")

    if len(data) == 0:
        print("No se pudieron cargar datos del sensor de muñeca para este archivo.")
        return

    # Detectar caída
    is_fall = detector.detect_fall(data)
    fall_probability = detector.get_fall_probability(data)

    # Mostrar resultados
    print(f"Archivo: {file_name}")
    print(f"Descripción: {metadata.get('description', 'No disponible')}")
    print(f"Es caída (real): {metadata.get('is_fall', 'No disponible')}")
    print(f"Es caída (predicción): {is_fall}")
    print(f"Probabilidad de caída: {fall_probability:.2f}")

    # Visualizar datos
    plt.figure(figsize=(12, 8))

    # Graficar aceleración en los tres ejes
    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0] - data[0, 0], data[:, 1], 'r-', label='X')
    plt.plot(data[:, 0] - data[0, 0], data[:, 2], 'g-', label='Y')
    plt.plot(data[:, 0] - data[0, 0], data[:, 3], 'b-', label='Z')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración')
    plt.title(f"Datos del acelerómetro - {metadata.get('description', 'No disponible')}")
    plt.legend()
    plt.grid(True)

    # Graficar magnitud
    plt.subplot(2, 1, 2)
    magnitude = np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2)
    plt.plot(data[:, 0] - data[0, 0], magnitude, 'k-')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Magnitud')
    plt.title(f"Magnitud de la aceleración - Predicción: {'Caída' if is_fall else 'No caída'} ({fall_probability:.2f})")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/sample_visualization.png')
    plt.show()

if __name__ == "__main__":
    main()