import zipfile
import re
import pandas as pd
import os

def extract_wrist_data(file_content, file_name):
    lines = file_content.split('\n')
    metadata = {'file': file_name}

    # Detectar tipo de actividad y subactividad
    if "_Fall_" in file_name:
        metadata['activity_type'] = "Fall"
        metadata['label'] = 1
        fall_types = {
            'backwardFall': 'Fall_backwardFall',
            'forwardFall': 'Fall_forwardFall',
            'lateralFall': 'Fall_lateralFall'
        }
        metadata['subactivity'] = next((fall_types[ft] for ft in fall_types if ft in file_name), "Unknown")
    elif "_ADL_" in file_name:
        metadata['activity_type'] = "ADL"
        metadata['label'] = 0
        adl_types = {
            'Aplausing': 'ADL_Aplausing',
            'Bending': 'ADL_Bending',
            'GoDownstairs': 'ADL_GoDownstairs',
            'GoUpstairs': 'ADL_GoUpstairs',
            'HandsUp': 'ADL_HandsUp',
            'Hopping': 'ADL_Hopping',
            'Jogging': 'ADL_Jogging',
            'LyingDown': 'ADL_LyingDown',
            'MakingACall': 'ADL_MakingACall',
            'OpeningDoor': 'ADL_OpeningDoor',
            'Sitting': 'ADL_Sitting',
            'Walking': 'ADL_Walking'
        }
        metadata['subactivity'] = next((adl_types[adl] for adl in adl_types if adl in file_name), "Unknown")
    else:
        metadata['activity_type'] = "Unknown"
        metadata['label'] = None
        metadata['subactivity'] = "Unknown"

    # Extraer metadatos personales
    for line in lines:
        if "Age:" in line:
            match = re.search(r'Age:\s*(\d+)', line)
            if match:
                metadata['age'] = int(match.group(1))
        if "Gender:" in line:
            match = re.search(r'Gender:\s*([MF])', line)
            if match:
                metadata['gender'] = match.group(1).strip()
        if "Height(cm):" in line:
            match = re.search(r'Height\(cm\):\s*(\d+)', line)
            if match:
                metadata['height'] = int(match.group(1))
        if "Weight(Kg):" in line:
            match = re.search(r'Weight\(Kg\):\s*(\d+)', line)
            if match:
                metadata['weight'] = int(match.group(1))

    # Identificar sensor muñeca
    wrist_sensor_id = None
    for line in lines:
        if "WRIST" in line and ";" in line:
            parts = line.split(';')
            if len(parts) >= 3:
                wrist_sensor_id = parts[1].strip()
                break
    if wrist_sensor_id is None:
        return []

    # Extraer datos del sensor muñeca
    wrist_data = []
    for line in lines:
        if re.match(r'^\d+;\d+;', line):
            parts = line.split(';')
            if len(parts) >= 7 and parts[1].strip() == wrist_sensor_id:
                try:
                    timestamp = float(parts[0])
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    row = metadata.copy()
                    row.update({'timestamp': timestamp, 'x': x, 'y': y, 'z': z})
                    wrist_data.append(row)
                except:
                    pass
    return wrist_data

def process_all_files(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        print(f"Procesando {len(all_files)} archivos...")

        all_wrist_data = []
        for i, file in enumerate(all_files):
            if i % 50 == 0:
                print(f"Progreso: {i}/{len(all_files)} archivos")
            try:
                with zip_ref.open(file) as f:
                    content = f.read().decode('utf-8')
                    wrist_data = extract_wrist_data(content, file)
                    if wrist_data:
                        all_wrist_data.extend(wrist_data)
            except Exception as e:
                print(f"Error procesando {file}: {e}")

        if all_wrist_data:
            df = pd.DataFrame(all_wrist_data)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Datos extraídos guardados en: {output_path}")
            return df
        else:
            print("No se encontraron datos del sensor de muñeca.")
            return None

if __name__ == "__main__":
    ZIP_PATH = "UMAFall_Dataset.zip"
    OUTPUT_PATH = "data/wrist_filtered/wrist_data_corrected.csv"
    process_all_files(ZIP_PATH, OUTPUT_PATH)