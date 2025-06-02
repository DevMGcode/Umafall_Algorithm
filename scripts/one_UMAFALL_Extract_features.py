import zipfile
import pandas as pd
import numpy as np
import os
import re
from os import walk
import time
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Para mostrar barras de progreso

# Suprimir advertencias
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def extract_metadata(file_path):
    """Extrae metadatos del archivo de manera optimizada"""
    metadata = {
        "Type_of_Movement": "", "Description_of_movement": "",
        "Age": "", "Height": "", "Weight": "", "Gender": ""
    }

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Leer solo las primeras 20 líneas para metadatos
            lines = [next(f) for _ in range(20) if f]

            for line in lines:
                if "Type of Movement" in line:
                    match = re.search(r'Type of Movement:\s*(\w+)', line)
                    if match: metadata["Type_of_Movement"] = match.group(1)
                elif "Description" in line and "movement" in line:
                    metadata["Description_of_movement"] = line.split(':')[-1].strip()
                elif "Age" in line:
                    match = re.search(r'Age:\s*(\d+)', line)
                    if match: metadata["Age"] = match.group(1)
                elif "Height" in line:
                    match = re.search(r'Height\(cm\):\s*(\d+)', line)
                    if match: metadata["Height"] = match.group(1)
                elif "Weight" in line:
                    match = re.search(r'Weight\(Kg\):\s*(\d+)', line)
                    if match: metadata["Weight"] = match.group(1)
                elif "Gender" in line:
                    match = re.search(r'Gender:\s*(\w+)', line)
                    if match: metadata["Gender"] = match.group(1)
    except Exception:
        pass  # Ignorar errores en metadatos

    return metadata

def find_data_start(file_path):
    """Encuentra la línea donde comienzan los datos"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if "TimeStamp" in line or "SampleNo" in line:
                    return i
    except:
        pass
    return 0

def safe_correlation(x, y):
    """Calcula correlación de manera segura"""
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return x.corr(y)

def calculate_features(df):
    """Calcula características estadísticas para cada columna del DataFrame"""
    features = {}

    # Calcular estadísticas para cada eje y norma
    for col in ['X', 'Y', 'Z', 'N_XYZ', 'N_HOR', 'N_VER']:
        features[f'var_{col}'] = df[col].var()
        features[f'mean_{col}'] = df[col].mean()
        features[f'std_{col}'] = df[col].std()
        features[f'max_{col}'] = df[col].max()
        features[f'min_{col}'] = df[col].min()
        features[f'range_{col}'] = features[f'max_{col}'] - features[f'min_{col}']

    # Calcular correlaciones de manera segura
    features['corr_XY'] = safe_correlation(df['X'], df['Y'])
    features['corr_XZ'] = safe_correlation(df['X'], df['Z'])
    features['corr_YZ'] = safe_correlation(df['Y'], df['Z'])
    features['corr_NV'] = safe_correlation(df['N_XYZ'], df['N_VER'])
    features['corr_NH'] = safe_correlation(df['N_XYZ'], df['N_HOR'])
    features['corr_HV'] = safe_correlation(df['N_HOR'], df['N_VER'])

    return features

def process_trial_file(args):
    """
    Procesa un archivo y extrae características basadas en una ventana centrada
    en el valor máximo de la norma euclidiana de aceleración.
    """
    trial_file_name, window_sizes, uma_directory = args

    try:
        file_path = os.path.join(uma_directory, trial_file_name)

        # Extraer metadatos
        metadata = extract_metadata(file_path)

        # Encontrar línea de inicio de datos
        data_start_line = find_data_start(file_path)

        # Leer datos
        try:
            df_Mediciones_all = pd.read_csv(file_path, header=data_start_line, sep=';',
                                          encoding='utf-8', on_bad_lines='skip')
        except:
            try:
                df_Mediciones_all = pd.read_csv(file_path, header=data_start_line, sep=';',
                                              on_bad_lines='skip')
            except Exception:
                return []

        # Filtrar datos si es necesario
        if 'Type' in df_Mediciones_all.columns and 'sensor_ID' in df_Mediciones_all.columns:
            df_Mediciones_file = df_Mediciones_all.loc[(df_Mediciones_all['Type']==0) &
                                                     (df_Mediciones_all['sensor_ID']== 3)]
        else:
            df_Mediciones_file = df_Mediciones_all

        # Extraer columnas X, Y, Z
        if 'X' in df_Mediciones_file.columns and 'Y' in df_Mediciones_file.columns and 'Z' in df_Mediciones_file.columns:
            df_Mediciones = df_Mediciones_file[["X", "Y", "Z"]].copy()
        else:
            numeric_cols = df_Mediciones_file.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                df_Mediciones = df_Mediciones_file[numeric_cols[:3]].copy()
                df_Mediciones.columns = ["X", "Y", "Z"]
            else:
                return []

        # Determinar tipo de actividad y Type_of_Movement corregido
        if '_Fall_' in trial_file_name:
            act = "F"
            type_of_movement = "FALL"
        elif '_ADL_' in trial_file_name:
            act = "D"
            type_of_movement = "ADL"
        else:
            act = "U"
            type_of_movement = "UNKNOWN"

        # Calcular la norma euclidiana de aceleración XYZ
        df_Mediciones['N_XYZ'] = np.sqrt(df_Mediciones['X']**2 +
                                       df_Mediciones['Y']**2 +
                                       df_Mediciones['Z']**2)

        # Encontrar el valor máximo y su índice
        max_N = df_Mediciones['N_XYZ'].max()
        max_N_index = df_Mediciones['N_XYZ'].idxmax()

        all_results = []

        for window_size in window_sizes:
            sf = 50# Frecuencia original del dataset
            ws_samples = int(window_size * sf)

            if len(df_Mediciones) < ws_samples:
                continue

            if (max_N_index - round(ws_samples/2) < 0):
                df_window = df_Mediciones.iloc[0:min(ws_samples, len(df_Mediciones))].copy()
            elif (max_N_index + round(ws_samples/2) + 1 > len(df_Mediciones)):
                start_idx = max(0, len(df_Mediciones) - ws_samples)
                df_window = df_Mediciones.iloc[start_idx:len(df_Mediciones)].copy()
            else:
                start_idx = max(0, max_N_index - round(ws_samples/2))
                end_idx = min(len(df_Mediciones), max_N_index + round(ws_samples/2) + 1)
                df_window = df_Mediciones.iloc[start_idx:end_idx].copy()

            df_window['N_HOR'] = np.sqrt(df_window['Y']**2 + df_window['Z']**2)
            df_window['N_VER'] = np.sqrt(df_window['X']**2 + df_window['Z']**2)

            features = calculate_features(df_window)

            result_parts = [
                trial_file_name,  # Solo nombre del archivo, sin índice
                act,             # F=Fall, D=ADL
                act,             # Repetido para compatibilidad
                type_of_movement,
                metadata['Description_of_movement'],
                metadata['Age'],
                metadata['Height'],
                metadata['Weight'],
                metadata['Gender']
            ]

            for feature_name, feature_value in features.items():
                if pd.isna(feature_value):
                    feature_value = 0
                result_parts.append(str(feature_value))

            result = ",".join(result_parts)
            all_results.append((window_size, result))

        return all_results

    except Exception:
        return []

def process_files_in_batches(file_names, window_sizes, uma_directory, batch_size=50, max_workers=None):
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    all_results = {size: [] for size in window_sizes}

    with tqdm(total=len(file_names), desc="Procesando archivos") as pbar:
        for batch_start in range(0, len(file_names), batch_size):
            batch_end = min(batch_start + batch_size, len(file_names))
            batch_files = file_names[batch_start:batch_end]

            args_list = [(file_name, window_sizes, uma_directory) for file_name in batch_files]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(process_trial_file, args_list))

            for file_results in batch_results:
                for window_size, result in file_results:
                    all_results[window_size].append(result)

            pbar.update(len(batch_files))

    return all_results

def main():
    start_time = time.time()

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))  # carpeta scripts
    parent_dir = os.path.dirname(base_dir)  # carpeta AlgoritmoUmaFall
    datasets_dir = os.path.join(parent_dir, "datasets")
    zip_path = os.path.join(datasets_dir, "UMAFall_Dataset.zip")

    output_dir = os.path.join(base_dir, "unified_UMAFALL_dataset")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio '{output_dir}' creado.")

    extracted_dir = os.path.join(base_dir, "extracted_files")
    if not os.path.exists(extracted_dir):
        print(f"Descomprimiendo archivo ZIP desde {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        print("Descompresión completada.")

    uma_directory = extracted_dir

    file_names = []
    for root, _, files in os.walk(uma_directory):
        for file in files:
            if file.endswith('.csv'):
                file_names.append(file)

    print(f"Se encontraron {len(file_names)} archivos CSV.")

    window_sizes = [1.5, 2, 3]

    headers = [
        "FileName", "Fall_ADL", "Fall_ADL2", "Type_of_Movement",
        "Description_of_movement", "Age", "Height", "Weight", "Gender",
        "var_X", "mean_X", "std_X", "max_X", "min_X", "range_X",
        "var_Y", "mean_Y", "std_Y", "max_Y", "min_Y", "range_Y",
        "var_Z", "mean_Z", "std_Z", "max_Z", "min_Z", "range_Z",
        "var_N_XYZ", "mean_N_XYZ", "std_N_XYZ", "max_N_XYZ", "min_N_XYZ", "range_N_XYZ",
        "var_N_HOR", "mean_N_HOR", "std_N_HOR", "max_N_HOR", "min_N_HOR", "range_N_HOR",
        "var_N_VER", "mean_N_VER", "std_N_VER", "max_N_VER", "min_N_VER", "range_N_VER",
        "corr_XY", "corr_XZ", "corr_YZ", "corr_NV", "corr_NH", "corr_HV"
    ]

    print("\nIniciando procesamiento de ventanas temporales...")

    batch_size = 50
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Usando {max_workers} procesos para el procesamiento paralelo")

    results_by_window = process_files_in_batches(
        file_names, window_sizes, uma_directory,
        batch_size=batch_size, max_workers=max_workers
    )

    total_windows = 0

    for window_size in window_sizes:
        results = results_by_window[window_size]

        if not results:
            print(f"No se generaron resultados para ventana de {window_size}s")
            continue

        output_file = os.path.join(output_dir, f"UMAFall_window_{window_size}s_all.csv")

        with open(output_file, "w") as f:
            f.write(",".join(headers) + "\n")
            for result in results:
                f.write(result + "\n")

        print(f"Archivo para ventana de {window_size}s guardado: {output_file}")
        print(f"Contiene {len(results)} ventanas de datos.")

        fall_count = sum(1 for result in results if result.split(',')[1] == "F")
        adl_count = sum(1 for result in results if result.split(',')[1] == "D")

        if len(results) > 0:
            print(f"Distribución de tipos de actividad:")
            print(f"- Caídas (F): {fall_count} ({fall_count/len(results)*100:.1f}%)")
            print(f"- Actividades diarias (D): {adl_count} ({adl_count/len(results)*100:.1f}%)")

        total_windows += len(results)

    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)

    print(f"\nProcesamiento completado en {minutes} minutos y {seconds} segundos.")
    print(f"Se han procesado {len(file_names)} archivos.")
    print(f"Se han generado {total_windows} ventanas de datos en total.")
    print(f"Los resultados se han guardado en la carpeta '{output_dir}'.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()