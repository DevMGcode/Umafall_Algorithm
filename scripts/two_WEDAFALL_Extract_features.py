#two_WEDAFALL_Extract_features.py

import os
import zipfile
import pandas as pd
import numpy as np
import shutil
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Diccionario con datos demográficos basado en README
subject_info = {
    "U01": {"Age": 22, "Height": 176, "Weight": 56.3, "Gender": "M"},
    "U02": {"Age": 22, "Height": 178, "Weight": 56.0, "Gender": "M"},
    "U03": {"Age": 20, "Height": 173, "Weight": 69.5, "Gender": "M"},
    "U04": {"Age": 21, "Height": 170, "Weight": 57.1, "Gender": "F"},
    "U05": {"Age": 23, "Height": 167, "Weight": 59.6, "Gender": "M"},
    "U06": {"Age": 22, "Height": 167, "Weight": 69.0, "Gender": "M"},
    "U07": {"Age": 21, "Height": 178, "Weight": 68.1, "Gender": "M"},
    "U08": {"Age": 23, "Height": 162, "Weight": 61.0, "Gender": "F"},
    "U09": {"Age": 22, "Height": 170, "Weight": 52.0, "Gender": "F"},
    "U10": {"Age": 23, "Height": 183, "Weight": 77.0, "Gender": "M"},
    "U11": {"Age": 23, "Height": 169, "Weight": 61.8, "Gender": "F"},
    "U12": {"Age": 23, "Height": 178, "Weight": 64.5, "Gender": "F"},
    "U13": {"Age": 22, "Height": 179, "Weight": 66.0, "Gender": "M"},
    "U14": {"Age": 46, "Height": 184, "Weight": 83.0, "Gender": "M"},
    "U21": {"Age": 95, "Height": 170, "Weight": 71.0, "Gender": "M"},
    "U22": {"Age": 85, "Height": 153, "Weight": 62.0, "Gender": "F"},
    "U23": {"Age": 82, "Height": 160, "Weight": 60.0, "Gender": "F"},
    "U24": {"Age": 81, "Height": 152, "Weight": 63.0, "Gender": "F"},
    "U25": {"Age": 81, "Height": 173, "Weight": 72.0, "Gender": "F"},
    "U26": {"Age": 83, "Height": 175, "Weight": 85.0, "Gender": "M"},
    "U27": {"Age": 89, "Height": 171, "Weight": 71.5, "Gender": "M"},
    "U28": {"Age": 88, "Height": 157, "Weight": 52.5, "Gender": "F"},
    "U29": {"Age": 77, "Height": 160, "Weight": 65.9, "Gender": "F"},
    "U30": {"Age": 80, "Height": 179, "Weight": 72.0, "Gender": "M"},
    "U31": {"Age": 88, "Height": 163, "Weight": 53.0, "Gender": "F"},
}

fall_mapping = {
    "F01": "Fall_forwardFall",
    "F04": "Fall_forwardFall",
    "F06": "Fall_forwardFall",
    "F02": "Fall_lateralFall",
    "F08": "Fall_lateralFall",
    "F03": "Fall_backwardFall",
    "F05": "Fall_backwardFall",
    "F07": "Fall_backwardFall"
}

adl_mapping = {
    "D01": "ADL_Walking",
    "D02": "ADL_Jogging",
    "D03": "ADL_GoUpstairs",
    "D04": "ADL_Sitting",
    "D06": "ADL_Bending",
    "D08": "ADL_Hopping",
    "D10": "ADL_Aplausing",
    "D11": "ADL_OpeningDoor"
}

def extract_and_group_weda(zip_file, extract_dir, output_dir, fall_timestamps_file):
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Archivo {zip_file} extraído en {extract_dir}")

    fall_timestamps = pd.read_csv(fall_timestamps_file)

    os.makedirs(output_dir, exist_ok=True)

    for activity_code in os.listdir(extract_dir):
        activity_path = os.path.join(extract_dir, activity_code)
        if not os.path.isdir(activity_path):
            continue

        if activity_code in fall_mapping:
            target_folder = fall_mapping[activity_code]
        elif activity_code in adl_mapping:
            target_folder = adl_mapping[activity_code]
        else:
            print(f"Omitido: {activity_code}")
            continue

        target_path = os.path.join(output_dir, target_folder)
        os.makedirs(target_path, exist_ok=True)

        for filename in os.listdir(activity_path):
            if not filename.endswith("_accel.csv"):
                continue

            src_file = os.path.join(activity_path, filename)

            if activity_code in fall_mapping:
                key = f"{activity_code}/{filename.replace('_accel.csv','')}"
                row = fall_timestamps[fall_timestamps['filename'] == key]

                if row.empty:
                    shutil.copyfile(src_file, os.path.join(target_path, f"{activity_code}_{filename}"))
                    continue

                start_time = row['start_time'].values[0]
                end_time = row['end_time'].values[0]

                df = pd.read_csv(src_file)
                if 'accel_time_list' in df.columns:
                    df_window = df[(df['accel_time_list'] >= start_time) & (df['accel_time_list'] <= end_time)]
                    if df_window.empty:
                        shutil.copyfile(src_file, os.path.join(target_path, f"{activity_code}_{filename}"))
                    else:
                        dst_file = os.path.join(target_path, f"{activity_code}_{filename}")
                        df_window.to_csv(dst_file, index=False)
                else:
                    shutil.copyfile(src_file, os.path.join(target_path, f"{activity_code}_{filename}"))
            else:
                dst_file = os.path.join(target_path, f"{activity_code}_{filename}")
                shutil.copyfile(src_file, dst_file)

    print("✔️ Archivos agrupados y recortados exitosamente.")

def safe_correlation(x, y):
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return x.corr(y)

def calculate_features(df):
    features = {}
    for col in ['X', 'Y', 'Z', 'N_XYZ', 'N_HOR', 'N_VER']:
        features[f'var_{col}'] = df[col].var()
        features[f'mean_{col}'] = df[col].mean()
        features[f'std_{col}'] = df[col].std()
        features[f'max_{col}'] = df[col].max()
        features[f'min_{col}'] = df[col].min()
        features[f'range_{col}'] = features[f'max_{col}'] - features[f'min_{col}']

    features['corr_XY'] = safe_correlation(df['X'], df['Y'])
    features['corr_XZ'] = safe_correlation(df['X'], df['Z'])
    features['corr_YZ'] = safe_correlation(df['Y'], df['Z'])
    features['corr_NV'] = safe_correlation(df['N_XYZ'], df['N_VER'])
    features['corr_NH'] = safe_correlation(df['N_XYZ'], df['N_HOR'])
    features['corr_HV'] = safe_correlation(df['N_HOR'], df['N_VER'])
    return features

def process_trial_file(args):
    trial_file_name, window_sizes, base_dir = args
    try:
        file_path = os.path.join(base_dir, trial_file_name)
        df = pd.read_csv(file_path)

        # Extraer Description_of_movement del nombre de la carpeta padre
        parent_folder = os.path.normpath(trial_file_name).split(os.sep)[0]
        if '_' in parent_folder:
            description = parent_folder.split('_', 1)[1]
        else:
            description = ""

        # Extraer ID del sujeto: buscar el patrón después del segundo guion bajo
        base_name = os.path.basename(trial_file_name)
        parts = base_name.split('_')
        if len(parts) > 2:
            subject_id = parts[1]
        else:
            subject_id = None

        # Obtener datos demográficos del sujeto
        if subject_id and subject_id in subject_info:
            age = subject_info[subject_id]["Age"]
            height = subject_info[subject_id]["Height"]
            weight = subject_info[subject_id]["Weight"]
            gender = subject_info[subject_id]["Gender"]
        else:
            age = height = weight = gender = ""

        # Extraer columnas X,Y,Z
        if all(c in df.columns for c in ['accel_x_list', 'accel_y_list', 'accel_z_list']):
            df_Mediciones = df[['accel_x_list', 'accel_y_list', 'accel_z_list']].copy()
            df_Mediciones.columns = ['X', 'Y', 'Z']
        else:
            return []

        df_Mediciones['N_XYZ'] = np.sqrt(df_Mediciones['X']**2 + df_Mediciones['Y']**2 + df_Mediciones['Z']**2)
        max_N_index = df_Mediciones['N_XYZ'].idxmax()

        all_results = []
        sf = 50

        if trial_file_name.startswith("ADL"):
            type_of_movement = "ADL"
            act = "D"
        elif trial_file_name.startswith("Fall"):
            type_of_movement = "FALL"
            act = "F"
        else:
            type_of_movement = "UNKNOWN"
            act = "U"

        feature_order = [
            'var_X', 'mean_X', 'std_X', 'max_X', 'min_X', 'range_X',
            'var_Y', 'mean_Y', 'std_Y', 'max_Y', 'min_Y', 'range_Y',
            'var_Z', 'mean_Z', 'std_Z', 'max_Z', 'min_Z', 'range_Z',
            'var_N_XYZ', 'mean_N_XYZ', 'std_N_XYZ', 'max_N_XYZ', 'min_N_XYZ', 'range_N_XYZ',
            'var_N_HOR', 'mean_N_HOR', 'std_N_HOR', 'max_N_HOR', 'min_N_HOR', 'range_N_HOR',
            'var_N_VER', 'mean_N_VER', 'std_N_VER', 'max_N_VER', 'min_N_VER', 'range_N_VER',
            'corr_XY', 'corr_XZ', 'corr_YZ', 'corr_NV', 'corr_NH', 'corr_HV'
        ]

        for window_size in window_sizes:
            ws_samples = int(window_size * sf)
            if len(df_Mediciones) < ws_samples:
                continue

            start_idx = max(0, max_N_index - ws_samples // 2)
            end_idx = min(len(df_Mediciones), start_idx + ws_samples)
            df_window = df_Mediciones.iloc[start_idx:end_idx].copy()

            df_window['N_HOR'] = np.sqrt(df_window['Y']**2 + df_window['Z']**2)
            df_window['N_VER'] = np.sqrt(df_window['X']**2 + df_window['Z']**2)

            features = calculate_features(df_window)

            result_parts = [
                trial_file_name,
                act,
                act,
                type_of_movement,
                description,
                age,
                height,
                weight,
                gender
            ]

            # Convertir todos a string para evitar error en join
            result_parts = [str(x) for x in result_parts]

            for k in feature_order:
                val = features.get(k, 0)
                if pd.isna(val):
                    val = 0
                result_parts.append(str(val))

            result = ",".join(result_parts)
            all_results.append((window_size, result))

        return all_results
    except Exception as e:
        print(f"Error procesando {trial_file_name}: {e}")
        return []

def process_files_in_batches(file_names, window_sizes, base_dir, batch_size=50, max_workers=None):
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    all_results = {size: [] for size in window_sizes}

    with tqdm(total=len(file_names), desc="Procesando archivos") as pbar:
        for batch_start in range(0, len(file_names), batch_size):
            batch_end = min(batch_start + batch_size, len(file_names))
            batch_files = file_names[batch_start:batch_end]

            args_list = [(file_name, window_sizes, base_dir) for file_name in batch_files]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(process_trial_file, args_list))

            for file_results in batch_results:
                for window_size, result in file_results:
                    all_results[window_size].append(result)

            pbar.update(len(batch_files))

    return all_results

def main():
    zip_file = "datasets/50Hz.zip"
    extract_dir = "datasets/weda_fall_extracted"
    grouped_dir = "datasets/weda_fall_grouped"
    fall_timestamps_file = "datasets/fall_timestamps.csv"
    window_sizes = [1.5, 2, 3]

    extract_and_group_weda(zip_file, extract_dir, grouped_dir, fall_timestamps_file)

    file_names = []
    for root, _, files in os.walk(grouped_dir):
        for file in files:
            if file.endswith('.csv'):
                rel_path = os.path.relpath(os.path.join(root, file), grouped_dir)
                file_names.append(rel_path)

    print(f"Archivos agrupados para procesamiento: {len(file_names)}")

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

    # Crear carpeta de salida si no existe
    output_dir = "unified_WEDAFALL_dataset"
    os.makedirs(output_dir, exist_ok=True)

    results_by_window = process_files_in_batches(file_names, window_sizes, grouped_dir)

    for window_size in window_sizes:
        results = results_by_window[window_size]
        if not results:
            print(f"No se generaron resultados para ventana {window_size}s")
            continue

        output_file = os.path.join(output_dir, f"WEDAFall_window_{window_size}s_all.csv")
        with open(output_file, "w") as f:
            f.write(",".join(headers) + "\n")
            for line in results:
                f.write(line + "\n")

        print(f"Archivo generado: {output_file} con {len(results)} registros")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()