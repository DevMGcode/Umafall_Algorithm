import zipfile
import os
import pandas as pd
import numpy as np
import re
import math
from tqdm import tqdm
import sys

# Paths
zip_path = 'UMAFall_Dataset.zip'
extract_dir = 'UMAFall_Extracted/'
unified_dir = 'unified_dataset/'

# Extraer zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

os.makedirs(unified_dir, exist_ok=True)

def parse_metadata(file_path):
    metadata = {}
    with open(file_path, 'r') as f:
        for _ in range(50):
            line = f.readline().strip()
            if line.startswith('% Gender:'):
                metadata['Gender'] = line.split(':')[1].strip()
            elif line.startswith('% Age:'):
                try:
                    metadata['Age'] = int(re.findall(r'\d+', line)[0])
                except:
                    metadata['Age'] = None
            elif line.startswith('% Weight(Kg):'):
                try:
                    metadata['Weight'] = float(re.findall(r'\d+\.?\d*', line)[0])
                except:
                    metadata['Weight'] = None
            elif line.startswith('% Height(cm):'):
                try:
                    metadata['Height'] = float(re.findall(r'\d+\.?\d*', line)[0])
                except:
                    metadata['Height'] = None
            elif line.startswith('% Type of Movement:'):
                if 'Type_of_Movement' not in metadata:
                    metadata['Type_of_Movement'] = line.split(':')[1].strip()
            elif line.startswith('% Description of the movement:'):
                metadata['Specific_Activity'] = line.split(':')[1].strip()
    return metadata

def compute_features(df, window_size_sec):
    sf = 19
    ws_samples = int(sf * window_size_sec)

    df = df.loc[(df['Type'] == 0) & (df['sensor_ID'] == 3)].copy()
    if df.empty:
        return None

    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
    df = df.dropna(subset=['X', 'Y', 'Z'])

    if df.empty:
        return None

    fn = lambda row: math.sqrt(row.X**2 + row.Y**2 + row.Z**2)
    col = df.apply(fn, axis=1)
    df = df.assign(N_XYZ=col.values)

    max_idx = df['N_XYZ'].idxmax()
    len_df = len(df)

    if max_idx - ws_samples // 2 < 0:
        df_win = df.iloc[0:ws_samples]
    elif max_idx + ws_samples // 2 > len_df:
        df_win = df.iloc[len_df - ws_samples:len_df]
    else:
        df_win = df.iloc[max_idx - ws_samples // 2 : max_idx + ws_samples // 2]

    df_win = df_win.copy()  # Evitar SettingWithCopyWarning
    df_win['N_HOR'] = np.sqrt(df_win['Y']**2 + df_win['Z']**2)
    df_win['N_VER'] = np.sqrt(df_win['X']**2 + df_win['Z']**2)

    features = {}
    axes = ['X', 'Y', 'Z', 'N_XYZ', 'N_HOR', 'N_VER']
    for axis in axes:
        features[f'var_{axis}'] = df_win[axis].var()
        features[f'mean_{axis}'] = df_win[axis].mean()
        features[f'std_{axis}'] = df_win[axis].std()
        features[f'max_{axis}'] = df_win[axis].max()
        features[f'min_{axis}'] = df_win[axis].min()
        features[f'range_{axis}'] = features[f'max_{axis}'] - features[f'min_{axis}']
        features[f'kurtosis_{axis}'] = df_win[axis].kurtosis()
        features[f'skewness_{axis}'] = df_win[axis].skew()

    features['corr_XY'] = df_win['X'].corr(df_win['Y'])
    features['corr_XZ'] = df_win['X'].corr(df_win['Z'])
    features['corr_YZ'] = df_win['Y'].corr(df_win['Z'])
    features['corr_NV'] = df_win['N_XYZ'].corr(df_win['N_VER'])
    features['corr_NH'] = df_win['N_XYZ'].corr(df_win['N_HOR'])
    features['corr_HV'] = df_win['N_HOR'].corr(df_win['N_VER'])

    return features

def process_file(file_path, window_size_sec):
    metadata = parse_metadata(file_path)
    df = pd.read_csv(file_path, sep=';', skiprows=41,
                     names=['TimeStamp', 'SampleNo', 'X', 'Y', 'Z', 'Type', 'sensor_ID', 'Position'])

    features = compute_features(df, window_size_sec)
    if features is None:
        return None

    # Crear diccionario con File_Name primero
    features_new = {'File_Name': os.path.basename(file_path)}
    features_new.update(features)
    features_new.update(metadata)
    features_new['Fall_ADL'] = 'F' if metadata.get('Type_of_Movement', '').upper() == 'FALL' else 'D'
    features_new['Act_Type'] = metadata.get('Specific_Activity', 'Unknown')

    return features_new

def main():
    window_sizes = [1.5, 2, 3]
    os.makedirs(unified_dir, exist_ok=True)

    file_list = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.csv'):
                file_list.append(os.path.join(root, file))

    for ws in window_sizes:
        all_features = []
        print(f"\nProcesando ventana temporal de {ws} segundos...\n")
        for file_path in tqdm(file_list, desc="Archivos procesados", unit="archivo", ncols=80, file=sys.stdout):
            try:
                feat = process_file(file_path, ws)
                if feat is not None:
                    all_features.append(feat)
            except Exception as e:
                print(f"\n[Error] Archivo {os.path.basename(file_path)} no procesado: {e}", file=sys.stderr)

        if not all_features:
            print(f"No se generaron características para ventana {ws}s. Saltando archivo de salida.", file=sys.stderr)
            continue

        df_out = pd.DataFrame(all_features)

        # Reordenar columnas para que File_Name y metadata estén al inicio, excluyendo Specific_Activity
        cols_order = ['File_Name', 'Age', 'Height', 'Weight', 'Gender', 'Type_of_Movement', 'Fall_ADL', 'Act_Type']
        other_cols = [col for col in df_out.columns if col not in cols_order and col != 'Specific_Activity']
        df_out = df_out[cols_order + other_cols]

        output_file = os.path.join(unified_dir, f'Unified_UMA_{ws}s.txt')
        df_out.to_csv(output_file, index=False)
        print(f"\nArchivo generado exitosamente: {output_file}\n")

if __name__ == '__main__':
    main()