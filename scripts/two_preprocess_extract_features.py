#two_preprocess_extract_features.py
import pandas as pd
import numpy as np
import os
from scipy.stats import zscore, kurtosis, skew
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

def butter_lowpass_filter(data, cutoff=3.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    try:
        return filtfilt(b, a, data)
    except:
        return data

def segment_around_peak(df, window_size_sec=2, fs=50):
    window_size_samples = int(window_size_sec * fs)
    half_window = window_size_samples // 2
    df = df.reset_index(drop=True)
    df['acc_magnitude_norm'] = np.sqrt(df['x_norm']**2 + df['y_norm']**2 + df['z_norm']**2)
    peak_idx = df['acc_magnitude_norm'].idxmax()
    start_idx = max(peak_idx - half_window, 0)
    end_idx = start_idx + window_size_samples
    if end_idx > len(df):
        end_idx = len(df)
        start_idx = max(end_idx - window_size_samples, 0)
    return df.iloc[start_idx:end_idx].copy()

def extract_features(window_df):
    features = {}
    axes = ['x_norm', 'y_norm', 'z_norm']
    for axis in axes:
        data = window_df[axis]
        features[f'mean_{axis}'] = data.mean()
        features[f'var_{axis}'] = data.var()
        features[f'std_{axis}'] = data.std()
        features[f'median_{axis}'] = data.median()
        features[f'max_{axis}'] = data.max()
        features[f'min_{axis}'] = data.min()
        features[f'range_{axis}'] = data.max() - data.min()
        features[f'kurtosis_{axis}'] = kurtosis(data, fisher=True, bias=False)
        features[f'skewness_{axis}'] = skew(data, bias=False)

    mag = np.sqrt(window_df['x_norm']**2 + window_df['y_norm']**2 + window_df['z_norm']**2)
    features['mean_acc_magnitude_norm'] = mag.mean()
    features['var_acc_magnitude_norm'] = mag.var()
    features['std_acc_magnitude_norm'] = mag.std()
    features['max_acc_magnitude_norm'] = mag.max()
    features['min_acc_magnitude_norm'] = mag.min()
    features['range_acc_magnitude_norm'] = mag.max() - mag.min()
    features['kurtosis_acc_magnitude_norm'] = kurtosis(mag, fisher=True, bias=False)
    features['skewness_acc_magnitude_norm'] = skew(mag, bias=False)

    features['corr_xy'] = window_df['x_norm'].corr(window_df['y_norm'])
    features['corr_xz'] = window_df['x_norm'].corr(window_df['z_norm'])
    features['corr_yz'] = window_df['y_norm'].corr(window_df['z_norm'])

    features['label'] = window_df['label'].iloc[0]
    features['activity_type'] = window_df['activity_type'].iloc[0]
    features['file'] = window_df['file'].iloc[0]

    return features

def select_optimized_features(df_features):
    X = df_features.drop(columns=['label', 'activity_type', 'file'])
    y = df_features['label']
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)

    low_importance = mi_series[mi_series < 0.001].index.tolist()

    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]

    to_drop = set(low_importance).union(set(to_drop_corr))

    selected_features = [col for col in df_features.columns if col not in to_drop or col in ['label', 'activity_type', 'file']]

    print(f"Eliminando {len(to_drop)} características (baja importancia o alta correlación).")
    print(f"Características finales: {len(selected_features)}")

    return df_features[selected_features]

def preprocess_data(input_file, output_preprocessed, output_features, output_features_optimized, window_size_sec=2, fs=50):
    print(f"Cargando datos desde {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Datos cargados: {len(df)} filas")

    df_clean = df.dropna(subset=['x', 'y', 'z', 'timestamp', 'label']).copy()
    df_clean['label'] = df_clean['label'].astype(int)

    inconsistent = ((df_clean['label'] == 1) & (df_clean['activity_type'] == 'ADL')) | \
                   ((df_clean['label'] == 0) & (df_clean['activity_type'] == 'Fall'))
    if inconsistent.any():
        df_clean.loc[df_clean['activity_type'] == 'Fall', 'label'] = 1
        df_clean.loc[df_clean['activity_type'] == 'ADL', 'label'] = 0

    df_clean = df_clean.sort_values('timestamp')
    df_clean = df_clean[df_clean['timestamp'].diff().fillna(0) >= 0]

    df_clean = df_clean[(np.abs(zscore(df_clean[['x', 'y', 'z']])) < 3).all(axis=1)]

    for axis in ['x', 'y', 'z']:
        df_clean[f'{axis}_filtered'] = butter_lowpass_filter(df_clean[axis], cutoff=3.0, fs=fs)

    scaler = StandardScaler()
    df_clean[['x_norm', 'y_norm', 'z_norm']] = scaler.fit_transform(df_clean[['x_filtered', 'y_filtered', 'z_filtered']])

    df_clean['acc_magnitude'] = np.sqrt(df_clean['x']**2 + df_clean['y']**2 + df_clean['z']**2)
    df_clean['acc_magnitude_norm'] = np.sqrt(df_clean['x_norm']**2 + df_clean['y_norm']**2 + df_clean['z_norm']**2)

    segmented = []
    features = []
    for file_name, group in df_clean.groupby('file'):
        window_df = segment_around_peak(group, window_size_sec, fs)
        window_df['file'] = file_name
        segmented.append(window_df)
        features.append(extract_features(window_df))

    df_segmented = pd.concat(segmented, ignore_index=True)
    df_features = pd.DataFrame(features)

    os.makedirs(os.path.dirname(output_preprocessed), exist_ok=True)
    df_segmented.to_csv(output_preprocessed, index=False)
    print(f"Datos preprocesados guardados en: {output_preprocessed}")

    os.makedirs(os.path.dirname(output_features), exist_ok=True)
    df_features.to_csv(output_features, index=False)
    print(f"Características extraídas guardadas en: {output_features}")

    df_features_optimized = select_optimized_features(df_features)
    os.makedirs(os.path.dirname(output_features_optimized), exist_ok=True)
    df_features_optimized.to_csv(output_features_optimized, index=False)
    print(f"Características optimizadas guardadas en: {output_features_optimized}")

    return df_segmented, df_features, df_features_optimized

if __name__ == "__main__":
    INPUT_FILE = "data/wrist_filtered/wrist_data_corrected.csv"
    OUTPUT_PREPROCESSED = "data/wrist_filtered/preprocessed/wrist_data_preprocessed_segmented.csv"
    OUTPUT_FEATURES = "data/wrist_filtered/features/wrist_features.csv"
    OUTPUT_FEATURES_OPTIMIZED = "data/wrist_filtered/features/wrist_features_optimized.csv"
    preprocess_data(INPUT_FILE, OUTPUT_PREPROCESSED, OUTPUT_FEATURES, OUTPUT_FEATURES_OPTIMIZED)