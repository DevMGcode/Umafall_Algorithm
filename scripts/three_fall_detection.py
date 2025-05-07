import os
import math
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

test_files_dir = 'Test_files_selected/'
models_dir = 'models/'

def extract_features_from_file(path, sampling_freq=50, window_sec=3):
    try:
        df = pd.read_csv(path, header=None, names=['X', 'Y', 'Z'], skiprows=1)
    except Exception as e:
        print(f"Error leyendo {path}: {e}")
        return None

    sf = int(sampling_freq)
    ws_samples = sf * window_sec

    df['N_XYZ'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)

    max_idx = df['N_XYZ'].idxmax()
    len_df = len(df)

    start_idx = max(0, max_idx - ws_samples//2)
    end_idx = min(len_df, start_idx + ws_samples)
    if end_idx - start_idx < ws_samples:
        start_idx = max(0, end_idx - ws_samples)

    df_win = df.iloc[start_idx:end_idx].reset_index(drop=True)

    df_win['N_HOR'] = np.sqrt(df_win['Y']**2 + df_win['Z']**2)
    df_win['N_VER'] = np.sqrt(df_win['X']**2 + df_win['Z']**2)

    def stats_features(series):
        def safe_stat(func, s):
            val = func(s)
            return 0 if pd.isna(val) else val

        return {
            'var': safe_stat(pd.Series.var, series),
            'mean': safe_stat(pd.Series.mean, series),
            'std': safe_stat(pd.Series.std, series),
            'max': safe_stat(pd.Series.max, series),
            'min': safe_stat(pd.Series.min, series),
            'range': safe_stat(lambda x: x.max() - x.min(), series),
            'kurtosis': safe_stat(pd.Series.kurtosis, series),
            'skewness': safe_stat(pd.Series.skew, series)
        }

    features = {}

    for axis in ['X', 'Y', 'Z', 'N_XYZ', 'N_HOR', 'N_VER']:
        s = df_win[axis]
        f = stats_features(s)
        for k, v in f.items():
            features[f'{k}_{axis}'] = v

    def safe_corr(s1, s2):
        val = s1.corr(s2)
        return 0 if pd.isna(val) else val

    features['corr_XY'] = safe_corr(df_win['X'], df_win['Y'])
    features['corr_XZ'] = safe_corr(df_win['X'], df_win['Z'])
    features['corr_YZ'] = safe_corr(df_win['Y'], df_win['Z'])
    features['corr_NV'] = safe_corr(df_win['N_XYZ'], df_win['N_VER'])
    features['corr_NH'] = safe_corr(df_win['N_XYZ'], df_win['N_HOR'])
    features['corr_HV'] = safe_corr(df_win['N_HOR'], df_win['N_VER'])

    return pd.DataFrame([features])

def assign_label(filename):
    if 'Fall' in filename or 'fall' in filename:
        return 'F'
    else:
        return 'D'

def extract_features_and_labels():
    dfs = []
    for fname in os.listdir(test_files_dir):
        if fname.endswith('.csv'):
            path = os.path.join(test_files_dir, fname)
            features_df = extract_features_from_file(path)
            if features_df is None:
                print(f"Error procesando {fname}, se omite.")
                continue
            features_df['Fall_ADL'] = assign_label(fname)
            dfs.append(features_df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def evaluate_model(name, model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print(f"\nEvaluación modelo {name}:")
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred, labels=['F', 'D'])
    print("Matriz de confusión:")
    print(cm)

    auc = roc_auc_score(y.astype('category').cat.codes.values, y_proba)
    print(f"AUC: {auc:.3f}")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm / cm.sum(axis=1)[:, None], annot=True, fmt='.2f',
                xticklabels=['F', 'D'], yticklabels=['F', 'D'])
    plt.title(f'Matriz de confusión normalizada - {name}')
    plt.xlabel('Clase predicha')
    plt.ylabel('Clase verdadera')
    plt.show()

def main():
    print("Extrayendo características de archivos de prueba...")
    df_features = extract_features_and_labels()
    if df_features.empty:
        print("No se pudieron extraer características de los archivos de prueba.")
        return

    X = df_features.drop(columns=['Fall_ADL'])
    y = df_features['Fall_ADL']

    # Eliminar filas con NaN en X y etiquetas correspondientes
    mask = ~X.isnull().any(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"Datos después de eliminar NaN: {X.shape[0]} muestras")

    print("Cargando modelos y evaluando...")
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    for mf in model_files:
        model_path = os.path.join(models_dir, mf)
        model_name = mf.replace('_model.pkl', '').replace('_', ' ').title()
        model = pickle.load(open(model_path, 'rb'))
        evaluate_model(model_name, model, X, y)

if __name__ == '__main__':
    main()