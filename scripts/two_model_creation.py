import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import pickle

# Configura aquí la ruta a tu carpeta unified_dataset
unified_directory = 'unified_dataset/'

# Carpeta para guardar modelos
models_dir = 'models/'
os.makedirs(models_dir, exist_ok=True)

# Columnas esperadas (igual que en model_creation.py)
colnames = ["na",'File', 'Fall_ADL', 'Act_Type', 'var_X', 'mean_X',
    'std_X', 'max_X', 'min_X', 'range_X', 'kurtosis_X',
    'skewness_X', 'var_Y', 'mean_Y', 'std_Y', 'max_Y',
    'min_Y', 'range_Y', 'kurtosis_Y', 'skewness_Y', 'var_Z',
    'mean_Z', 'std_Z','max_Z', 'min_Z', 'range_Z',
    'kurtosis_Z', 'skewness_Z', 'var_N_XYZ', 'mean_N_XYZ',
    'std_N_XYZ', 'max_N_XYZ', 'min_N_XYZ', 'range_N_XYZ',
    'kurtosis_N_XYZ', 'skewness_N_XYZ', 'var_N_HOR',
    'mean_N_HOR', 'std_N_HOR', 'max_N_HOR', 'min_N_HOR',
    'range_N_HOR', 'kurtosis_N_HOR', 'skewness_N_HOR',
    'var_N_VER', 'mean_N_VER', 'std_N_VER', 'max_N_VER',
    'min_N_VER', 'range_N_VER', 'kurtosis_N_VER',
    'skewness_N_VER', 'corr_XY', 'corr_XZ', 'corr_YZ', 'corr_NV',
    'corr_NH', 'corr_HV']

def load_uma_data(window_size):
    file_path = unified_directory + f'Unified_UMA_{window_size}s.txt'
    df = pd.read_csv(file_path, sep=',', names=colnames, header=None)
    df.drop('na', axis=1, inplace=True)
    return df

def main():
    window_size = "1.5"
    print(f"Cargando datos UMA ventana {window_size}s...")
    df_uma = load_uma_data(window_size)

    # Limpiar filas con valores inválidos en Fall_ADL
    df_uma = df_uma[df_uma['Fall_ADL'].isin(['D', 'F'])].copy()

    print("Distribución de clases antes de SMOTE:")
    print(df_uma['Fall_ADL'].value_counts())

    # Separar características y etiquetas
    X = df_uma.drop(['File', 'Fall_ADL', 'Act_Type'], axis=1)
    y = df_uma['Fall_ADL']

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Aplicar SMOTE solo al conjunto de entrenamiento
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Distribución de clases después de SMOTE en entrenamiento:")
    print(pd.Series(y_train_res).value_counts())

    # Normalizar
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_res)
    X_test_norm = scaler.transform(X_test)

    # Entrenar modelos con mejores parámetros fijos para rapidez
    knn = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    rf = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=42)
    svc = SVC(C=2, kernel='linear', gamma='scale', probability=True, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 64), max_iter=500, random_state=42)

    knn.fit(X_train_res, y_train_res)
    rf.fit(X_train_res, y_train_res)
    svc.fit(X_train_res, y_train_res)
    mlp.fit(X_train_norm, y_train_res)

    def evaluate_model(name, model_obj, X_test_data, y_true, is_mlp=False):
        if is_mlp:
            probs = model_obj.predict_proba(X_test_data)[:, 1]
            preds = model_obj.predict(X_test_data)
        else:
            probs = model_obj.predict_proba(X_test_data)[:, 1]
            preds = model_obj.predict(X_test_data)

        print(f"\nEvaluación modelo {name}:")
        print(classification_report(y_true, preds))
        cm = confusion_matrix(y_true, preds, labels=['F', 'D'])
        print("Matriz de confusión:")
        print(cm)

        auc = roc_auc_score(y_true.astype('category').cat.codes.values, probs)
        print(f"AUC: {auc:.3f}")

        plt.figure(figsize=(5,4))
        sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f',
                    xticklabels=['F', 'D'], yticklabels=['F', 'D'])
        plt.title(f'Matriz de confusión normalizada - {name}')
        plt.xlabel('Clase predicha')
        plt.ylabel('Clase verdadera')
        plt.show()

        return probs

    models = [knn, rf, svc, mlp]
    model_names = ['KNN', 'Random Forest', 'SVC', 'MLP']

    probs_list = []
    for m, name in zip(models, model_names):
        if name == 'MLP':
            probs = evaluate_model(name, m, X_test_norm, y_test, is_mlp=True)
        else:
            probs = evaluate_model(name, m, X_test, y_test)
        probs_list.append(probs)

    plt.figure(figsize=(8,6))
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test.astype('category').cat.codes.values, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test.astype('category').cat.codes.values, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='AUC = 0.5 (No Skill)')

    for probs, name in zip(probs_list, model_names):
        auc = roc_auc_score(y_test.astype('category').cat.codes.values, probs)
        fpr, tpr, _ = roc_curve(y_test.astype('category').cat.codes.values, probs)
        plt.plot(fpr, tpr, label=f'AUC ({name}) = {auc:.2f}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curvas ROC ventana {window_size}s con SMOTE')
    plt.legend()
    plt.show()

    # Guardar modelos entrenados
    for model, name in zip(models, model_names):
        model_path = os.path.join(models_dir, f'{name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo {name} guardado en {model_path}")

if __name__ == '__main__':
    main()