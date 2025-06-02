# Sistema de Detección de Caídas con Machine Learning

Sistema inteligente para detectar caídas automáticamente usando sensores inerciales y algoritmos de Machine Learning.


[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/accuracy-92.55%25-brightgreen.svg)](#resultados)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-97.41%25-brightgreen.svg)](#resultados)

## Descripción

Este proyecto detecta caídas en adultos mayores usando datos de acelerómetros y giroscopios. Combina los datasets UMAFall y WEDAFall para entrenar algoritmos de Machine Learning que clasifican eventos de caída vs. actividades normales.

### Resultados Principales
- **Exactitud**: 92.55%
- **ROC-AUC**: 97.41%
- **Algoritmo ganador**: Random Forest optimizado
- **Ventana de análisis**: 1.5 segundos
- **Características extraídas**: 42 por ventana

## Instalación

```bash
git clone https://github.com/DevMGcode/FallDetection-ML-System.git
cd FallDetection-ML-System
pip install -r requirements.txt
```

## Uso

### 1. Procesar datasets
```bash
python scripts/one_UMAFALL_Extract_features.py
python scripts/two_WEDAFALL_Extract_features.py
python scripts/three_preprocesamiento_combinado.py
```

### 2. Entrenar modelos
```bash
python scripts/five_ImplementacionAlgoritmos.py
python scripts/six_optimizacion_modelo_randomforest.py
```

### 3. Validar resultados
```bash
python scripts/seven_validacion_final.py
```

## Estructura del Proyecto

```
FallDetection-ML-System/
├── datasets/              # Datasets originales
├── scripts/               # Scripts de procesamiento
├── results/               # Resultados y modelos
├── unified_UMAFALL_dataset/
├── unified_WEDAFALL_dataset/
└── README.md
```

## Algoritmos Comparados

| Algoritmo | F1-Score | Ranking |
|-----------|----------|---------|
| Random Forest | 0.8909 | 1° |
| Decision Tree | 0.8307 | 2° |
| KNN | 0.7739 | 3° |
| Neural Network | 0.7226 | 4° |
| SVM | 0.6793 | 5° |

## Datasets Utilizados

- **UMAFall**: 31 sujetos, 8 tipos de caídas, 8 actividades diarias
- **WEDAFall**: 31 sujetos, rango de edad amplio (20-95 años)
- **Frecuencia**: 50 Hz
- **Sensores**: Acelerómetros tri-axiales y giroscopios

## Características Extraídas

Para cada ventana de 1.5s se calculan:
- Estadísticas por eje (X,Y,Z): varianza, media, std, max, min, rango
- Normas: euclidiana, horizontal, vertical
- Correlaciones entre ejes

## Resultados

### Métricas del Modelo Final
- Exactitud: 92.55%
- ROC-AUC: 97.41%
- Precisión promedio: 96.05%
- Falsos positivos: <8%

### Características Más Importantes
1. Varianza en eje Z (vertical) - 15.2%
2. Norma euclidiana máxima - 12.8%
3. Correlación X-Z - 9.4%

## Dependencias

```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
```

## Implementación Base

Este proyecto se desarrolló tomando como referencia e inspiración la implementación base del repositorio:

📚 **Repositorio base**: [Fall Detection Algorithm](https://github.com/arnautiendat/fall_detection/tree/master)

El algoritmo fue adaptado y mejorado para:
- Integrar múltiples datasets (UMAFall + WEDAFall)
- Comparar 5 algoritmos diferentes de ML
- Optimizar Random Forest con GridSearchCV
- Implementar validación cruzada estratificada
- Añadir balanceamiento de clases con SMOTE

## Autor

**Melissa García**  
Facultad de Ingeniería - Universidad EAN

## Referencias y Créditos

### Datasets
- **UMAFall Dataset**: [UMA ADL FALL Dataset - Figshare](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283)
- **WEDA-FALL Dataset**: [WEDA-FALL GitHub Repository](https://github.com/joaojtmarques/WEDA-FALL)

### Implementación Base
- **Fall Detection Algorithm**: [arnautiendat/fall_detection](https://github.com/arnautiendat/fall_detection/tree/master)

### Artículos Científicos
- Casilari, E., et al. (2017). "UMAFall: A multisensor dataset for the research on automatic fall detection"
- Marques, J., Moreno, P. (2023). "Online Fall Detection Using Wrist Devices" *Sensors*, 23(3), 1146

