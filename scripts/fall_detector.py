import numpy as np
from scipy import stats
import joblib

class FallDetector:
    def __init__(self, model_path='models/random_forest_model.pkl'):
        self.model = joblib.load(model_path)

    def extract_features(self, data):
        """
        Extrae características de los datos del acelerómetro.

        Args:
            data: Array de numpy con forma (n, 4) donde cada fila contiene [timestamp, x, y, z]

        Returns:
            Array de numpy con las características extraídas
        """
        # Extraer componentes x, y, z
        x = data[:, 1]
        y = data[:, 2]
        z = data[:, 3]

        # Calcular magnitud
        magnitude = np.sqrt(x**2 + y**2 + z**2)

        # Características estadísticas para cada eje
        feature_vector = []

        # Características para el eje X
        feature_vector.extend([
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75),
            stats.skew(x), stats.kurtosis(x)
        ])

        # Características para el eje Y
        feature_vector.extend([
            np.mean(y), np.std(y), np.min(y), np.max(y),
            np.percentile(y, 25), np.percentile(y, 50), np.percentile(y, 75),
            stats.skew(y), stats.kurtosis(y)
        ])

        # Características para el eje Z
        feature_vector.extend([
            np.mean(z), np.std(z), np.min(z), np.max(z),
            np.percentile(z, 25), np.percentile(z, 50), np.percentile(z, 75),
            stats.skew(z), stats.kurtosis(z)
        ])

        # Características para la magnitud
        feature_vector.extend([
            np.mean(magnitude), np.std(magnitude), np.min(magnitude), np.max(magnitude),
            np.percentile(magnitude, 25), np.percentile(magnitude, 50), np.percentile(magnitude, 75),
            stats.skew(magnitude), stats.kurtosis(magnitude)
        ])

        # Correlaciones entre ejes
        feature_vector.extend([
            np.corrcoef(x, y)[0, 1],
            np.corrcoef(x, z)[0, 1],
            np.corrcoef(y, z)[0, 1]
        ])

        return np.array([feature_vector])

    def detect_fall(self, data):
        """
        Detecta si hay una caída en los datos proporcionados.

        Args:
            data: Array de numpy con forma (n, 4) donde cada fila contiene [timestamp, x, y, z]

        Returns:
            True si se detecta una caída, False en caso contrario
        """
        features = self.extract_features(data)
        prediction = self.model.predict(features)[0]
        return bool(prediction)

    def get_fall_probability(self, data):
        """
        Obtiene la probabilidad de caída para los datos proporcionados.

        Args:
            data: Array de numpy con forma (n, 4) donde cada fila contiene [timestamp, x, y, z]

        Returns:
            Probabilidad de caída (entre 0 y 1)
        """
        features = self.extract_features(data)
        probability = self.model.predict_proba(features)[0, 1]
        return probability