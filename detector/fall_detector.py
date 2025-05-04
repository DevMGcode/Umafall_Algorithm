import joblib
import pandas as pd
import numpy as np

class FallDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def extract_features(self, df):
        features = []
        for col in df.columns:
            features += [
                df[col].mean(), df[col].std(), df[col].min(),
                df[col].max(), df[col].median(), df[col].mad()
            ]
        return np.array(features).reshape(1, -1)

    def predict(self, df):
        features = self.extract_features(df)
        return self.model.predict(features)[0]
