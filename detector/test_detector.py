import pandas as pd
from detector.fall_detector import FallDetector

def test_detector(sample_file, model_path):
    df = pd.read_csv(sample_file)
    detector = FallDetector(model_path)
    prediction = detector.predict(df)
    print(f"Predicción: {'Caída' if prediction == 1 else 'No caída'}")

if __name__ == "__main__":
    test_detector("data/wrist_filtered/01_fall_wrist.csv", "models/random_forest_model.pkl")
