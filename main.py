from scripts.one_extract_data_wrist_filter import extract_and_filter_wrist_data
from scripts.two_preprocess_extract_features import eda_on_wrist_data
from scripts.three_eda import extract_features_from_folder
from scripts.four_train_model import train_model
from scripts.five_evaluate_model import evaluate_model

def main():
    print("🔧 Paso 1: Extracción y filtrado de datos...")
    extract_and_filter_wrist_data("UMAFall_Dataset.zip", "data/raw")

    print("📊 Paso 2: Análisis exploratorio...")
    eda_on_wrist_data("data/wrist_filtered", "eda/two_eda_summary.cvs", "eda/plots/class_distribution.png")

    print("📈 Paso 3: Extracción de características...")
    extract_features_from_folder("data/wrist_filtered", "data/features/features.csv")

    print("🤖 Paso 4: Entrenamiento del modelo...")
    train_model("data/features/features.csv", "models/random_forest_model.pkl")

    print("✅ Paso 5: Evaluación del modelo...")
    evaluate_model("data/features/features.csv", "models/random_forest_model.pkl")

    print("🎉 Pipeline completo.")

if __name__ == "__main__":
    main()
