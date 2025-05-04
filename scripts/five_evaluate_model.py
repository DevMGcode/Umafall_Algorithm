#five_evaluate_model.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    os.makedirs('evaluate', exist_ok=True)

    clf = joblib.load('models/random_forest_fall_detection_optimized.pkl')
    X_test = pd.read_csv('models/X_test.csv')
    y_test = pd.read_csv('models/y_test.csv').squeeze()

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=['ADL', 'Fall'])
    print("Reporte de clasificación:")
    print(report)

    with open('evaluate/classification_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall']
    )
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig('evaluate/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()