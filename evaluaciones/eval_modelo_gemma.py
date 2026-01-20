import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluar_modelo(csv_path):
    # 1. Cargar los resultados
    df = pd.read_csv(csv_path)
    
    # 2. Limpiar datos: eliminar filas donde el LLM dio error (pred_sexista es NaN)
    df = df.dropna(subset=['pred_sexista'])
    
    # 3. 
    # El dataset EXIST usa 'sexist'/'non-sexist'. 
    # Tu script guarda True/False. Ajustamos para que coincidan:
    y_true = df['true_label'].map({'sexist': True, 'non-sexist': False})
    y_pred = df['pred_sexista'].astype(bool)

    # 4. Generar métricas
    print("=== CLASIFICACIÓN ===")
    print(classification_report(y_true, y_pred, target_names=['No Sexista', 'Sexista']))
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy Global: {acc:.2f}")

    # 5. Matriz de Confusión Visual
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: No Sexista', 'Pred: Sexista'],
                yticklabels=['Real: No Sexista', 'Real: Sexista'])
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.title('Matriz de Confusión - Detección de Sexismo')
    plt.show()

if __name__ == "__main__":
    # Cambia esto por la ruta real de tu archivo de salida
    PATH_RESULTADOS = "dataset_entrenamiento/salida_EXIST_anotado_final.csv"
    evaluar_modelo(PATH_RESULTADOS)