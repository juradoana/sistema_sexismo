import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluar_modelo(csv_path):
    # cargar resultados
    df = pd.read_csv(csv_path)
    
    # limpiar datos: eliminar filas donde Gemma dio error (pred_sexista es NaN)
    df = df.dropna(subset=['pred_sexista'])
    
 
    # el dataset EXIST usa sexist/non-sexist 
    # ajustar para que coincidan sexist:true, non-sexist:false
    y_true = df['true_label'].map({'sexist': True, 'non-sexist': False})
    y_pred = df['pred_sexista'].astype(bool) # asegurar que las predicciones se traten como valores lógicos

    # generar métricas
    print(" CLASIFICACIÓN ")
    # incluye Precision, Recall y F1-Score
    print(classification_report(y_true, y_pred, target_names=['No Sexista', 'Sexista'], digits=4))
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy Global: {acc:.4f}")

    # matriz de confusión 
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
    PATH_RESULTADOS = "dataset_entrenamiento/salida_EXIST_anotado_final.csv"
    evaluar_modelo(PATH_RESULTADOS)