import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# configuración visual para ver la tabla completa en consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# CONFIGURACIÓN
# ruta del corpus de evaluación
test_file_path = '/mnt/beegfs/ajh00015/proyecto_llm/datos/test_limpio.csv'

output_dir = "./graficas" 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# diccionario de mapeo para asegurar la consistencia entre etiquetas de texto e índices numéricos
label_map = {"non-sexist": 0, "sexist": 1}
target_names = ["Not Sexist", "Sexist"]

# DEFINICIÓN DE MODELOS: para comparar modelos base con modelos entrenados (fine-tuned)
models_to_evaluate = {
    # BASE
    "XLM-RoBERTa (Base)": "/mnt/beegfs/sinai-data/FacebookAI/xlm-roberta-base",
    "mDeBERTa (Base)": "/mnt/beegfs/sinai-data/microsoft/mdeberta-v3-base",
    "RoBERTa-BNE (Base)": "/mnt/beegfs/sinai-data/PlanTL-GOB-ES/roberta-base-bne",
    
    # TRAINED
    "XLM-RoBERTa (Trained)": "/mnt/beegfs/ajh00015/proyecto_llm/modelos2/xlm-roberta-base_exist/final_model",
    "mDeBERTa (Trained)": "/mnt/beegfs/ajh00015/proyecto_llm/modelos2/mdeberta-v3-base_exist/final_model",
    "RoBERTa-BNE (Trained)": "/mnt/beegfs/ajh00015/proyecto_llm/modelos2/roberta-base-bne_exist/final_model"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Usando dispositivo: {device}")


# CARGA DE DATOS
# carga el archivo CSV y realiza una limpieza de etiquetas para asegurar que coincidan con el mapeo esperado por los modelos
def load_data(path):
    if not os.path.exists(path):
        dummy_data = {"text": ["Test 1", "Test 2"], "label": ["sexist", "non-sexist"]}
        pd.DataFrame(dummy_data).to_csv(path, index=False)

    df = pd.read_csv(path)
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.strip().str.lower() # estandarización de strings para evitar errores de concordancia 
    
    label_map_cleaned = {k.lower(): v for k, v in label_map.items()}
    df['label_id'] = df['label'].map(label_map_cleaned)
    df = df.dropna(subset=['label_id', 'text']) # eliminación de posibles registros incompletos
    df['label_id'] = df['label_id'].astype(int)
    
    print(f" Datos cargados: {len(df)} muestras.")
    return df['text'].tolist(), df['label_id'].tolist()

texts, true_labels = load_data(test_file_path)


# EVALUACIÓN
# realiza la inferencia por lotes y calcula las métricas de rendimiento
def evaluate_model(model_name, model_path, texts, true_labels):
    print(f"\n  Evaluando: {model_name} ")
    
    try:
        # instanciación de pesos y tokenizador desde el almacenamiento local
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    except OSError as e:
        print(f"Error cargando {model_name}: {e}")
        return None

    model.to(device)
    model.eval()
    
    all_preds = []
    batch_size = 32
    
    # ciclo de inferencia por lotes para evitar desbordamiento de memoria
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    # generación de métricas detalladas por clase
    report_dict = classification_report(
        true_labels, 
        all_preds, 
        labels=[0, 1], 
        target_names=target_names, 
        output_dict=True, 
        zero_division=0
    )
    
    # cálculo de la matriz de confusión para análisis de errores (Falsos Positivos/Negativos)
    cm = confusion_matrix(true_labels, all_preds, labels=[0, 1])

    return {
        "Model": model_name,
        "Accuracy": report_dict["accuracy"],
        "Macro F1": report_dict["macro avg"]["f1-score"],  
        "Macro Prec": report_dict["macro avg"]["precision"], 
        "Macro Rec": report_dict["macro avg"]["recall"],
        "Sexist F1": report_dict["Sexist"]["f1-score"],
        "Sexist Prec": report_dict["Sexist"]["precision"],  
        "Sexist Rec": report_dict["Sexist"]["recall"],      
        "Confusion Matrix": cm
    }

# ejecución del bucle de evaluación sobre todos los modelos definidos
results_list = []
for name, path in models_to_evaluate.items():
    res = evaluate_model(name, path, texts, true_labels)
    if res:
        results_list.append(res)

if not results_list:
    sys.exit(" No hay resultados válidos.")


# IMPRIMIR RESULTADOS Y GUARDAR GRÁFICAS

# MOSTRAR TABLA EN PANTALLA
results_df = pd.DataFrame(results_list)
cols_to_show = ["Model", "Accuracy", "Macro F1","Macro Prec", "Macro Rec", "Sexist F1", "Sexist Prec", "Sexist Rec"]
summary_df = results_df[cols_to_show].sort_values(by="Macro F1", ascending=False)

print("\n\n COMPARATIVA FINAL DE MÉTRICAS ")
print(summary_df.round(4).to_string(index=False)) # Imprime la tabla bonita y completa
print("="*100 + "\n")

# GUARDAR MATRICES DE CONFUSIÓN 
print("Generando 6 matrices de confusión individuales")

for res in results_list:
    model_name_safe = res["Model"].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(res["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 14})
    
    plt.title(f"Matriz de Confusión\n{res['Model']}", fontsize=14, fontweight='bold')
    plt.ylabel("Realidad", fontsize=12)
    plt.xlabel("Predicción", fontsize=12)
    
    filename = f"Matriz_{model_name_safe}.png"
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  Guardada: {filename}")

