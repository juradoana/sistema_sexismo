import argparse
import numpy as np
import os
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

#  ARGUMENTOS 
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="ruta o nombre del modelo base")
parser.add_argument("--data_path", type=str, default="/mnt/beegfs/ajh00015/proyecto_llm/datos/EXIST_Unificado_ES_limpio_final.csv", help="Ruta del archivo CSV")
args = parser.parse_args()

MODEL_NAME = args.model_name
DATA_PATH = args.data_path
OUTPUT_DIR = f"/mnt/beegfs/ajh00015/proyecto_llm/modelos2/{MODEL_NAME.split('/')[-1]}_exist"

print(f" Inicio del entrenamiento para: {MODEL_NAME}")

#  CARGAR DATOS 
try:
    raw_dataset = load_dataset('csv', data_files=DATA_PATH)
    
    # Mapeo de etiquetas, convertir clases categóricas a formato numérico 
    label_map = {"not sexist": 0, "sexist": 1} 
    
    def map_labels(example):
        # si ya es entero se deja, si es string se mapea
        l = example['label'] 
        if isinstance(l, str):
            # convertir a minúsculas para evitar errores por mayúsculas
            l = l.lower().strip()
            return {'labels': label_map.get(l, 0)} # Default a 0 si falla
        return {'labels': int(l)}

    # aplicar el mapeo y renombrar a 'labels' 
    # primero buscar la columna correcta
    col_names = raw_dataset['train'].column_names
    target_col = 'label' if 'label' in col_names else ('task1' if 'task1' in col_names else None)
    
    if target_col is None:
        raise ValueError(f"No encuentro columna 'label' o 'task1'. Columnas: {col_names}")
    
    if target_col != 'label':
        raw_dataset = raw_dataset.rename_column(target_col, 'label')

    # Convertimos etiquetas a números
    print(" Convirtiendo etiquetas a números")
    raw_dataset = raw_dataset.map(map_labels)

    # sivisión del dataset: 80% etrenamiento y 20% validación ya que el dataset original no tiene validación
    split_dataset = raw_dataset['train'].train_test_split(test_size=0.2, seed=42)
    dataset = split_dataset
    dataset['validation'] = dataset.pop('test')
    
except Exception as e:
    print(f"Error preparando datos: {e}")
    exit()

# TOKENIZADOR 
# cargar el tokenizador asociado al modelo pre-entrenado
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# truncar las secuencias a 128 tokens para optimizar velocidad de entrenamiento
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=128)

print(" Tokenizando")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# LIMPIEZA FINAL DE COLUMNAS 
# eliminar columnas de texto original para evitar errores de tipo en el modelo
cols_to_keep = ['input_ids', 'attention_mask', 'labels']
if 'token_type_ids' in tokenized_datasets['train'].column_names:
    cols_to_keep.append('token_type_ids') # Algunos modelos lo usan

print(" Eliminando columnas de texto innecesarias para el entrenamiento")
tokenized_datasets.set_format("torch", columns=cols_to_keep)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# cargar el modelo con una cabeza de clasificación binaria 
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# función para el cálculo de métricas durante la evaluación de cada época
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    f1_sexist = f1_score(labels, predictions, pos_label=1)
    return {"accuracy": acc, "f1_macro": f1, "f1_sexist": f1_sexist}

# configuración de hiperparámetros de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch", # evaluación al final de cada época
    save_strategy="epoch",
    load_best_model_at_end=True, # asegurar que nos quedamos con la versión de mayor rendimiento
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    learning_rate=2e-5, # tasa de aprendizaje baja para fine-tuning estable
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,  # número de iteraciones completas sobre el dataset
    weight_decay=0.01,  # técnica de regularización para evitar sobreajuste
    warmup_ratio=0.1, # aumento gradual del LR al inicio del entrenamiento
    fp16=True,
    dataloader_num_workers=4,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_total_limit=2,  # se mantienen solo los últimos 2 checkpoints para ahorrar espacio
    report_to="none"
)

# inicialización del objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer, 
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(" Comenzando entrenamiento")
trainer.train()

final_path = OUTPUT_DIR
print(f" Guardando en: {final_path}")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print(" Métricas Finales:")
print(trainer.evaluate())
