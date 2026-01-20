import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Cargar los datos TF-IDF del dataset entrenamiento y test ya generados
df_train_tfidf = pd.read_csv('dataset_entrenamiento/EXIST2021_tfidf.csv')
df_test_tfidf = pd.read_csv('dataset_entrenamiento/EXIST2021_test_tfidf.csv')

# Cargar las etiquetas del test
df_test_labels = pd.read_csv('dataset_entrenamiento/EXIST2021_test_limpio.csv')

# Separar características y etiquetas (TRAIN)
X_train = df_train_tfidf.drop(columns=['task1', 'task2']) # eliminar columnas de etiquetas, se quedaolo con el texto 
y_train_task1 = df_train_tfidf['task1'] # indica si un texto es sexista o no s
y_train_task2 = df_train_tfidf['task2'] # esta etiqueta clasifica el tipo específico de sexismo

# Características de TEST
X_test = df_test_tfidf  

# etiquetas reales de TEST
y_test_task1 = df_test_labels['task1']
y_test_task2 = df_test_labels['task2']

print("ENTRENAMIENTO Y EVALUACIÓN - TASK 1")

# Entrenar modelo Random Forest para TASK 1
modelo_rf_task1 = RandomForestClassifier(
    n_estimators=200,           # Número de árboles
    max_depth=30,               # Profundidad máxima
    max_features='sqrt',        # Características por split
    min_samples_split=5,        # Mínimo de muestras para split
    min_samples_leaf=2,         # Mínimo de muestras en hoja
    random_state=42,
    n_jobs=-1                   # Usar todos los cores
)

modelo_rf_task1.fit(X_train, y_train_task1) # hace que el modelo aprenda patrones de los datos de entrenamiento


# Predicciones TASK 1
y_pred_task1 = modelo_rf_task1.predict(X_test)

# Calcular métricas TASK 1
accuracy_task1 = accuracy_score(y_test_task1, y_pred_task1)
f1_task1_macro = f1_score(y_test_task1, y_pred_task1, average='macro')
f1_task1_weighted = f1_score(y_test_task1, y_pred_task1, average='weighted')

print(f"\nAccuracy: {accuracy_task1:.4f}")
print(f"F1-Score (macro): {f1_task1_macro:.4f}")
print(f"F1-Score (weighted): {f1_task1_weighted:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test_task1, y_pred_task1))


print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_task1, y_pred_task1))

print("\n" + "="*60)
print("ENTRENAMIENTO Y EVALUACIÓN - TASK 2")
print("="*60)

# Entrenar modelo Random Forest para TASK 2
modelo_rf_task2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

modelo_rf_task2.fit(X_train, y_train_task2)

# Predicciones TASK 2
y_pred_task2 = modelo_rf_task2.predict(X_test)

# Calcular métricas TASK 2
accuracy_task2 = accuracy_score(y_test_task2, y_pred_task2)
f1_task2_macro = f1_score(y_test_task2, y_pred_task2, average='macro')
f1_task2_weighted = f1_score(y_test_task2, y_pred_task2, average='weighted')

print(f"\nAccuracy: {accuracy_task2:.4f}")
print(f"F1-Score (macro): {f1_task2_macro:.4f}")
print(f"F1-Score (weighted): {f1_task2_weighted:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test_task2, y_pred_task2))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_task2, y_pred_task2))

# RESUMEN DE MÉTRICAS
print("\n" + "="*60)
print("RESUMEN DE MÉTRICAS")
print("="*60)

resumen = pd.DataFrame({
    'Tarea': ['Task 1', 'Task 2'],
    'Accuracy': [accuracy_task1, accuracy_task2],
    'F1-Score (macro)': [f1_task1_macro, f1_task2_macro],
    'F1-Score (weighted)': [f1_task1_weighted, f1_task2_weighted]
})

print("\n", resumen)

# Guardar solo las métricas (sin el modelo)
resumen.to_csv('metricas_resumen_rf.csv', index=False)
print("\nResumen guardado en 'metricas_resumen_rf.csv'")