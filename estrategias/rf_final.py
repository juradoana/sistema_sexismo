import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


print("Cargando datos etiquetados...")
df = pd.read_csv('dataset_entrenamiento/EXIST_Unificado_ES_tfidf_train.csv')

# Separamos Características (X) y Etiqueta (y)
X = df.drop(columns=['task1']) # Todo menos la etiqueta
y = df['task1']                # La etiqueta (sexist / non-sexist)


# Esto es vital para sacar métricas reales.
# 80% para aprender (X_train), 20% para examinar (X_test)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20,   # 20% para test
    random_state=42,  # Semilla para reproducibilidad
    stratify=y        # Mantiene la misma proporción de sexismo en ambos conjuntos
)

print(f"Dimensiones Entrenamiento: {X_train.shape}")
print(f"Dimensiones Test (Evaluación): {X_test.shape}")

# random forest
print("\n" + "="*60)
print("MODELO 1: RANDOM FOREST")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Métricas RF
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro') # Macro para clases desbalanceadas

print(f"Accuracy: {acc_rf:.4f}")
print(f"F1-Score (Macro): {f1_rf:.4f}")
print("\nMatriz de Confusión (RF):")
print(confusion_matrix(y_test, y_pred_rf))
print("\nReporte detallado:")
print(classification_report(y_test, y_pred_rf))

# logistic regression
print("\n" + "="*60)
print("MODELO 2: REGRESIÓN LOGÍSTICA")
print("="*60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)

lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Métricas LR
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')

print(f"Accuracy: {acc_lr:.4f}")
print(f"F1-Score (Macro): {f1_lr:.4f}")
print("\nMatriz de Confusión (LR):")
print(confusion_matrix(y_test, y_pred_lr))
print("\nReporte detallado:")
print(classification_report(y_test, y_pred_lr))

# resumen
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

resumen = pd.DataFrame({
    'Modelo': ['Random Forest', 'Regresión Logística'],
    'Accuracy': [acc_rf, acc_lr],
    'F1-Score (Macro)': [f1_rf, f1_lr]
})

print(resumen)
