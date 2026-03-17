import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# definición de palabras vacías (stopwords) en español para filtrar términos irrelevantes
spanish_stopwords = stopwords.words('spanish')

# normalización y limpieza de texto
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'#(\w+)', r'\1', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# cargar y limpiar datasets
# train
df_train = pd.read_csv('dataset_entrenamiento/EXIST_Unificado_ES.csv')
df_train['texto_limpio'] = df_train['text'].apply(limpiar_texto)
# test
df_test = pd.read_csv('datos_test_originales/test_limpio.csv')
df_test['texto_limpio'] = df_test['text'].apply(limpiar_texto)


# creación de TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000, # se limita a las 5000 palabras más frecuentes
    min_df=2, # se ignoran términos que aparecen en menos de 2 documentos
    max_df=0.8, # descartar palabras muy comunes, que aparecen en el 80% de los textos
    ngram_range=(1,2), # analiza palabras sueltas y pares de palabras (unigramas y bigramas)
    strip_accents='unicode',
    stop_words=spanish_stopwords
)

X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['texto_limpio']) # analiza el vocabulario del set de entrenamiento y calcula la importancia de cada término
X_test_tfidf  = tfidf_vectorizer.transform(df_test['texto_limpio']) # el modelo se evalua con el vocabulario aprendido en entrenamiento

# convertir a DataFrame 
df_tfidf_train = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_tfidf_train['task1'] = df_train['task1']
df_tfidf_train.to_csv('dataset_entrenamiento/EXIST_Unificado_ES_tfidf_train1.csv', index=False)

df_tfidf_test = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_tfidf_test['id_tweet'] = df_test.index  
df_tfidf_test.to_csv('datos_test_originales/EXIST_Unificado_test_tfidf.csv', index=False)


# definición de variables predictoras (X) y variable objetivo (y)
X_train = X_train_tfidf # representación numérica de los textos
y_train = df_train['task1']

X_test = X_test_tfidf
y_test = df_test['label']  # etiquetas reales

# random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# métricas Random Forest
print("\n RANDOM FOREST ")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1-Score (Macro):", f1_score(y_test, y_pred_rf, average='macro'))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_rf))
print("Precision (Macro):", precision_score(y_test, y_pred_rf, average='macro'))
print("Recall (Macro):", recall_score(y_test, y_pred_rf, average='macro'))
print("\nReporte:\n", classification_report(y_test, y_pred_rf))

#  Logistic Regression 
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# métricas Logistic Regression
print("\n LOGISTIC REGRESSION ")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1-Score (Macro):", f1_score(y_test, y_pred_lr, average='macro'))
print("Precision (Macro):", precision_score(y_test, y_pred_lr, average='macro'))
print("Recall (Macro):", recall_score(y_test, y_pred_lr, average='macro'))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_lr))
print("\nReporte:\n", classification_report(y_test, y_pred_lr))

# eesumen comparativo de métricas 
resumen = pd.DataFrame({
    'Modelo': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)],
    'F1-Score (Macro)': [f1_score(y_test, y_pred_rf, average='macro'), f1_score(y_test, y_pred_lr, average='macro')]
})

print("\n RESUMEN ")
print(resumen)

# crear las matrices de confusión para ambos modelos

# calcular matriz
cm_rf = confusion_matrix(y_test, y_pred_rf)

# crear figura
plt.figure(figsize=(6,5))


plt.imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Random Forest")
plt.colorbar()

# etiquetas
classes = ["Not Sexist", "Sexist"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel("Predicción")
plt.ylabel("Realidad")

# escribir números dentro
thresh = cm_rf.max() / 2.
for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        plt.text(j, i, format(cm_rf[i, j], 'd'),
                 ha="center",
                 color="white" if cm_rf[i, j] > thresh else "black")

plt.tight_layout()

# guardar imagen
plt.savefig("graficas/confusion_matrix_random_forest.png", dpi=300)
plt.show()

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6,5))
plt.imshow(cm_lr, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Logistic Regression")
plt.colorbar()

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel("Predicción")
plt.ylabel("Realidad")

thresh = cm_lr.max() / 2.
for i in range(cm_lr.shape[0]):
    for j in range(cm_lr.shape[1]):
        plt.text(j, i, format(cm_lr[i, j], 'd'),
                 ha="center",
                 color="white" if cm_lr[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("graficas/confusion_matrix_logistic_regression.png", dpi=300)
plt.show()