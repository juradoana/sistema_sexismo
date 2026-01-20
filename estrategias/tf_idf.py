import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

spanish_stopwords = stopwords.words('spanish')


# -------------------------------
# 1. Función para limpiar texto
# -------------------------------
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'#(\w+)', r'\1', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# -------------------------------
# 2. Cargar y limpiar datasets
# -------------------------------
df_train = pd.read_csv('dataset_entrenamiento/EXIST_Unificado_ES.csv')
df_train['texto_limpio'] = df_train['text'].apply(limpiar_texto)

df_test = pd.read_csv('datos_test_originales/test_limpio.csv')
df_test['texto_limpio'] = df_test['text'].apply(limpiar_texto)

# -------------------------------
# 3. Crear TF-IDF
# -------------------------------
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1,2),
    strip_accents='unicode',
    stop_words=spanish_stopwords
)

X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['texto_limpio'])
X_test_tfidf  = tfidf_vectorizer.transform(df_test['texto_limpio'])

# Convertir a DataFrame (opcional, útil para guardar)
df_tfidf_train = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_tfidf_train['task1'] = df_train['task1']
df_tfidf_train.to_csv('dataset_entrenamiento/EXIST_Unificado_ES_tfidf_train1.csv', index=False)

df_tfidf_test = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_tfidf_test['id_tweet'] = df_test.index  # o df_test['id_tweet'] si existe
df_tfidf_test.to_csv('datos_test_originales/EXIST_Unificado_test_tfidf.csv', index=False)

# -------------------------------
# 4. Entrenar y evaluar modelos
# -------------------------------
X_train = X_train_tfidf
y_train = df_train['task1']

X_test = X_test_tfidf
y_test = df_test['label']  # etiquetas reales

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Métricas RF
print("\n--- RANDOM FOREST ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1-Score (Macro):", f1_score(y_test, y_pred_rf, average='macro'))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_rf))
print("\nReporte:\n", classification_report(y_test, y_pred_rf))

# --- Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Métricas LR
print("\n--- LOGISTIC REGRESSION ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1-Score (Macro):", f1_score(y_test, y_pred_lr, average='macro'))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_lr))
print("\nReporte:\n", classification_report(y_test, y_pred_lr))

# --- Resumen final ---
resumen = pd.DataFrame({
    'Modelo': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)],
    'F1-Score (Macro)': [f1_score(y_test, y_pred_rf, average='macro'), f1_score(y_test, y_pred_lr, average='macro')]
})

print("\n--- RESUMEN ---")
print(resumen)
