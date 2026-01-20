import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Cargar el archivo de datos de entrenamiento (CSV, no TSV)
df = pd.read_csv('dataset_entrenamiento/EXIST_Unificado_ES.csv') # <--- CAMBIO: Nombre y separador automático (coma)

# Función para limpiar texto (Se mantiene igual)
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'#(\w+)', r'\1', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# Limpiar texto (Entrenamiento usa columna 'text')
df['texto_limpio'] = df['text'].apply(limpiar_texto)

# 2. Guardar dataset limpio
# <--- CAMBIO: Eliminamos 'task2' porque no existe en este archivo
df_limpio = df[['texto_limpio', 'task1']] 
df_limpio.to_csv('dataset_entrenamiento/EXIST_Unificado_ES_limpio.csv', index=False, encoding='utf-8')

print("Archivo guardado como 'dataset_entrenamiento/EXIST_Unificado_ES_limpio.csv'")

# Info dataset
print("ANÁLISIS DEL DATASET")
print(f"\n Total de ejemplos: {len(df)}")
print(f"\nDistribución de clases:")
conteo = df['task1'].value_counts()
print(conteo)

# TF-IDF
X_train = df['texto_limpio']

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    strip_accents='unicode',
    stop_words='english'
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Convertir matriz TF-IDF a DataFrame
feature_names = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(
    X_train_tfidf.toarray(),
    columns=feature_names
)

# Añadir las etiquetas
df_tfidf['task1'] = df['task1'].values

# Guardar
df_tfidf.to_csv('dataset_entrenamiento/EXIST_Unificado_ES_tfidf_train.csv', index=False)
print(f"\nMatriz TF-IDF guardada en 'dataset_entrenamiento/EXIST_Unificado_ES_tfidf_train.csv'")

#

# 3. Cargar test (CSV)
df_test = pd.read_csv('EXIST_test_unificado.csv') # <--- CAMBIO: Nombre archivo

# 4. Limpiar (Test usa columna 'texto', no 'text')
df_test['texto_limpio'] = df_test['texto'].apply(limpiar_texto) # <--- CAMBIO: columna 'texto'

# 5. Guardar dataset test limpio
# <--- CAMBIO: NO guardamos task1/task2 porque el test no las tiene. 
# Guardamos 'id_tweet' para poder identificar luego qué predicción es de qué tweet.
df_test_limpio = df_test[['id_tweet', 'texto_limpio']]
df_test_limpio.to_csv('EXIST_Unificado_test_limpio.csv', index=False, encoding='utf-8')

# tf-idf para el conjunto de datos test
X_test_tfidf = tfidf_vectorizer.transform(df_test['texto_limpio'])

# Convertir matriz sparse a DataFrame
df_tfidf_test = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)

# <--- CAMBIO Opcional: Añadir el ID al TF-IDF del test para no perder el rastro
# df_tfidf_test['id_tweet'] = df_test['id_tweet'].values 

# Guardar la matriz TF-IDF del test a CSV
df_tfidf_test.to_csv('EXIST_Unificado_test_tfidf.csv', index=False)

print(f"\n TF-IDF del dataset test guardada en 'EXIST_Unificado_test_tfidf.csv'")