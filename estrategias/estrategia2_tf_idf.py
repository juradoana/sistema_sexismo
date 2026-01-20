import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el archivo de datos de entrenamiento
df = pd.read_csv('dataset_entrenamiento/EXIST2021_training.tsv', sep='\t')


# Función para limpiar texto
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'@\w+', '', texto)                      # quitar menciones
    texto = re.sub(r'#(\w+)', r'\1', texto)                 # quitar el símbolo # de hashtags
    texto = re.sub(r'\d+', '', texto)                        # quitar números
    texto = re.sub(r'[^\w\s]', ' ', texto)                   # quitar emojis y caracteres especiales
    texto = re.sub(r'\s+', ' ', texto)                       # quitar espacios extras
    return texto.strip()

# Limpiar texto 
df['texto_limpio'] = df['text'].apply(limpiar_texto)

# Guardar dataset limpio 
df_limpio = df[['texto_limpio', 'task1', 'task2']]
df_limpio.to_csv('dataset_entrenamiento/EXIST2021_limpio.csv', index=False, encoding='utf-8')

print("Archivo guardado como 'dataset_entrenamiento/EXIST2021_limpio.csv'")

# Info dataset
print("ANÁLISIS DEL DATASET")
print(f"\n Total de ejemplos: {len(df)}")
print(f"\nDistribución de clases:")
conteo = df['task1'].value_counts()
print(conteo)

# TF-IDF
# Cargar el archivo de datos limpio
df = pd.read_csv('dataset_entrenamiento/EXIST2021_limpio.csv')

# División de datos en textos (X) y etiquetas (y)
X_train = df['texto_limpio']

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,           # Máximo número de características
    min_df=2,                    # Ignorar términos que aparecen en menos de 2 documentos
    max_df=0.8,                  # Ignorar términos que aparecen en más del 80% de documentos
    ngram_range=(1, 2),          # Unigramas y bigramas
    strip_accents='unicode',     # Eliminar acentos
    stop_words='english'         # Eliminar stopwords en inglés 
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
df_tfidf['task2'] = df['task2'].values

#Guardar
df_tfidf.to_csv('dataset_entrenamiento/EXIST2021_tfidf.csv', index=False)

print(f"\nMatriz TF-IDF guardada en 'dataset_entrenamiento/EXIST2021_tfidf.csv'")
print(f"Dimensiones: {df_tfidf.shape[0]} filas x {df_tfidf.shape[1]} columnas")

# cargo y limpio el conjunto de datos test
df_test = pd.read_csv('dataset_entrenamiento/EXIST2021_test_labeled.tsv', sep='\t')

# limpio
df_test['texto_limpio'] = df_test['text'].apply(limpiar_texto)

# guardo el dataset test limpio
df_test_limpio = df_test[['texto_limpio', 'task1', 'task2']]
df_test_limpio.to_csv('dataset_entrenamiento/EXIST2021_test_limpio.csv', index=False, encoding='utf-8')

# tf-idf para el conjunto de datos test
X_test_tfidf = tfidf_vectorizer.transform(df_test['texto_limpio'])

# Convertir matriz sparse a DataFrame
feature_names = tfidf_vectorizer.get_feature_names_out()
df_tfidf_test = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)

# Guardar la matriz TF-IDF del test a CSV
df_tfidf_test.to_csv('dataset_entrenamiento/EXIST2021_test_tfidf.csv', index=False)
print(f"\n TF-IDF del dataset test guardada en 'dataset_entrenamiento/EXIST2021_test_tfidf.csv'")