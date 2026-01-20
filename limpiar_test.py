import pandas as pd
import re

# RUTAS (Ajusta esto a donde tengas tu archivo)
INPUT_FILE = "dataset_entrenamiento/EXIST2021_test_labeled.tsv" # Tu archivo original
OUTPUT_FILE = "datos_test_originales/test_limpio.csv" # Archivo final

def clean_tweet_text(text):
    """Función para limpiar URLs y usuarios del texto"""
    if not isinstance(text, str):
        return ""
    
    # 1. Eliminar URLs (http://... o https://...)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 2. Eliminar menciones de usuario (@usuario)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Eliminar espacios múltiples y espacios al inicio/final
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_data():
    print("Leyendo archivo...")
    # Leer TSV (separado por tabulaciones)
    df = pd.read_csv(INPUT_FILE, sep='\t')
    
    # 1. Filtrar solo Español
    if 'language' in df.columns:
        df = df[df['language'] == 'es'].copy()
        print(f"Tweets en español encontrados: {len(df)}")
    
    # 2. Limpiar el texto
    print("Limpiando textos (eliminando URLs y @usuarios)...")
    df['text_clean'] = df['text'].apply(clean_tweet_text)
    
    # 3. Seleccionar solo texto limpio y etiqueta original
    # Renombramos 'task1' a 'label' para mayor claridad, o puedes dejarlo como 'task1'
    df_final = df[['text_clean', 'task1']].rename(columns={'text_clean': 'text', 'task1': 'label'})
    
    # Filtrar filas que hayan quedado vacías tras la limpieza (opcional pero recomendado)
    initial_len = len(df_final)
    df_final = df_final[df_final['text'] != ""]
    if len(df_final) < initial_len:
        print(f"Se eliminaron {initial_len - len(df_final)} tweets que quedaron vacíos tras la limpieza.")

    # 4. Guardar
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Archivo guardado: {OUTPUT_FILE}")
    print(df_final.head())

if __name__ == "__main__":
    process_data()
