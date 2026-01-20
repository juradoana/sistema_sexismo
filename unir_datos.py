import pandas as pd
import json
from collections import Counter

# ---------------------------------------------------------
# 1. Función para EXIST 2021 (TSV)
# ---------------------------------------------------------
def cargar_exist_2021(ruta_archivo):
    # Carga con separador tabulador ('\t')
    df = pd.read_csv(ruta_archivo, sep='\t')
    
    # 1. Filtrar idioma español
    # Nota: Asegúrate de que la columna se llame 'language' exactamente
    df = df[df['language'] == 'es'].copy()
    
    # 2. Seleccionar columnas clave
    # En tu muestra: id, text, task1
    df = df[['id', 'text', 'task1']]
    
    # 3. Renombrar para estandarizar
    df.columns = ['id_tweet', 'texto', 'etiqueta']
    
    # 4. Añadir año
    df['anio'] = 2021
    
    # (Opcional) Normalizar etiquetas si quieres todo en minúsculas
    # df['etiqueta'] = df['etiqueta'].str.lower()
    
    return df

# ---------------------------------------------------------
# 2. Función para EXIST 2023 (JSON Diccionario)
# ---------------------------------------------------------
def cargar_exist_2023(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    # Iteramos sobre los VALORES del diccionario (cada objeto tweet)
    for tweet_id, entry in data.items():
        
        # 1. Filtrar idioma español
        if entry.get('lang') != 'es':
            continue
            
        # 2. Extraer datos
        # El ID a veces viene como clave del dict o dentro como 'id_EXIST'
        # Usamos el de dentro si existe, o la clave si no.
        id_tweet = entry.get('id_EXIST', tweet_id)
        texto = entry.get('tweet')
        
        # 3. Gestión de etiquetas (Voto por Mayoría)
        labels = entry.get('labels_task1', [])
        
        if not labels:
            continue 
            
        conteo = Counter(labels)
        voto_mayoria = conteo.most_common(1)[0][0] # 'YES' o 'NO'
        
        # Convertir a formato 2021 ('sexist' / 'non-sexist')
        etiqueta_final = 'sexist' if voto_mayoria == 'YES' else 'non-sexist'
        
        processed_data.append({
            'id_tweet': id_tweet,
            'texto': texto,
            'etiqueta': etiqueta_final,
            'anio': 2023
        })
        
    return pd.DataFrame(processed_data)

# ---------------------------------------------------------
# 3. Ejecución Principal
# ---------------------------------------------------------

# ¡IMPORTANTE! CAMBIA ESTAS RUTAS POR LAS DE TUS ARCHIVOS
archivo_2021 = 'dataset_entrenamiento/EXIST2021_training.tsv' 
archivo_2023 = 'dataset_entrenamiento/EXIST2023_training.json'

try:
    print("--- Procesando EXIST 2021 ---")
    df_2021 = cargar_exist_2021(archivo_2021)
    print(f"Tweets recuperados: {len(df_2021)}")
    print(df_2021.head(2))

    print("\n--- Procesando EXIST 2023 ---")
    df_2023 = cargar_exist_2023(archivo_2023)
    print(f"Tweets recuperados: {len(df_2023)}")
    print(df_2023.head(2))

    # Unir
    df_final = pd.concat([df_2021, df_2023], ignore_index=True)

    # Guardar
    nombre_salida = 'dataset_entrenamiento/EXIST_Unificado_ES.csv'
    df_final.to_csv(nombre_salida, index=False)
    
    print(f"\n✅ ¡Éxito! Archivo guardado como: {nombre_salida}")
    print(f"Total tweets: {len(df_final)}")
    print("Distribución por año:")
    print(df_final['anio'].value_counts())

except Exception as e:
    print(f"\n❌ Error: {e}")