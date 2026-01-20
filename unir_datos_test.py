import pandas as pd
import json

def cargar_exist_tsv(ruta_archivo, anio):
    """Carga archivos TSV de 2021 o 2022 filtrando por español."""
    df = pd.read_csv(ruta_archivo, sep='\t')
    # Filtrar por idioma español
    df = df[df['language'] == 'es'].copy()
    # Seleccionar y renombrar columnas
    df = df[['id', 'text']]
    df.columns = ['id_tweet', 'texto']
    df['anio'] = anio
    return df

def cargar_exist_2023(ruta_archivo):
    """Carga el archivo JSON de 2023 filtrando por español."""
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for key in data:
        item = data[key]
        if item['lang'] == 'es':
            processed_data.append({
                'id_tweet': item['id_EXIST'],
                'texto': item['tweet'],
                'anio': 2023
            })
    return pd.DataFrame(processed_data)

# --- Rutas de los archivos ---
file_2021 = 'datos_test_originales/EXIST2021_test.tsv'
file_2022 = 'datos_test_originales/EXIST2022_test.tsv'
file_2023 = 'datos_test_originales/EXIST2023_test_clean.json'

try:
    print("Procesando datos...")
    df_21 = cargar_exist_tsv(file_2021, 2021)
    df_22 = cargar_exist_tsv(file_2022, 2022)
    df_23 = cargar_exist_2023(file_2023)

    # Unir todos los conjuntos
    df_final = pd.concat([df_21, df_22, df_23], ignore_index=True)

    # Guardar el resultado
    df_final.to_csv('EXIST_test_unificado.csv', index=False, encoding='utf-8')
    
    print(f"¡Hecho! Se han unido {len(df_final)} tweets.")
    print(df_final.sample(5)) # Muestra aleatoria de 5 filas

except Exception as e:
    print(f"Error al procesar los archivos: {e}")