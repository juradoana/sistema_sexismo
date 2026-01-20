import pandas as pd
import re

def clean_text(text):
    """Limpieza integral de texto para NLP"""
    if not isinstance(text, str):
        return ""
        
    # 1. Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Eliminar menciones (@usuario)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Eliminar símbolo hashtag (#) pero dejar el texto
    text = re.sub(r'#', '', text)
    
    # 4. Eliminar caracteres especiales de control (saltos de línea, tabs)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # 5. Normalizar espacios (eliminar dobles espacios y espacios al inicio/final)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    input_file = "dataset_entrenamiento/EXIST_Unificado_ES.csv"
    output_file = "dataset_entrenamiento/EXIST_Unificado_ES_limpio_final.csv"
    
    # Cargar datos
    df = pd.read_csv(input_file)
    print(f"Tweets iniciales: {len(df)}")

    # 1. Limpiar el texto directamente en la columna 'text'
    df['text'] = df['text'].apply(clean_text)
    
    # 2. Eliminar filas que hayan quedado vacías tras la limpieza
    df = df[df['text'] != '']
    
    # 3. Eliminar duplicados exactos de texto (crucial para no sesgar el modelo)
    # Al limpiar (quitar URLs/menciones), tweets que parecían distintos pueden volverse idénticos
    filas_antes = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    filas_despues = len(df)
    
    print(f"Duplicados eliminados: {filas_antes - filas_despues}")
    print(f"Tweets finales: {filas_despues}")
    
    # Guardar manteniendo todas las columnas originales
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Dataset limpio guardado como: {output_file}")

if __name__ == "__main__":
    main()
