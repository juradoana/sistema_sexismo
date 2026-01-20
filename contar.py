import pandas as pd
import os

# Nombre de tu archivo de test
test_file = 'datos_test_originales/test_limpio.csv'

if not os.path.exists(test_file):
    print(f"❌ Error: No encuentro el archivo '{test_file}'")
    exit()

try:
    df = pd.read_csv(test_file)
    
    # Intentar detectar la columna de etiqueta (label, task1, sexist, etc.)
    target_col = None
    possible_names = ['label', 'task1', 'sexist', 'hard_label', 'labels']
    
    for col in df.columns:
        if col.lower() in possible_names:
            target_col = col
            break
            
    if not target_col:
        print(f"⚠️ No encuentro una columna de etiqueta clara. Columnas disponibles: {list(df.columns)}")
        # Intentamos usar la segunda columna si no encontramos nombre conocido
        if len(df.columns) >= 2:
            target_col = df.columns[1]
            print(f"   -> Usando '{target_col}' por defecto.")

    if target_col:
        print(f"📊 Análisis de '{test_file}' (Total: {len(df)} filas)")
        print("-" * 40)
        
        counts = df[target_col].value_counts()
        percents = df[target_col].value_counts(normalize=True) * 100
        
        # Mostrar tabla bonita
        stats = pd.DataFrame({'Cantidad': counts, '% Porcentaje': percents})
        print(stats.round(2))
        
        print("-" * 40)
        
        # Alerta si hay desbalanceo extremo (>90% de una clase)
        if percents.max() > 90:
            print("⚠️ ALERTA: Tu test está MUY desbalanceado (casi todo es de una clase).")
            print("   Esto explica por qué los modelos base dan 0% o 99%.")
        else:
            print("✅ El balance parece razonable para evaluar.")
            
except Exception as e:
    print(f"❌ Error leyendo el CSV: {e}")
