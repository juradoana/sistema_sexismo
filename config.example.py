# config.example.py
# Copia este archivo como config.py y rellena con tus credenciales

# API KEY (necesaria para generar explicaciones con Gemma)
API_KEY = "TU_API_KEY_AQUI"

# Modelo LLM
MODEL = "nombre_o_ruta_del_modelo_gemma"

# Configuración de procesamiento
SAMPLE_SIZE = 100  # Cambiar a None para procesar todo
TEMPERATURE = 0.1
MAX_RETRIES = 3
SAVE_EVERY = 50

# Archivos de entrada
TRAIN_FILE = 'dataset_entrenamiento/EXIST2021_limpio.csv'
TEST_FILE = 'dataset_entrenamiento/EXIST2021_test_limpio.csv'

# Archivos de salida
OUTPUT_FILE = 'resultados_gemma/task1_predictions.csv'
METRICS_FILE = 'resultados_gemma/metricas.csv'
PARTIAL_FILE = 'resultados_gemma/partial_results.csv'
