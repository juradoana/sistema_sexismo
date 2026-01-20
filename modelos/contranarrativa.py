import os
import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from omegaconf import OmegaConf
from json_repair import json_repair

# Importamos tu clase existente
from llm_api import LLMApi

class GeneradorContranarrativas:
    def __init__(self, config_path: str, input_csv: str):
        # 1. Cargar configuración
        self.config = OmegaConf.load(config_path)
        self.api_key = self.config.llm_api.api_key
        self.model = self.config.llm_api.model
        
        # Inicializar conexión a la API
        self.llm = LLMApi(api_key=self.api_key, model=self.model)
        
        self.input_csv = input_csv
        # Nombre del archivo de salida
        self.output_csv = input_csv.replace(".csv", "_CONTRANARRATIVAS.csv")
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.write_lock = Lock()

        # 2. Definir el Prompt del Sistema
        # Diseñado para convertir la explicación del sexismo en una respuesta educativa
        self.system_prompt = """Eres un experto en intervención contra el discurso de odio y comunicación inclusiva.
Tu objetivo es generar una "contranarrativa" (respuesta constructiva) para un tweet que ha sido identificado como sexista.

Instrucciones:
1. **Analiza** por qué el tweet es sexista (se te dará el contexto).
2. **Genera una respuesta** que neutralice el odio usando argumentos lógicos, empatía o datos.
3. **Tono**: Calmado, firme y educativo. Nunca agresivo ni insultante.
4. **Formato**: Estilo tweet (máximo 280 caracteres).
5. **Salida**: Devuelve SOLO un objeto JSON.

Ejemplo de JSON de salida:
{
    "contranarrativa": "Aquí va tu respuesta al tweet...",
    "estrategia": "Breve descripción de la táctica (ej: Reencuadre, Aporte de datos)"
}
"""

    def generar_respuesta(self, text: str, tweet_id: str, razones: str, max_retries: int = 3) -> dict:
        """
        Genera la contranarrativa usando el texto original y la razón del sexismo.
        """
        
        # Construimos el prompt del usuario inyectando el contexto que ya tienes
        prompt_usuario = f"""Tweet sexista: "{text}"
Contexto (por qué es sexista): {razones}

Genera una contranarrativa."""

        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_usuario}
        ]

        for attempt in range(max_retries):
            try:
                # Temperature 0.7 para dar variedad y naturalidad a la respuesta
                raw_response = self.llm.send_request(
                    chat=chat, 
                    max_tokens=250, 
                    temperature=0.7
                )
                
                response_text = raw_response.json()["choices"][0]["message"]["content"]
                
                # Usamos json_repair para arreglar posibles errores de formato del LLM
                parsed = json_repair.loads(response_text)
                
                return {
                    "contranarrativa": parsed.get("contranarrativa", ""),
                    "estrategia": parsed.get("estrategia", "No especificada"),
                    "error": None
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "contranarrativa": None,
                        "estrategia": None,
                        "error": f"Fallo tras {max_retries} intentos: {str(e)[:100]}"
                    }
                time.sleep(1) # Espera breve antes de reintentar

    def process_tweets(self, max_workers: int = 4):
        self.logger.info(f"Leyendo archivo: {self.input_csv}")
        
        # 1. Recuperar estado anterior (si se cortó la ejecución)
        existing_ids = set()
        if os.path.exists(self.output_csv):
            with open(self.output_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_ids = {row['tweet_id'] for row in reader if row.get('tweet_id')}
            self.logger.info(f"Encontrados {len(existing_ids)} tweets ya procesados. Se saltarán.")

        # 2. Filtrar tweets candidatos
        tweets_to_process = []
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Leemos la columna 'pred_sexista' de tu CSV
                # Convertimos "True"/"False" string a booleano real
                es_sexista = str(row.get('pred_sexista', '')).strip() == 'True'
                
                tweet_id = row.get('tweet_id')
                
                # Solo procesamos si es sexista Y no lo hemos hecho ya
                if es_sexista and tweet_id not in existing_ids:
                    tweets_to_process.append({
                        'tweet_id': tweet_id,
                        'text': row.get('text', ''),
                        'razones': row.get('pred_razones', '') # Info clave para el modelo
                    })

        if not tweets_to_process:
            self.logger.info("No hay nuevos tweets sexistas para procesar.")
            return

        self.logger.info(f"Generando contranarrativas para {len(tweets_to_process)} tweets...")

        # 3. Procesamiento en paralelo
        with open(self.output_csv, 'a', encoding='utf-8', newline='') as f:
            fieldnames = ['tweet_id', 'text', 'razones_sexismo', 'contranarrativa', 'estrategia', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Escribir cabecera solo si el archivo es nuevo
            if not existing_ids:
                writer.writeheader()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Lanzamos las tareas
                future_to_tweet = {
                    executor.submit(self.generar_respuesta, tw['text'], tw['tweet_id'], tw['razones']): tw 
                    for tw in tweets_to_process
                }

                # Barra de progreso
                for future in tqdm(as_completed(future_to_tweet), total=len(tweets_to_process), desc="Generando"):
                    tweet = future_to_tweet[future]
                    try:
                        result = future.result()
                        
                        # Escritura thread-safe
                        with self.write_lock:
                            writer.writerow({
                                'tweet_id': tweet['tweet_id'],
                                'text': tweet['text'],
                                'razones_sexismo': tweet['razones'],
                                'contranarrativa': result['contranarrativa'],
                                'estrategia': result['estrategia'],
                                'error': result['error']
                            })
                            f.flush() # Guardar en disco inmediatamente
                            
                    except Exception as e:
                        self.logger.error(f"Error crítico en tweet {tweet['tweet_id']}: {e}")

if __name__ == "__main__":
    # Nombres de tus archivos reales
    CONFIG_PATH = "personal_config.yaml"
    INPUT_CSV = "dataset_entrenamiento/salida_EXIST_anotado_final.csv"
    
    if not os.path.exists(INPUT_CSV):
        print(f" Error: No encuentro el archivo '{INPUT_CSV}'")
    else:
        generador = GeneradorContranarrativas(config_path=CONFIG_PATH, input_csv=INPUT_CSV)
        # Ajusta max_workers según la capacidad de tu servidor/API (4 u 8 suele ir bien)
        generador.process_tweets(max_workers=4)
