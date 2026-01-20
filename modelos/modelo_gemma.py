import os
import json
import time
import csv
import logging
from llm_api import LLMApi
from omegaconf import OmegaConf
from json_repair import json_repair
from tqdm import tqdm
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re
from lingua import Language, LanguageDetectorBuilder


# Mantenemos el detector de español (útil para filtrar spam)
class LinguaSpanishDetector:
    def __init__(self):
        # Instalar: pip install lingua-language-detector
        from lingua import Language, LanguageDetectorBuilder
        self.detector = LanguageDetectorBuilder.from_languages(
            Language.SPANISH, Language.ENGLISH, Language.FRENCH, 
            Language.PORTUGUESE, Language.ITALIAN, Language.CATALAN
        ).with_minimum_relative_distance(0.1).build()
        
        # Palabras comunes en español para heurística rápida
        self.spanish_indicators = {
            'que', 'qué', 'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del', 'se', 'por', 'un', 'para'
        }
    
    def is_spanish(self, text: str, confidence_threshold: float = 0.7) -> bool:
        """Detecta si el texto está en español."""
        if not text or len(text) < 20:
            return False
        
        # Limpieza básica
        text_clean = re.sub(r'[^\w\sáéíóúñü]', '', text.lower())
        
        # Heurística rápida: si tiene ñ o tildes, es probablemente español
        if any(c in text for c in 'ñáéíóú'):
            return True
        
        # Verificar si tiene palabras comunes en español
        words = set(text_clean.split())
        if len(words.intersection(self.spanish_indicators)) > 3:
            return True
        
        # Lingua detector como backup, para evitar falsos positivos 
        confidence = self.detector.compute_language_confidence(text, Language.SPANISH)
        return confidence > confidence_threshold # devuelve un número entre 0 y 1 que indica la confianza en que es español


class SexismoAnnotator:
    def __init__(self, config_path: str, input_csv: str):
        """
        Annotator para detectar sexismo en tweets usando LLM.
        
        :param config_path: Ruta a personal_config.yaml
        :param input_csv: Ruta al CSV limpio (EXIST2021_limpio.csv)
        """
        # Cargar configuración desde el archivo personal_config.yaml
        self.config = OmegaConf.load(config_path)
        self.api_key = self.config.llm_api.api_key
        self.model = self.config.llm_api.model
        
        # Inicializar LLM (usando tu LLMApi)
        self.llm = LLMApi(api_key=self.api_key, model=self.model)
        
        # Archivos de entrada/salida
        self.input_csv = input_csv
        self.output_folder = os.path.dirname(input_csv) 
        self.output_csv = os.path.join(self.output_folder, "salida_EXIST_anotado_final.csv")
        
        # Logging para mostrar mensajes informativos, advertencias y errores
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Prompt SYSTEM para sexismo 
        self.prompt = """Eres un experto en análisis de discurso sexista y misógino en redes sociales.
        Tu tarea es clasificar si un tweet es **sexista** o **no sexista**.
        
        **Definición de sexista**: 
        Contenido que denigra, menosprecia, generaliza negativamente sobre las mujeres por ser mujeres, 
        o refuerza estereotipos dañinos de género.
        
        **Definición de NO sexista**:
        Comentarios neutros, positivos, críticas legítimas a personas sin generalización de género, 
        discusiones académicas, apoyo a la igualdad, o lenguaje inclusivo.
        
        **Criterios de evaluación**:
        1. ¿Contiene insultos o expresiones despectivas dirigidas específicamente a mujeres?
        2. ¿Promueve estereotipos negativos sobre capacidades femeninas?
        3. ¿Usa lenguaje que objetifica o sexualiza de manera despectiva?
        4. ¿Justifica o minimiza la discriminación de género?
        5. ¿Usa sarcasmo o humor para encubrir sexismo?
        
        **Ejemplos de sexismo**:
        - "Las mujeres no sirven para la ciencia" (estereotipo)
        - "Vuelve a la cocina" (denigración)
        - "Las chicas solo sirven para..." (objetificación)
        
        **NO es sexismo**:
        - Críticas legítimas a personas individuales sin generalización de género
        - Comentarios sobre igualdad de género positivos
        - Discusiones académicas sobre roles de género sin lenguaje despectivo
        
        **Instrucciones de salida**:
        Responde SOLO con un objeto JSON exacto:
        {
          "sexista": true/false,
          "razones": "Breve explicación (máx 200 caracteres)"
        }
        """
        
        # Detector de español (opcional pero útil)
        self.spanish_detector = LinguaSpanishDetector()
        
        # Lock para escritura segura en paralelo
        self.write_lock = Lock()
        
    def annotate_tweet(self, text: str, tweet_id: str, max_retries: int = 3) -> dict:
        """
        Llama al LLM para clasificar un tweet.
        Devuelve: {"sexista": bool, "razones": str}
        """
        if not text or len(text.strip()) < 5: #filtrar textos vacíos o muy cortos, si es así devuelve error
            return {
                "sexista": None,
                "razones": "Texto vacío o muy corto",
                "error": "Texto inválido"
            }
        
        # Verificar si es español (opcional, para filtrar ruido)
        '''if not self.spanish_detector.is_spanish(text):
            return {
                "sexista": None,
                "razones": "Texto no identificado como español",
                "error": "Idioma no español"
            }'''
        # construir la entrada de chat para enviar al LLM
        chat = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Tweet ID {tweet_id}: {text}"}
        ]
        # se hace hasta 3 intentos, por si hay error de red o el LLM no responde bien
        for attempt in range(max_retries):
            try:
                # Llamada al LLM usando LLMApi.send_request (igual que tag_comment)
                raw_response = self.llm.send_request(chat=chat, max_tokens=150, temperature=0.1)
                response_text = raw_response.json()["choices"][0]["message"]["content"]
                
                # Parsear JSON con json_repair (pq el modelo a veces lo genera con fallos)
                parsed = json_repair.loads(response_text)
                
                # Validar estructura
                if "sexista" in parsed and "razones" in parsed:
                    return {
                        "sexista": bool(parsed["sexista"]),
                        "razones": str(parsed["razones"])[:200],
                        "error": None
                    }
                else:
                    raise ValueError(f"JSON incompleto. Claves: {list(parsed.keys())}")
                    
            except Exception as e:
                if attempt == max_retries - 1: # si no funciona tras todos los intentos: error definitivo 
                    return {
                        "sexista": None,
                        "razones": f"Error tras {max_retries} intentos: {str(e)[:150]}",
                        "error": "LLM Error"
                    }
                time.sleep(1)  # Esperar antes de reintentar
    
    # Método para evitar tweets repetidos    
    def get_existing_ids(self) -> set:
        """Obtiene IDs de tweets ya procesados para no repetir."""
        if not os.path.exists(self.output_csv): # si no existe el archivo de salida, devuelve conjunto vacío
            return set()
        
        try:
            with open(self.output_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f) # lee cada fila como un diccionario
                return {row['tweet_id'] for row in reader if row['tweet_id']} # crea y devuelve un conjunto con todos los tweet_id ya procesados
        except Exception as e:
            self.logger.warning(f"No se pudo leer CSV existente: {e}")
            return set()
    
    # procesa todo el conjunto de datos en paralelo
    def process_tweets(self, max_workers: int = 4):
        """
        Lee tweets desde CSV, los clasifica y guarda resultados.
        """
        self.logger.info(f"Iniciando procesamiento de {self.input_csv}")
        
        # 1. Leer tweets existentes
        existing_ids = self.get_existing_ids() # obtiene los IDs ya procesados
        self.logger.info(f"Encontrados {len(existing_ids)} tweets ya procesados")
        
        # 2. Cargar tweets nuevos desde CSV, construye la lista de los que faltan
        tweets_to_process = []
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tweet_id = row.get('id', row.get('tweet_id', f"tw_{len(tweets_to_process)}"))
                text = row.get('texto_limpio', row.get('text', ''))
                
                if tweet_id not in existing_ids and text.strip(): # solo si no está ya procesado y el texto no está vacío
                    tweets_to_process.append({ # guarda tweet_id, texto y etiqueta real (si existe)
                        'tweet_id': tweet_id,
                        'text': text,
                        'true_label': row.get('task1', 'unknown')  # Para evaluar después
                    })
        
        if not tweets_to_process:
            self.logger.info("No hay tweets nuevos para procesar")
            return
        
        self.logger.info(f"Procesando {len(tweets_to_process)} tweets")
        
        # 3. Procesar en paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor: #crea el ejecutor de hilos 
            # Submit tareas
            future_to_tweet = { #lanza las tareas en paralelo
                executor.submit(self.annotate_tweet, tw['text'], tw['tweet_id']): tw #coge la función y los argumentos del tweet, y lanza esa tarea en uno de los hilos disponibles 
                for tw in tweets_to_process
            }
            
            # Abrir CSV de salida (append)
            # tweet_id: identificador único del tweet
            # text: texto del tweet
            # true_label: etiqueta real del dataset
            # pred_sexista: predicción del modelo (True/False)
            # pred_razones: explicación breve sobre por qué o por qué no es sexista
            # error: si hubo error, descripción breve
            with open(self.output_csv, 'a', encoding='utf-8', newline='') as f:
                fieldnames = ['tweet_id', 'text', 'true_label', 'pred_sexista', 'pred_razones', 'error']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Escribir header solo si archivo nuevo
                if not existing_ids:
                    writer.writeheader()
                
                # Procesar resultados con barra de progreso
                for future in tqdm(as_completed(future_to_tweet), total=len(tweets_to_process), desc="Clasificando"): # tqdm para barra de progreso
                    tweet = future_to_tweet[future] #va recogiendo los resultados a medida que se van completando 
                    try:
                        result = future.result() # espera a que esa tarea termine y recoge el resultado
                        
                        # Escribir inmediatamente (thread-safe)
                        with self.write_lock: # asegura que solo un hilo escribe a la vez
                            writer.writerow({
                                'tweet_id': tweet['tweet_id'],
                                'text': tweet['text'],
                                'true_label': tweet['true_label'],
                                'pred_sexista': result['sexista'],
                                'pred_razones': result['razones'],
                                'error': result['error'] # si algo sale mal, ese error se captura y se guarda en el archivo de salida para ese tweet
                            })
                            f.flush()
                    # este bloque se ejecuta si algo falla al obtener el resultado de la tarea (future.result()) o al escribir       
                    except Exception as e:
                        self.logger.error(f"Excepción en tweet {tweet['tweet_id']}: {e}") #deja constancia del error en el log, indicando qué tweet dio problemas
                        with self.write_lock:
                            writer.writerow({
                                'tweet_id': tweet['tweet_id'],
                                'text': tweet['text'],
                                'true_label': tweet['true_label'],
                                'pred_sexista': None, #porque no hay predicción válida
                                'pred_razones': f"Excepción: {str(e)[:150]}", #resumen del error
                                'error': 'Exception'
                            })
                            f.flush() #para forzar que se escriba en disco inmediatamente
        
        self.logger.info(f"Procesamiento completado. Resultados en {self.output_csv}")

# =============== EJECUCIÓN ===============
if __name__ == "__main__":
    # Configuración
    CONFIG_PATH = "personal_config.yaml"  # Debe tener llm_api.api_key y llm_api.model
    INPUT_CSV = "dataset_entrenamiento/EXIST_Unificado_ES_limpio_final.csv"    # Tu archivo limpio
    
    # Crear anotador
    annotator = SexismoAnnotator(config_path=CONFIG_PATH, input_csv=INPUT_CSV)
    
    # Ejecutar con 4 workers (ajusta según tu CPU)
    annotator.process_tweets(max_workers=4)
