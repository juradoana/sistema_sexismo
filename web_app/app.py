
from flask import Flask, render_template, request, jsonify #framework para crear el servidor web
import sys
import os
import yaml
from omegaconf import OmegaConf
import json_repair # para arreglar los json mal formados

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import LLMApi, SexismClassifier # para el clasificador que vamos a usar y la clase para poder usar Gemma
#las funciones que construyen las preguntas para gemma 
from prompts import get_sexism_explanation_prompt, get_non_sexism_explanation_prompt

# inicialización de la aplicación Flask (Micro-framework web)
app = Flask(__name__)

# configuration de las rutas de archivos 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "personal_config.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "final_model")

# variables globales para persistir los modelos en memoria y optimizar el tiempo de respuesta
llm_api = None
classifier = None


# carga API para el LLM y clasificador local.
def load_resources():
    global llm_api, classifier
    
    # carga de configuración mediante OmegaConf
    try:
        if os.path.exists(CONFIG_PATH):
            config = OmegaConf.load(CONFIG_PATH)
            api_key = config.llm_api.api_key
            model_name = config.llm_api.model
            base_url = config.llm_api.get("base_url", "http://ada01.ujaen.es:8080/v1")
            llm_api = LLMApi(api_key, model_name, base_url)
            print("LLM API Initializado")
        else:
            print(f"Archivo de configuración no encontrado en {CONFIG_PATH}")
    except Exception as e:
        print(f"Error al inicializar la API del LLM: {e}")

    # carga del clasificador RoBERTa-BNE
    try:
        if os.path.exists(MODEL_PATH):
            classifier = SexismClassifier(MODEL_PATH)
            print(" Clasificador RoBERTa-BNE inicializado")
        else:
            print(f" Directorio del modelo no encontrado en {MODEL_PATH}")
    except Exception as e:
        print(f"Error al inicializar el clasificador: {e}")

# ejecución de la carga de recursos al arrancar el servidor
load_resources()

# endpoints de la aplicación web
@app.route('/')
def index():
    # renderiza la interfaz de usuario principal (Frontend)
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # Punto de entrada de la API para el análisis de textos
    # implementa pipeline híbrido, clasificación + explicación 

    data = request.json # recoge todo el JSON
    text = data.get('text', '').strip() #saca el texto
    strategy = data.get('strategy', '0-shot') # saca la estrategia elegida por el usuario (0-shot, 1-shot, few-shot)
    
    # validaciones de seguridad y estado del servidor
    if not text:
        return jsonify({"error": "No se proporcionó texto para analizar"}), 400

    if not classifier:
        return jsonify({"error": "El clasificador no se ha cargado correctamente"}), 500

    # Paso 1: clasificación del texto con el modelo RoBERTa-BNE
    try:
        is_sexist, confidence = classifier.predict(text)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    # Estructura base del JSON de respuesta
    result = {
        "is_sexist": is_sexist,
        "confidence": float(confidence),
        "text": text
    }

    # Paso 2: Generación de explicaciones con Gemma3-12b
    # si el API está activo generamos una justificación cualitativa
    if llm_api:
        try:
            if is_sexist:
                # Caso SEXISTA: explicación del sesgo y propuesta de contranarrativa
                messages = get_sexism_explanation_prompt(text, strategy)
                llm_response = llm_api.generate_response(messages, max_tokens=1500)
                
                print(f"DEBUG - Raw LLM Response (sexist): {llm_response}")
                # json_repair porque a veces se añade texto extra al JSON
                parsed_response = json_repair.loads(llm_response)
                
                result["explanation"] = parsed_response.get("explicacion", "No se pudo generar explicación.").replace("**", "")
                result["counter_narrative"] = parsed_response.get("contranarrativa", "No se pudo generar contranarrativa.").replace("**", "")
            else:
                # Caso NO-SEXISTA: Explicación de por qué el texto es neutro o respetuoso
                messages = get_non_sexism_explanation_prompt(text, strategy)
                llm_response = llm_api.generate_response(messages, max_tokens=1500)
                
                print(f"DEBUG - Raw LLM Response (non-sexist): {llm_response}")

                parsed_response = json_repair.loads(llm_response)
                
                result["explanation"] = parsed_response.get("explicacion_no_sexista", "No se pudo generar explicación.").replace("**", "")
                
        except Exception as e:
            print(f"LLM Error: {e}")
            result["explanation"] = "Error al conectar con el asistente inteligente."
            if is_sexist:
                result["counter_narrative"] = "No disponible."

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
