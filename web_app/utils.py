import os
import requests
import json_repair
from openai import OpenAI
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CLIENTE PARA LA API DEL MODELO DE LENGUAJE
"""
    Esta clase actúa como un puente (Wrapper) para interactuar con el modelo Gemma
    alojado en el servidor privado de la UJA. Implementa compatibilidad con OpenAI.
"""
class LLMApi:
    def __init__(self, api_key, model):
        """
        Inicializa la clase LLMApi con los parámetros necesarios.
        """
        # Endpoint específico del servidor privado
        self.url = "http://ada01.ujaen.es:8080/v1/chat/completions" 
        if not api_key:
            raise ValueError("API Key is required.")
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        # inicialización del cliente compatible con la arquitectura de OpenAI
        self.client = OpenAI(
            api_key=self.api_key,  
            base_url="http://ada01.ujaen.es:8080/v1" 
        )

    def generate_response(self, messages, max_tokens=256, temperature=0.2):
        """
        envía una petición POST al servidor y gestiona la respuesta generativa
        se usa una temperatura baja (0.2) para reducir la alucinación y asegurar respuestas más deterministas y técnicas
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # ejecución de la llamada síncrona mediante la librería requests
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            # extracción del contenido textual de la respuesta del modelo
            content = response.json()['choices'][0]['message']['content']
            
        
            return content
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            traceback.print_exc()
            return None

# CLASIFICADOR LOCAL RoBERTa-BNE
class SexismClassifier:
    """
    Gestiona la carga del modelo RoBERTa-BNE y realiza la clasificación binaria.
    Optimizado para ejecutarse en GPU si está disponible.
    """
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading RoBERTa model from {model_path} on {self.device}...")
        try:
            # carga del tokenizador y los pesos del modelo pre-entrenado
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, text):
        """
        realiza el proceso completo de predicción: Tokenización -> Inferencia -> Softmax.
        """
        # tokenización y truncamiento de la secuencia
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # roberta procesa el texto y genera logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            #convierte lo logits en probabilidades 
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # asignación de etiquetas según el entrenamiento del modelo (0: No sexista, 1: Sexista) y extrae la probabilidad de la clase sexista 
        sexist_prob = probabilities[0][1].item()
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        is_sexist = (predicted_class == 1)
        return is_sexist, sexist_prob
