#  Sistema de Detección de Sexismo en Texto

> **Trabajo de Fin de Grado** — Universidad de Jaén  
> Autora: Ana Jurado

Sistema híbrido que combina un **clasificador RoBERTa-BNE** (fine-tuned) con el modelo generativo **Gemma 3 12B** para detectar contenido sexista en español, generar explicaciones del sesgo detectado y proponer contranarrativas constructivas.

---

##  ¿Qué hace este sistema?

El sistema analiza textos en español y determina si contienen lenguaje sexista. Funciona en dos fases:

### Fase 1 — Clasificación automática (RoBERTa-BNE)
Un modelo [RoBERTa-BNE](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne) fine-tuned sobre datos del shared task [EXIST](http://nlp.uned.es/exist/) (2021 + 2023) clasifica el texto como **sexista** o **no sexista**, devolviendo una puntuación de confianza (0 a 1).

### Fase 2 — Explicación con LLM (Gemma 3 12B)
Si el texto es clasificado como sexista, el modelo Gemma genera:
- Una **explicación** de por qué la frase es sexista.
- Una **contranarrativa** que responde al discurso de forma educativa y constructiva.

Si el texto es clasificado como no sexista, Gemma explica por qué la frase es respetuosa.

---
## Demostración de Uso

<p align="center">
  <video src="assets/demo.mp4" controls="controls" style="max-width: 100%;">
    Tu navegador no soporta la reproducción de vídeos.
  </video>
</p>

##  Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    Interfaz Web (Flask)                  │
│              HTML + CSS + JavaScript                    │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /api/analyze
                       ▼
┌──────────────────────────────────────────────────────────┐
│                  Pipeline Híbrido                        │
│                                                          │
│  ┌─────────────────────┐    ┌──────────────────────────┐ │
│  │  RoBERTa-BNE        │    │  Gemma 3 12B (LLM)      │ │
│  │  (Clasificación     │───▶│  (Explicación +          │ │
│  │   binaria)          │    │   Contranarrativa)       │ │
│  └─────────────────────┘    └──────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## Componentes del Proyecto

### 🌐 `web_app/` — Aplicación Web
Interfaz interactiva desarrollada con **Flask** donde el usuario introduce un texto y recibe el análisis completo. Incluye:
- **`app.py`** — Servidor Flask con el endpoint `/api/analyze` que orquesta el pipeline completo.
- **`utils.py`** — Wrapper para la API del LLM (`LLMApi`) y el clasificador local (`SexismClassifier`).
- **`prompts.py`** — Prompts estructurados con estrategias de *In-context Learning*:
  - **0-shot**: Sin ejemplos, el modelo genera la respuesta directamente.
  - **1-shot**: Un ejemplo de referencia para guiar al modelo.
  - **Few-shot**: 5 ejemplos variados (sexismo hostil, benevolente, mansplaining, asimetría lingüística...).

### 🤖 `encoder/` — Entrenamiento del Modelo RoBERTa-BNE
- **`entrenamiento_modelo.py`** — Script de fine-tuning del modelo RoBERTa-BNE sobre los datasets EXIST.
- **`evaluar.py`** — Evaluación del modelo con métricas de rendimiento.

### `estrategias/` — Experimentos de ML Clásico
Scripts de experimentación con técnicas clásicas de Machine Learning, usados como baseline para comparar con el modelo deep learning:
- **TF-IDF** como método de representación textual.
- **Logistic Regression** y **Random Forest** como clasificadores.

### 🧠 `modelos/` — Módulos del LLM
- **`llm_api.py`** — Cliente para comunicarse con el servidor de Gemma (compatible con API de OpenAI).
- **`contranarrativa.py`** — Generación de contranarrativas mediante el LLM.
- **`modelo_gemma.py`** — Integración completa con el modelo Gemma.

---

## 🛠️ Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| Clasificador | RoBERTa-BNE (Hugging Face Transformers) |
| LLM | Gemma 3 12B |
| Backend web | Flask |
| Frontend | HTML + CSS + JavaScript |
| ML clásico | scikit-learn (TF-IDF, Logistic Regression, Random Forest) |
| Gestión de prompts | In-context Learning (0-shot, 1-shot, few-shot) |
| Datos | EXIST 2021, EXIST 2023 |

---

## 📊 Modelo Clasificador

El modelo RoBERTa-BNE fine-tuned está disponible públicamente en Hugging Face:

 **[anajurado/roberta-bne-sexism-detection](https://huggingface.co/anajurado/roberta-bne-sexism-detection)**

Ha sido entrenado sobre datos del shared task EXIST (ediciones 2021 y 2023), que contienen textos reales en español etiquetados como sexistas o no sexistas.

---

## 📁 Estructura del Repositorio

```
sistema_sexismo/
│
├── web_app/                       # 🌐 Aplicación web 
│   ├── app.py                     #     Servidor Flask y endpoints
│   ├── utils.py                   #     Clasificador RoBERTa + wrapper LLM
│   ├── prompts.py                 #     Prompts (0-shot, 1-shot, few-shot)
│   ├── templates/index.html       #     Interfaz de usuario
│   ├── static/                    #     CSS, JS e imágenes
│
├── encoder/                       # 🤖 Entrenamiento y evaluación del modelo
│   ├── entrenamiento_modelo.py
│   └── evaluar.py
│
├── estrategias/                   # 🧪 Experimentos ML clásico (baselines)
│   └── tf_idf.py
│
├── modelos/                     
│   ├── llm_api.py
│   └── modelo_gemma.py
│
├── README.md
├── requirements.txt

```

---

