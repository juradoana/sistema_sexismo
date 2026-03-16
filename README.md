# 🛡️ Sistema de Detección de Sexismo en Texto

Sistema híbrido que combina un **clasificador RoBERTa-BNE** (fine-tuned) con el modelo **Gemma 3 12B** para detectar contenido sexista en español, generar explicaciones y proponer contranarrativas.

> **Trabajo de Fin de Grado** — Universidad de Jaén

---

## 📋 Descripción

El sistema funciona en dos fases:

1. **Clasificación automática**: Un modelo RoBERTa-BNE entrenado sobre el dataset [EXIST](http://nlp.uned.es/exist/) clasifica el texto como *sexista* o *no sexista*, devolviendo una puntuación de confianza.
2. **Explicación con LLM**: Si el texto es sexista, Gemma genera una explicación del sesgo detectado y propone una contranarrativa. Si no lo es, explica por qué el texto es respetuoso.

Todo se expone a través de una **interfaz web Flask** donde el usuario puede analizar textos de forma interactiva.

---

## 📁 Estructura del Repositorio

```
sistema_basico/
│
├── web_app/                   # 🌐 Aplicación web (Flask)
│   ├── app.py                 #     Servidor y endpoints API
│   ├── utils.py               #     Clasificador RoBERTa + wrapper LLM
│   ├── prompts.py             #     Prompts para Gemma (0-shot, 1-shot, few-shot)
│   ├── templates/             #     HTML de la interfaz
│   ├── static/                #     CSS, JS e imágenes
│   ├── tests/                 #     Tests unitarios (pytest)
│   └── *.md                   #     Documentación técnica y de despliegue
│
├── encoder/                   # Entrenamiento y evaluación del modelo RoBERTa-BNE
├── estrategias/               # Estrategias ML (TF-IDF, Logistic Regression, Random Forest)
├── modelos/                   # Scripts para Gemma y contranarrativas
├── evaluaciones/              # Evaluación del modelo Gemma
├── graficas/                  # Matrices de confusión (imágenes)
│
├── detector_basico.py         # Detector inicial de prueba
├── dataset_limpio.py          # Limpieza de datasets
├── unir_datos.py              # Unificación de datasets EXIST
├── plan_pruebas.tex           # Plan de pruebas (LaTeX)
│
├── requirements.txt           # Dependencias del proyecto
├── config.example.py          # Plantilla de configuración (sin credenciales)
├── personal_config.yaml.example  # Plantilla de config YAML (sin credenciales)
└── .gitignore
```

> **Nota:** Los datasets y el modelo entrenado no se incluyen en el repositorio por su tamaño. Ver la sección [Modelo y Datos](#-modelo-y-datos) para instrucciones.

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/sistema_basico.git
cd sistema_basico
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate       # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar credenciales

```bash
cp personal_config.yaml.example personal_config.yaml
```

Edita `personal_config.yaml` según tu situación:

**Si tienes acceso al servidor de la UJA:**
```yaml
llm_api:
  api_key: "tu_api_key_uja"
  model: "/mnt/beegfs/sinai-data/google/gemma-3-12b-it"
  base_url: "http://ada01.ujaen.es:8080/v1"
```

**Si eres usuario externo** (usa [Google AI Studio](https://aistudio.google.com/) — gratis):
```yaml
llm_api:
  api_key: "tu_api_key_de_google"
  model: "gemma-3-12b-it"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
```

> **Nota:** Si no configuras las credenciales, la aplicación funciona igualmente — solo la clasificación de RoBERTa, sin las explicaciones de Gemma.

### 4. Obtener el modelo clasificador

Descarga el modelo desde Hugging Face y colócalo en la carpeta `final_model/`:

```bash
# Opción 1: Con git (requiere git-lfs)
git clone https://huggingface.co/anajurado/roberta-bne-sexism-detection final_model

# Opción 2: Con la CLI de Hugging Face
pip install huggingface_hub
huggingface-cli download anajurado/roberta-bne-sexism-detection --local-dir final_model
```

También puedes descargarlo manualmente desde: **[anajurado/roberta-bne-sexism-detection](https://huggingface.co/anajurado/roberta-bne-sexism-detection)**

> Si quieres re-entrenar el modelo, usa `encoder/entrenamiento_modelo.py` con los datasets EXIST.

### 5. Ejecutar la aplicación

```bash
cd web_app
python app.py
```

Abre tu navegador en **http://127.0.0.1:8000**

---

## 🧪 Tests

```bash
cd web_app
python -m pytest tests/ -v
```

---

## 📊 Modelo y Datos

### Modelo clasificador

El modelo es un **RoBERTa-BNE** fine-tuned sobre datos del shared task EXIST (2021 + 2023). Está disponible en Hugging Face:

👉 **[anajurado/roberta-bne-sexism-detection](https://huggingface.co/anajurado/roberta-bne-sexism-detection)**

### Datasets

Los datos proceden del shared task **EXIST** (sEXism Identification in Social neTworks):
- [EXIST 2021](http://nlp.uned.es/exist2021/)
- [EXIST 2023](http://nlp.uned.es/exist2023/)

Los datasets no se incluyen por restricciones de tamaño y licencia. Descárgalos desde las páginas oficiales y colócalos en `dataset_entrenamiento/`.

---

## 🛠️ Tecnologías

| Componente | Tecnología |
|---|---|
| Clasificador | RoBERTa-BNE (Hugging Face Transformers) |
| LLM | Gemma 3 12B (API OpenAI-compatible) |
| Backend web | Flask |
| Frontend | HTML + CSS + JavaScript |
| ML clásico | scikit-learn (TF-IDF, LR, Random Forest) |
| Datos | EXIST 2021, EXIST 2023 |

---

## 📄 Licencia

Este proyecto forma parte de un Trabajo de Fin de Grado y se comparte con fines académicos.
