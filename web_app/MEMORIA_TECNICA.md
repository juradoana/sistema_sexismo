# Memoria Técnica del Sistema de Detección de Sexismo

Este documento detalla la arquitectura, tecnologías y decisiones de diseño tomadas para el desarrollo de la aplicación web del Trabajo de Fin de Grado (TFG).

## 1. Introducción y Objetivos
El objetivo del sistema es proporcionar una interfaz accesible para la detección automática de sexismo en textos cortos, utilizando modelos de lenguaje avanzados. El sistema no solo clasifica (Sexista / No Sexista), sino que proporciona una **explicación pedagógica** y una **contranarrativa** constructiva.

## 2. Arquitectura del Sistema
El sistema sigue una arquitectura **Cliente-Servidor** clásica, diseñada para ser ejecutada localmente debido al tamaño de los modelos.

### Diagrama de Flujo de Datos
1.  **Entrada**: El usuario introduce una frase en la Interfaz Web.
2.  **Solicitud**: El navegador envía un JSON vía `POST` al servidor Flask.
3.  **Detección (Nivel 1)**: El backend invoca al modelo **RoBERTa** (`final_model`).
    *   Si es `NO SEXISTA`: Se devuelve la respuesta inmediatamente.
    *   Si es `SEXISTA`: Se activa el Nivel 2.
4.  **Explicación (Nivel 2)**: El backend construye un "prompt" según la estrategia seleccionada (0-shot, 1-shot, etc.) y consulta al LLM **Gemma** a través de una API local.
5.  **Respuesta**: El backend combina la clasificación y la explicación en un JSON estructurado y lo envía al navegador.

## 3. Tecnologías Utilizadas

### Backend (Servidor)
*   **Lenguaje**: Python 3.12.
*   **Framework Web**: **Flask**. Se eligió por su ligereza y facilidad para integrar librerías de IA.
*   **Gestión de IA**:
    *   `transformers` (Hugging Face): Para cargar y ejecutar el modelo RoBERTa.
    *   `torch` (PyTorch): Motor de cálculo tensorial subyacente.
    *   `openai` (Cliente compatible): Para conectar con el servidor de inferencia del LLM Gemma (`ada01`).

### Frontend (Interfaz)
*   **Tecnologías**: HTML5, CSS3, JavaScript (Vanilla ES6).
*   **Diseño**: Moderno y responsivo, utilizando CSS Grid/Flexbox y paletas de colores semánticas (Rojo/Verde para alertas).
*   **Comunicación**: AJAX (`fetch` API) para enviar datos sin recargar la página, mejorando la experiencia de usuario (UX).

## 4. Desarrollo de Componentes Clave

### A. Controlador Principal (`app.py`)
Es el punto de entrada. Inicializa los modelos al arrancar para evitar tiempos de espera en cada petición ("lazy loading" vs "eager loading"). Define el endpoint `/api/analyze` que orquesta la lógica de decisión.

### B. Módulo de Utilidades (`utils.py`)
Encapsula la complejidad técnica:
*   **Clase `SexismClassifier`**: Carga el modelo RoBERTa y el tokenizador. Normaliza la salida (logits -> softmax) para dar un porcentaje de confianza.
*   **Clase `LLMApi`**: Abstrae la conexión con el servidor de Gemma, manejando cabeceras y autenticación.

### C. Ingeniería de Prompts (`prompts.py`)
Aquí reside la inteligencia "blanda" del sistema. Se diseñaron estrategias de **Prompt Engineering**:
*   *Zero-shot*: Se pide la tarea sin ejemplos.
*   *Few-shot*: Se incluyen ejemplos (JSON) de frases sexistas y sus correcciones para guiar al modelo por imitación (In-Context Learning).
*   **Formato JSON Forzado**: Se instruyó al LLM para responder estrictamente en formato JSON, facilitando que el código pueda separar la "Explicación" de la "Contranarrativa" automáticamente.

## 5. Despliegue y Accesibilidad

### Automatización Local
Se creó un script `run_app.bat` para Windows que:
1.  Verifica la instalación de Python.
2.  Instala automáticamente las dependencias (`requirements.txt`).
3.  Levanta el servidor Flask.
Esto permite la ejecución del proyecto sin conocimientos técnicos profundos de consola.

### Acceso Remoto (ngrok)
Dado el peso del modelo RoBERTa (~500MB) y los requisitos de hardware, no era viable un hosting gratuito en la nube. Se solucionó mediante **ngrok**, que crea un túnel seguro (tunneling) exponiendo el puerto local 8000 a una URL pública de internet, permitiendo demostraciones en tiempo real desde dispositivos móviles.
