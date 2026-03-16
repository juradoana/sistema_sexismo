# Resumen Final del Desarrollo del Proyecto (TFG)

Este documento resume todo el trabajo realizado para transformar el modelo de detección de sexismo en una aplicación web funcional, interactiva y accesible.

## 1. Objetivo Principal
Crear una interfaz amigable que permita a cualquier usuario interactuar con el modelo de Inteligencia Artificial desarrollado, sin necesidad de saber programación. El sistema debe no solo detectar sexismo, sino **explicar por qué** y **ofrecer una alternativa**.

## 2. Componentes Desarrollados

### A. Interfaz Web (Frontend)
Se ha creado una página web moderna y profesional (`index.html`, `style.css`, `script.js`) con las siguientes características:
*   **Diseño Limpio**: Colores suaves, tipografía clara y estructura centrada en la experiencia de usuario (UX).
*   **Interactividad**: Uso de barras de progreso para mostrar la "Confianza" del modelo.
*   **Feedback Visual**: Diferenciación clara entre resultados "Sexistas" (Rojo) y "No Sexistas" (Verde).
*   **Selector de Estrategia**: Un desplegable para elegir cómo debe comportarse el LLM (Zero-shot, Few-shot, etc.), permitiendo experimentar con diferentes niveles de "inteligencia".

### B. Servidor Backend (Flask)
Se ha implementado un servidor en Python (`app.py`) que actúa como cerebro de la operación:
*   **Carga Optimizada**: El modelo RoBERTa (`final_model`) se carga una sola vez al iniciar, haciendo que las predicciones sean instantáneas.
*   **Orquestación de Modelos**:
    1.  Recibe el texto.
    2.  Consulta a **RoBERTa** (¿Es sexista?).
    3.  Si es sexista, consulta a **Gemma** (¿Por qué? + Contranarrativa).
    4.  Empaqueta todo y lo envía a la web.

### C. Ingeniería de Prompts (Prompt Engineering)
Se ha trabajado a fondo en cómo "hablar" con el LLM (`prompts.py`) para obtener resultados consistentes:
*   **Formato Estructurado**: Se fuerza al modelo a responder siempre en formato JSON. Esto permite separar programáticamente la "Explicación" de la "Contranarrativa".
*   **Estrategia Few-Shot**: Se le dan ejemplos reales (lingüística, benevolente, hostil) al modelo antes de preguntar, mejorando drásticamente la calidad de sus respuestas.
*   **Control de Verbosidad**: Se ajustaron los límites de tokens (1500) y las instrucciones para evitar que las respuestas se corten a medias.

## 3. Retos Técnicos Resueltos

### Problema: El modelo pesaba mucho para un servidor gratuito.
*   **Solución**: Se optó por una ejecución **Local** + **Túnel**. La web corre en tu ordenador potente, y usamos **ngrok** para abrir una "ventana" segura a internet.
*   **Script de Automatización**: Se crearon `run_app.bat` y `run_ngrok.bat` para que todo este proceso complejo se realice con dos simples clics, configurando automáticamente dependencias y autenticación.

### Problema: Respuestas cortadas o con errores de formato.
*   **Solución**: Se implementó una librería de "reparación de JSON" (`json_repair`) que es capaz de entender la respuesta del LLM incluso si tiene pequeños errores de sintaxis, haciendo el sistema mucho más robusto.

## 4. Archivos Entregables

*   `web_app/`: Carpeta con todo el código fuente de la web.
*   `MEMORIA_TECNICA.md`: Documentación detallada de la arquitectura.
*   `README.md`: Manual de usuario simplificado.
*   `run_app.bat`: Script de inicio del sistema.
*   `run_ngrok.bat`: Script de conexión a internet.

## 5. Conclusión
El sistema ha pasado de ser un script de análisis de datos a una **aplicación web completa**, capaz de ser demostrada en tiempo real desde cualquier dispositivo móvil, cumpliendo con los requisitos de divulgación y usabilidad de un TFG.
