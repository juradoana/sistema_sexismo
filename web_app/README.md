# Guía de Uso Rápido - Detector de Sexismo

## ¿Cómo arranco la web?

Tienes dos formas de usarla:

### 1. MODO LOCAL (Solo para ti)
Usa esto si solo vas a trabajar tú en tu ordenador.

1.  Ve a la carpeta `sistema_basico`.
2.  Haz doble clic en **`run_app.bat`**.
3.  Espera a que salga el mensaje `Running on http://127.0.0.1:8000`.
4.  Abre tu navegador y entra en: **http://127.0.0.1:8000**

---

### 2. MODO PÚBLICO (Para enseñar en el móvil a otros)
Usa esto cuando quieras que alguien entre desde su teléfono o desde otro sitio.

1.  **Abre la "Cocina"**:
    *   Haz doble clic en **`run_app.bat`**.
    *   **NO** cierres esa ventana negra.

2.  **Abre el "Repartidor"**:
    *   Haz doble clic en **`run_ngrok.bat`**.
    *   Se abrirá una segunda ventana negra.
    *   Busca donde dice **Forwarding**.
    *   Copia la dirección web que sale ahí (`https://...ngrok-free.app`).

3.  **Comparte**:
    *   Manda esa dirección por WhatsApp o ábrela en el móvil.
    *   **IMPORTANTE**: Debes mantener **LAS DOS VENTANAS NEGRAS ABIERTAS** todo el tiempo que quieras usar la web. Si cierras una, dejará de funcionar.

---

## Solución de Problemas

*   **¿Se cierra ngrok nada más abrirlo?**
    *   Asegúrate de que el archivo `ngrok.exe` está en la misma carpeta.
    *   Mira si ya tienes otra ventana de ngrok abierta (solo puedes tener una).

*   **¿La web da error?**
    *   Reinicia la ventana de `run_app.bat` cerrándola y volviéndola a abrir.
