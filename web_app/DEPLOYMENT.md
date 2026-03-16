# Cómo compartir tu Web (Hacerla pública)

Actualmente, tu web funciona en `localhost` (tu ordenador). Para que otras personas puedan entrar desde sus móviles u ordenadores sin estar en tu casa, la opción más fácil para un TFG es usar **ngrok**.

## Opción Recomendada: ngrok (Fácil y Rápido)

Esta herramienta crea un "túnel" seguro desde internet hasta tu ordenador. No necesitas subir tu modelo pesado a ningún servidor.

### Pasos:

1. **Crear cuenta**: Regístrate gratis en [ngrok.com](https://ngrok.com).
2. **Descargar**: Baja la versión para Windows.
3. **Instalar**: Descomprime el archivo descargado.
4. **Conectar tu cuenta**:
   - En el panel de ngrok (dashboard), copia tu "Authtoken".
   - Abre la terminal (o el archivo descomprimido) y escribe:
     ```cmd
     ngrok config add-authtoken TU_TOKEN_AQUI
     ```
5. **Lanzar la web al mundo**:
   - Asegúrate de que tu web ya está funcionando (ejecuta `run_app.bat` primero).
   - Abre **otra** terminal (ventana negra) y escribe:
     ```cmd
     ngrok http 8000
     ```

### ¡Listo!
Verás una línea que dice **Forwarding**. Copia esa dirección web (algo como `https://a1b2-c3d4.ngrok-free.app`).

**Ese es tu enlace público.** Pásaselo a quien quieras y podrán usar tu detector de sexismo mientras tengas tu ordenador encendido.

---

## Opción Avanzada: Servidor en la Nube (Render/PythonAnywhere)

⛔ **No recomendada para este proyecto** porque:
1. Tu modelo (`final_model`) pesa casi 500MB. Los servidores gratuitos suelen tener límites de espacio muy bajos.
2. La API de la universidad (`ada01...`) podría estar protegida y no funcionar desde servidores externos (EEUU/Europa).
3. Es mucho más difícil de configurar.

Quédate con **ngrok** para presentaciones y pruebas.
