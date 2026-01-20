# paso_3_detector_basico.py
import re

class DetectorSexismoBasico:
    """
    Versión mejorada de tu detector con palabras clave
    """
    
    def __init__(self):
        # Lista ampliada de palabras/frases sexistas
        self.palabras_sexistas = [
            'lugar es la cocina',
            'lugar está en la cocina',
            'cosas de mujeres',
            'cosas de hombres',
            'todas las mujeres',
            'todos los hombres',
            'típico de mujer',
            'típico de hombre',
            'feminazi',
            'histérica',
            'las mujeres no saben',
            'las mujeres son',
            'las mujeres deberían',
        ]
    
    def preprocesar_texto(self, texto):
        """
        Limpia el texto antes de analizarlo
        """
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar caracteres especiales y múltiples espacios
        texto = re.sub(r'[^\w\sáéíóúñü]', '', texto)
        texto = re.sub(r'\s+', ' ', texto)
        
        return texto.strip()
    
    def clasificar_mensaje(self, mensaje):
        """
        Clasifica un mensaje como SEXISTA o NO SEXISTA
        DEVUELVE: diccionario con resultado y explicación
        """
        # Preprocesar
        mensaje_limpio = self.preprocesar_texto(mensaje)
        
        # Buscar palabras sexistas
        palabras_encontradas = []
        for palabra in self.palabras_sexistas:
            if palabra in mensaje_limpio:
                palabras_encontradas.append(palabra)
        
        # Resultado
        if palabras_encontradas:
            return {
                'resultado': 'SEXISTA',
                'confianza': len(palabras_encontradas) / len(self.palabras_sexistas),
                'palabras_encontradas': palabras_encontradas,
                'explicacion': f"Se encontraron {len(palabras_encontradas)} expresiones sexistas"
            }
        else:
            return {
                'resultado': 'NO SEXISTA',
                'confianza': 0.5,  # Baja confianza porque solo busca palabras
                'palabras_encontradas': [],
                'explicacion': "No se encontraron palabras sexistas conocidas"
            }
    
    def modo_interactivo(self):
        """
        Modo consola como tu código original
        """
        print("=" * 60)
        print("🚫 DETECTOR DE SEXISMO - VERSIÓN BÁSICA")
        print("=" * 60)
        print()
        
        while True:
            mensaje = input("📝 Introduce un mensaje (o 'salir'): ")
            
            if mensaje.lower() == 'salir':
                print("\n👋 ¡Hasta pronto!")
                break
            
            resultado = self.clasificar_mensaje(mensaje)
            
            print(f"\n{'─' * 60}")
            print(f"✅ Resultado: {resultado['resultado']}")
            print(f"📊 Confianza: {resultado['confianza']:.1%}")
            print(f"💬 {resultado['explicacion']}")
            if resultado['palabras_encontradas']:
                print(f"🔍 Palabras detectadas: {', '.join(resultado['palabras_encontradas'])}")
            print(f"{'─' * 60}\n")


# PROBAR EL DETECTOR
if __name__ == "__main__":
    detector = DetectorSexismoBasico()
    
    # Opción 1: Modo interactivo
    detector.modo_interactivo()
    
    # Opción 2: Probar con ejemplos directos (comenta/descomenta)
    # ejemplos = [
    #     "Las mujeres no saben conducir",
    #     "Me gusta la pizza",
    #     "Típico de mujer llorar por todo",
    #     "El café está delicioso"
    # ]
    # 
    # for ejemplo in ejemplos:
    #     print(f"Texto: {ejemplo}")
    #     resultado = detector.clasificar_mensaje(ejemplo)
    #     print(f"→ {resultado['resultado']}\n")
