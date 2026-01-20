# Lista de palabras clave sexistas
palabras_sexistas = [
    'lugar es la cocina',
    'cosas de mujeres',
    'cosas de hombres',
    'todas las mujeres',
    'todos los hombres',
    'típico de mujer',
    'típico de hombre',
    'feminazi',
    'histérica'
]

def clasificar_mensaje(mensaje):
    """
    Clasifica un mensaje como SEXISTA o NO SEXISTA
    """
    mensaje_lower = mensaje.lower()
    
    for palabra in palabras_sexistas:
        if palabra in mensaje_lower:
            return "SEXISTA"
    
    return "NO SEXISTA"


if __name__ == "__main__":
    print("=== DETECTOR DE SEXISMO ===\n")
    
    while True:
        mensaje = input("Introduce un mensaje (o 'salir' para terminar): ")
        
        if mensaje.lower() == 'salir':
            print("\nHas acabado")
            break
        
        
        
        resultado = clasificar_mensaje(mensaje)
        print(f"Resultado: {resultado}\n")