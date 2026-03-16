def get_sexism_explanation_prompt(text, strategy="0-shot"):
    """
    Genera los prompts estructurados para el LLM utilizando diferentes 
    estrategias de 'In-context Learning' (0-shot, 1-shot, few-shot).
    El objetivo es guiar al modelo para que genere explicaciones coherentes y contranarrativas.
    """
    # SYSTEM PROMPT: Definición del rol y comportamiento del modelo
    system_prompt = """Eres un experto en lingüística, igualdad de género y análisis del discurso. 
Tu tarea es explicar por qué una frase ha sido clasificada como sexista y generar una contranarrativa constructiva.
Sé educativo, claro y persuasivo. Responde siempre en formato JSON."""

    user_content = f"Frase: '{text}'"

    # ESTRATEGIA 1-SHOT: se proporciona un único ejemplo de referencia
    if strategy == "1-shot":
        example = """
Ejemplo:
Frase: 'Las mujeres no saben conducir, son un peligro.'
Respuesta: {
    "explicacion": "Esta frase es sexista porque generaliza una supuesta incompetencia basada en el género, reforzando estereotipos dañinos sin base real.",
    "contranarrativa": "La habilidad para conducir no depende del género, sino de la formación y la práctica. Las estadísticas demuestran que las mujeres tienen menos accidentes graves."
}
"""
        user_content = example + "\n" + user_content
        
    # ESTRATEGIA FEW-SHOT: se proporcionan varios ejemplos variados
    elif strategy == "few-shot":
        examples = """
Ejemplo 1:
Frase: 'Vete a fregar los platos.'
Respuesta: {
    "explicacion": "Es sexista porque relegar a la mujer al ámbito doméstico como si fuera su única función es una forma clásica de discriminación y desprecio.",
    "contranarrativa": "Las tareas del hogar son responsabilidad de quienes viven en la casa, independientemente de su género."
}

Ejemplo 2:
Frase: 'Corres como una niña.'
Respuesta: {
    "explicacion": "Utiliza 'niña' como un insulto, implicando debilidad o inferioridad física asociada al género femenino.",
    "contranarrativa": "Ser niña no es sinónimo de debilidad. Muchas atletas femeninas demuestran una fuerza y velocidad impresionantes."
}

Ejemplo 3:
Frase: 'Las mujeres son como flores, hay que cuidarlas y no dejar que hagan trabajos duros.'
Respuesta: {
    "explicacion": "Es un caso de sexismo benevolente. Aunque parece un elogio, posiciona a la mujer como un ser frágil y dependiente, justificando su exclusión.",
    "contranarrativa": "Las mujeres no son objetos frágiles, sino personas autónomas con derecho a decidir sus propios desafíos y capacidades."
}

Ejemplo 4:
Frase: 'Cariño, deja que yo te explique cómo funciona este software, es un poco complejo para ti.'
Respuesta: {
    "explicacion": "Presenta una condescendencia basada en el género (mansplaining) al asumir que una mujer tiene menos capacidad técnica.",
    "contranarrativa": "La competencia técnica se basa en el conocimiento y la experiencia, no en el género. Es importante tratar a todos con el mismo respeto intelectual."
}

Ejemplo 5:
Frase: 'Él es un soltero de oro, pero ella ya se está quedando para vestir santos.'
Respuesta: {
    "explicacion": "Muestra una asimetría lingüística donde la soltería se premia en el hombre y se penaliza en la mujer como un fracaso social.",
    "contranarrativa": "El valor de una persona no depende de su estado civil. Ambos géneros tienen el mismo derecho a disfrutar de su autonomía sin juicios."
}
"""
        user_content = examples + "\n" + user_content

    # Instrucción final con llaves escapadas para el f-string
    prompt = f"""
Analiza la siguiente frase que ha sido detectada como sexista:
{user_content}

Salida requerida en formato JSON (asegúrate de que sea un JSON válido):
{{
  "explicacion": "Breve explicación de por qué es sexista.",
  "contranarrativa": "Una respuesta alternativa que desafíe el sexismo de forma educada."
}}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]


def get_non_sexism_explanation_prompt(text, strategy="0-shot"):
    
    
    # Genera prompts para explicar la ausencia de sesgo en frases clasificadas como neutras, para así evitar que el usuario reciba una respuesta vacía ante un caso negativo.
    
    # SYSTEM PROMPT, enfoque en neutralidad lingüística y objetividad
    system_prompt = """Eres un experto en lingüística, igualdad de género y análisis del discurso. 
Tu tarea es explicar por qué una frase ha sido clasificada como NO sexista.
Sé educativo, claro y constructivo. Responde siempre en formato JSON."""

    user_content = f"Frase: '{text}'"

    if strategy == "1-shot":
        example = """
Ejemplo:
Frase: 'Hoy hace muy buen tiempo para pasear.'
Respuesta: {
    "explicacion_no_sexista": "Esta frase es neutra y no contiene ninguna referencia discriminatoria hacia ningún género. Es una simple observación sobre el clima."
}
"""
        user_content = example + "\n" + user_content
        
    elif strategy == "few-shot":
        examples = """
Ejemplo 1:
Frase: 'El equipo trabajó muy bien en este proyecto.'
Respuesta: {
    "explicacion_no_sexista": "Es una frase neutra que valora el trabajo colectivo sin hacer distinciones de género ni atribuir características estereotipadas."
}

Ejemplo 2:
Frase: 'Me gusta leer libros de ciencia ficción.'
Respuesta: {
    "explicacion_no_sexista": "Expresa una preferencia personal sin ningún contenido que discrimine o estereotipe a ningún género."
}

Ejemplo 3:
Frase: 'La reunión será a las 10 de la mañana.'
Respuesta: {
    "explicacion_no_sexista": "Es una comunicación informativa y objetiva que no contiene elementos sexistas ni discriminatorios."
}
"""
        user_content = examples + "\n" + user_content

    prompt = f"""
Analiza la siguiente frase que ha sido detectada como NO sexista:
{user_content}

Salida requerida en formato JSON (asegúrate de que sea un JSON válido):
{{
  "explicacion_no_sexista": "Breve explicación de por qué esta frase NO es sexista."
}}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]