
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generar_informe_tecnico(detecciones):
    if not detecciones:
        return "No se detectaron defectos visuales en la imagen."

    clases_detectadas = [d.get("class", "defecto") for d in detecciones]
    prompt = f"""
Eres un perito técnico de inspección de viviendas. Redacta un informe breve y profesional en lenguaje técnico, basado en los siguientes defectos detectados por un sistema de visión artificial:

{', '.join(clases_detectadas)}

El informe debe ser claro, sin exageraciones, y útil para incluir en un reporte de evaluación del estado de una estancia.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            { "role": "system", "content": "Eres un generador de informes técnicos claros y objetivos para inspección de viviendas." },
            { "role": "user", "content": prompt }
        ]
    )

    return response.choices[0].message.content.strip()
