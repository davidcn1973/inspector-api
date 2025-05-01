
import os
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

def generar_informe(detecciones):
    if not detecciones:
        return "No se han detectado defectos visibles."
    if client is None:
        return "⚠️ No se ha configurado la clave API de OpenAI. No se pudo generar el informe técnico."

    clases_detectadas = [d["class"] for d in detecciones]
    prompt = f"""
Eres un inspector técnico especializado en construcción. 
Genera un informe profesional breve a partir de los siguientes defectos detectados:

{', '.join(clases_detectadas)}

El informe debe describir los problemas de forma técnica y clara, apto para incluir en un reporte de inspección.
"""

    respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            { "role": "system", "content": "Eres un generador de informes técnicos de inspección de viviendas." },
            { "role": "user", "content": prompt }
        ]
    )

    return respuesta.choices[0].message.content.strip()
