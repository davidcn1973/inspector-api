
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.report_gen import generar_informe
from utils.yolo_detector import detectar_defectos_en_base64

app = FastAPI()

class ImagenBase64(BaseModel):
    imagen_base64: str

@app.post("/inspeccionar-base64")
def inspeccionar_base64(data: ImagenBase64):
    try:
        detecciones = detectar_defectos_en_base64(data.imagen_base64)
        informe = generar_informe(detecciones)

        return {
            "defectos_detectados": [d["class"] for d in detecciones],
            "informe_tecnico": informe
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
