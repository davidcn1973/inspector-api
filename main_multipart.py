import os
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from utils.report_gen import generar_informe

# Ruta del modelo y URL desde variable de entorno
MODEL_PATH = "model/best.pt"
MODEL_URL = os.getenv("MODEL_URL")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Descargando modelo desde {MODEL_URL}...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        r = requests.get(MODEL_URL, allow_redirects=True)
        if r.status_code != 200:
            raise RuntimeError("No se pudo descargar el modelo.")
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Modelo descargado exitosamente.")

# Descargar si no existe
download_model()

# Cargar modelo
model = YOLO(MODEL_PATH)

# Inicializar API
app = FastAPI()

# CORS para permitir conexión desde React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/inspeccionar")
async def inspeccionar(file: UploadFile = File(...)):
    try:
        results = model.predict(file.file)

        detecciones = []
        if results and results[0].boxes and results[0].boxes.data is not None:
            for r in results[0].boxes.data:
                cls_id = int(r[5].item())
                detecciones.append({
                    "class": model.names[cls_id],
                    "confidence": float(r[4].item())
                })

        informe = generar_informe(detecciones)

        return {
            "defectos_detectados": [d["class"] for d in detecciones],
            "informe_tecnico": informe
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
