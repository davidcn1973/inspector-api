from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from utils.report_gen import generar_informe
import os
import gdown

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Descargar el modelo desde Google Drive si no existe
model_path = "model/best.pt"
drive_id = "1HWwWKpr8vK-92ArBIy83YcatO97H1RpB"  # <-- REEMPLAZAR POR TU ID DE DRIVE

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    print("📥 Descargando modelo desde Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={drive_id}", model_path, quiet=False)

# Cargar modelo YOLO
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model = None

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/inspeccionar")
async def inspeccionar(file: UploadFile = File(...)):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo no cargado.")

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
        print(f"❌ Error en /inspeccionar: {e}")
        raise HTTPException(status_code=500, detail=str(e))