import os
import cv2
import gdown
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Ruta y Google Drive ID del modelo
MODEL_PATH = "model/best.pt"
GDRIVE_ID = "1HWwWKpr8vK-92ArBIy83YcatO97H1RpB"

# Descargar modelo si no existe
def descargar_modelo_si_no_existe():
    if not os.path.exists(MODEL_PATH):
        print("📥 Descargando modelo desde Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Modelo descargado.")
    else:
        print("🟢 Modelo ya existe en:", MODEL_PATH)

descargar_modelo_si_no_existe()
model = YOLO(MODEL_PATH)

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o especifica dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta principal
@app.get("/")
async def root():
    return {"message": "API de inspección activa"}

# Ruta de inspección
@app.post("/inspeccionar")
async def inspeccionar(file: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Ejecutar detección
        results = model(image)

        # Extraer y formatear detecciones
        detecciones = []
        for r in results:
            for box in r.boxes:
                clase_id = int(box.cls[0])
                clase = model.names[clase_id]
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                detecciones.append({
                    "clase": clase,
                    "confianza": round(conf, 2),
                    "coordenadas": coords
                })

        return {"defectos": detecciones}
    
    except Exception as e:
        print("❌ Error en inspección:", e)
        return {"error": str(e)}
