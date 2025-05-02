from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from utils.report_gen import generar_informe

app = FastAPI()

# Permitir CORS desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga del modelo YOLO con manejo de errores
try:
    model = YOLO("model/best.pt")
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