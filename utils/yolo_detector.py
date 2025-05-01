
from ultralytics import YOLO
from PIL import Image
import base64
import io

model = YOLO("model/best.pt")

def detectar_defectos_en_base64(imagen_base64):
    try:
        image_data = base64.b64decode(imagen_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        results = model(image)
        detecciones = []
        if results and results[0].boxes and results[0].boxes.data is not None:
            for r in results[0].boxes.data:
                cls_id = int(r[5].item())
                detecciones.append({
                    "class": model.names[cls_id],
                    "confidence": float(r[4].item())
                })
        return detecciones

    except Exception as e:
        print("Error en detección YOLO:", e)
        return []
