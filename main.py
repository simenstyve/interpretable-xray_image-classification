# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid
import os

from inference import detect_and_classify_compliance

app = FastAPI(title="CAMCIS X-Ray Compliance API")

UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):

    # Save file temporarily
    ext = file.filename.split(".")[-1]
    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{ext}")

    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = detect_and_classify_compliance(tmp_path)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    finally:
        os.remove(tmp_path)

    return result
