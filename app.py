# app.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from utils import preprocess_for_mnist, segment_digits_from_image

app = FastAPI()
model = load_model('best_mnist.h5')

def read_imagefile(file) -> np.ndarray:
    img = Image.open(BytesIO(file)).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.post("/predict/single")
async def predict_single_api(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_imagefile(contents)
    proc = preprocess_for_mnist(img)
    x = proc.reshape(1,28,28,1).astype('float32')
    preds = model.predict(x)
    label = int(np.argmax(preds))
    conf  = float(np.max(preds))
    return {"label": label, "confidence": conf}

@app.post("/predict/multi")
async def predict_multi_api(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_imagefile(contents)
    crops, boxes = segment_digits_from_image(img)
    out = []
    for c in crops:
        proc = cv2.resize(c, (28,28)).astype('float32')/255.0
        x = proc.reshape(1,28,28,1)
        preds = model.predict(x)
        out.append({"label": int(np.argmax(preds)), "confidence": float(np.max(preds))})
    return {"predictions": out}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
