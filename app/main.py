import os

from fastapi import FastAPI, File, UploadFile

from config import MODEL_FILE, INPUT_SIZE
from predictor import Predictor


app = FastAPI(predictor=Predictor(MODEL_FILE, INPUT_SIZE))


@app.post('/api/v1/images/predict')
async def predict(file: UploadFile = File(...)):
    label, _ = app.extra['predictor'].predict(file.filename)
    return label
