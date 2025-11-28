from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from src.components import load_config, load_model
from src.service import ForecastingService

class ForecastInput(BaseModel):
    horizon: int = Field(..., gt=0, le=60, description="Horizonte de previsão em meses")

def get_service():
    config = load_config()
    model = load_model(config["model"]["path"])
    return ForecastingService(model, config)

app = FastAPI()

@app.post("/predict")
def predict_endpoint(
    input_data: ForecastInput,
    service: ForecastingService = Depends(get_service)
):
    return service.predict(input_data.horizon)