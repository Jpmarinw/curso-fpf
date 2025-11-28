import pandas as pd
from datetime import datetime

class ForecastingService:
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config

    def predict(self, horizon: int):
        forecast = self.model.predict(n_periods=horizon)
        
        start_date = "1960-12-01"
        dates = pd.date_range(start=start_date, periods=horizon+1, freq='MS')[1:]
        
        results = []
        for date, value in zip(dates, forecast):
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": int(value)
            })
            
        return {
            "model": self.config["model"]["name"],
            "version": self.config["app"]["version"],
            "generated_at": datetime.now(),
            "prediction": results
        }