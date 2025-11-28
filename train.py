import os
import pickle
import pandas as pd
from pmdarima import auto_arima

CAMINHO_CSV = 'data/01-raw/airline-passengers.csv'
PASTA_MODELOS = 'models'
CAMINHO_MODELO = os.path.join(PASTA_MODELOS, 'model_arima.pkl')

if not os.path.exists(CAMINHO_CSV):
    raise FileNotFoundError(f"Arquivo não encontrado: {CAMINHO_CSV}.")

df = pd.read_csv(CAMINHO_CSV, parse_dates=['Month'], index_col='Month')

model = auto_arima(df['Passengers'], seasonal=True, m=12, suppress_warnings=True)

if not os.path.exists(PASTA_MODELOS):
    os.makedirs(PASTA_MODELOS)

with open(CAMINHO_MODELO, 'wb') as f:
    pickle.dump(model, f)