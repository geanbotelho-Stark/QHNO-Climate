import pandas as pd
import numpy as np


class ENSOForecastEngine:
    """
    Forecast simples baseado em série temporal ONI.
    Não é ML pesado ainda — é baseline científico.
    """

    def __init__(self, data_path="data/processed_oni.csv"):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.sort_values(by=["year"]).reset_index(drop=True)

    def compute_moving_average(self, window=12):
        self.df["ma"] = self.df["oni_index"].rolling(window=window).mean()

    def compute_trend(self):
        # derivada simples (diferença entre médias recentes)
        self.df["trend"] = self.df["oni_index"].diff()

    def predict_next(self, steps=6):
        """
        projeção linear simples baseada na tendência média recente
        """
        last_values = self.df["oni_index"].values[-12:]
        x = np.arange(len(last_values))
        coef = np.polyfit(x, last_values, 1)  # regressão linear simples

        slope = coef[0]
        intercept = coef[1]

        future_x = np.arange(len(last_values), len(last_values) + steps)
        forecast = slope * future_x + intercept

        return forecast

    def classify_state(self, value):
        if value >= 0.5:
            return "El Niño"
        elif value <= -0.5:
            return "La Niña"
        else:
            return "Neutro"

    def run(self):
        print("[FORECAST] Executando previsão ENSO...")

        self.load_data()
        self.compute_moving_average()
        self.compute_trend()

        forecast = self.predict_next()

        states = [self.classify_state(v) for v in forecast]

        result = {
            "forecast_values": forecast.tolist(),
            "forecast_states": states,
            "current_state": self.classify_state(self.df["oni_index"].iloc[-1])
        }

        print("[FORECAST] Resultado:")
        print(result)

        return result
