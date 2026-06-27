import pandas as pd


class ClimatePatternEngine:
    """
    Engine de padrões climáticos ENSO.
    Analisa ONI e detecta regimes climáticos.
    """

    def __init__(self, dataset_path="data/processed_oni.csv"):
        self.dataset_path = dataset_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.dataset_path)
        return self.df

    def classify_enso(self, value):
        """
        Classificação simples ENSO baseada no ONI
        """
        if value >= 0.5:
            return "El Niño"
        elif value <= -0.5:
            return "La Niña"
        else:
            return "Neutro"

    def add_classification(self):
        self.df["enso_state"] = self.df["oni_index"].apply(self.classify_enso)

    def compute_trend(self):
        """
        Calcula tendência simples (variação média)
        """
        self.df["rolling_mean"] = self.df["oni_index"].rolling(window=6).mean()
        return self.df

    def summarize(self):
        summary = self.df["enso_state"].value_counts().to_dict()
        return summary

    def run(self):
        print("[PATTERN] Analisando padrões climáticos...")

        self.load_data()
        self.add_classification()
        self.compute_trend()

        summary = self.summarize()

        output_path = "data/enso_patterns.csv"
        self.df.to_csv(output_path, index=False)

        print(f"[PATTERN] Padrões salvos em: {output_path}")
        print(f"[PATTERN] Resumo: {summary}")

        return output_path
