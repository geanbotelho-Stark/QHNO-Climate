import requests
import os
from datetime import datetime


class NOAACollector:
    """
    Coletor oficial de ONI (El Niño / La Niña)
    """

    def __init__(self, save_path="data/raw"):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

    def fetch(self):
        r = requests.get(self.url, timeout=30)
        if r.status_code != 200:
            raise Exception("Falha ao acessar NOAA ONI")
        return r.text

    def save(self, data: str):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_path, f"oni_{ts}.txt")

        with open(path, "w") as f:
            f.write(data)

        return path

    def run(self):
        print("[NOAA] Coletando ONI...")

        data = self.fetch()
        path = self.save(data)

        print(f"[NOAA] Salvo em: {path}")

        return path
