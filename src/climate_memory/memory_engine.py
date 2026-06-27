import os
import pandas as pd
import re


class ClimateMemoryEngine:
    """
    Normalizador robusto de dados NOAA ONI.
    Adaptado para o formato padrão do arquivo oni.ascii.txt (SEAS YR TOTAL ANOM).
    """

    def __init__(self, data_path="data/raw"):
        self.data_path = data_path

    def load_files(self):
        # Evita falhas caso o diretório de dados ainda não exista
        if not os.path.exists(self.data_path):
            print(f"[Aviso] Diretório {self.data_path} não encontrado.")
            return []
            
        return sorted([
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.startswith("oni_")
        ])

    def parse_file(self, file_path):
        data = []

        # Mapeamento dos trimestres móveis da NOAA para meses de referência
        # Garante a compatibilidade com a estrutura de meses do seu Pattern Engine
        season_to_month = {
            "DJF": "Jan", "JFM": "Feb", "FMA": "Mar", "MAM": "Apr",
            "AMJ": "May", "MJJ": "Jun", "JJA": "Jul", "JAS": "Aug",
            "ASO": "Sep", "SON": "Oct", "OND": "Nov", "NDJ": "Dec"
        }

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Ignora cabeçalhos ou linhas vazias
                if not line or line.startswith("SEAS") or line.startswith("STN"):
                    continue

                # O formato do arquivo real da NOAA é: SEAS YR TOTAL ANOM
                # Exemplo: DJF    1950   24.72  -1.53
                parts = line.split()
                
                # Garante que a linha possui as colunas necessárias
                if len(parts) < 4:
                    continue

                season, year_str, _, anom_str = parts[:4]

                # Valida se o campo do ano é realmente um número de 4 dígitos
                if not re.match(r"^\d{4}$", year_str):
                    continue

                try:
                    year = int(year_str)
                    oni_value = float(anom_str)
                    
                    # Converte a sigla da estação para o mês correspondente
                    month = season_to_month.get(season, season)

                    data.append((year, month, oni_value))

                except ValueError:
                    continue

        return data

    def build_dataset(self):
        all_data = []
        files = self.load_files()
        
        if not files:
            print("[Erro] Nenhum arquivo 'oni_' encontrado para processar.")
            return pd.DataFrame(columns=["year", "month", "oni_index"])

        for file in files:
            all_data.extend(self.parse_file(file))

        # Cria o DataFrame com a estrutura esperada pelo sistema QHNO
        df = pd.DataFrame(all_data, columns=["year", "month", "oni_index"])
        return df

    def run(self):
        print("[MEMORY] Normalizando dados climáticos...")

        df = self.build_dataset()

        if df.empty:
            print("[MEMORY] Nenhum dado foi processado. Abortando salvamento.")
            return None

        # Garante que a pasta de saída de dados processados exista
        os.makedirs("data", exist_ok=True)
        
        output_path = "data/processed_oni.csv"
        df.to_csv(output_path, index=False)

        print(f"[MEMORY] Dataset salvo em: {output_path}")
        print(f"[MEMORY] Linhas processadas: {len(df)}")

        return output_path


# Código para execução direta/testes isolados do módulo
if __name__ == "__main__":
    engine = ClimateMemoryEngine()
    engine.run()
