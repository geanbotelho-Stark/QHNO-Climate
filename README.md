# QH-ENSO: Gêmeo Digital Híbrido para Previsão Climática 🌍⚛️

O **QH-ENSO** (Quantum-Hybrid Numerical Optimizer for ENSO) é um sistema pioneiro de inteligência artificial híbrida (Clássica-Quântica) focado na previsão e análise de eventos climáticos globais complexos, especificamente o El Niño e a La Niña (ENOS). 

O coração do sistema integra circuitos quânticos variacionais com redes neurais profundas clássicas para capturar correlações não-locais e teleconexões atmosféricas a partir de dados globais de telemetria oceânica.

## 🚀 Arquitetura do Sistema

- **Data Lake:** Módulo de ingestão preparado para receber telemetrias da NOAA (`api.ncep.noaa.gov`) e do Copernicus (`cds.climate.copernicus.eu`).
- **Quantum Feature Encoding:** Codificação quântica avançada usando portas customizadas $RY$ e $RZ$ para mapear variáveis clássicas diretamente no Espaço de Hilbert.
- **Camada Variacional Quântica (VQC):** Processamento com 2 camadas de emaranhamento forte (*Strongly Entangling Layers*) usando PennyLane.
- **Classical Forecast Head:** Rede neural feedforward profunda em PyTorch que realiza a calibração final e regressão do Índice ONI (Oceanic Niño Index).

## 📊 Resultados de Calibração Atual (Último Run)

O motor quântico-clássico demonstrou excelente convergência durante a fase de otimização dos parâmetros variacionais com 120 registros históricos:

- **Época 1/5** | MSE Global: `2.1853`
- **Época 2/5** | MSE Global: `2.0817`
- **Época 3/5** | MSE Global: `1.7959`
- **Época 4/5** | MSE Global: `1.2104`
- **Época 5/5** | MSE Global: `0.7240` (Convergência Estabilizada)

## 🛠️ Como Executar

1. Certifique-se de estar com o ambiente Conda `qhno` ativo.
2. Execute o motor principal:
   ```bash
   python qh_enso_core.py
