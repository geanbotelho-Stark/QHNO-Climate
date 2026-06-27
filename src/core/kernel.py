import logging
from dataclasses import dataclass

from src.collector.noaa_collector import NOAACollector
from src.climate_memory.memory_engine import ClimateMemoryEngine
from src.prediction.enso_forecast import ENSOForecastEngine


@dataclass
class QHNOConfig:
    project_name: str = "QHNO-Climate"
    version: str = "0.1.0"
    debug: bool = True


class QHNOKernel:

    def __init__(self, config: QHNOConfig):
        self.config = config
        self.logger = self._setup_logger()

        self.logger.info(
            f"{self.config.project_name} v{self.config.version} initializing..."
        )

        # REGISTRO CENTRAL (single source of truth)
        self.modules = {}

        self.collector = NOAACollector()
        self.memory = ClimateMemoryEngine()
        self.forecast = ENSOForecastEngine()

        self._register_core_modules()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.DEBUG if self.config.debug else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger("QHNO")

    def _register_core_modules(self):
        self.register_module("collector", self.collector)
        self.register_module("memory", self.memory)
        self.register_module("forecast", self.forecast)

    def register_module(self, name: str, module: object):
        self.modules[name] = module
        self.logger.info(f"Module registered: {name}")

    def start(self):
        self.logger.info("Starting QHNO system...")

        # EXECUTION PIPELINE FIXO (ORQUESTRADO PELO KERNEL)
        pipeline = [
            ("collector", "Running NOAA collector"),
            ("memory", "Building Climate Memory"),
            ("forecast", "Running ENSO Forecast"),
        ]

        for module_name, message in pipeline:
            self.logger.info(f"{message}...")

            module = self.modules.get(module_name)
            if module is None:
                self.logger.error(f"Module missing: {module_name}")
                continue

            module.run()

        self.logger.info("QHNO system is ready.")
