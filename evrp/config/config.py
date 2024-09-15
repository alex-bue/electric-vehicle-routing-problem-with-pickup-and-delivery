import os
import yaml
from typing import Optional
from pydantic_settings import BaseSettings


class SolverConfig(BaseSettings):
    first_solution_strategy: str = "AUTOMATIC"
    local_search_metaheuristic: str = "TABU_SEARCH"
    time_limit: int = 200
    lns_time_limit: Optional[int] = None


class PDPConfig(BaseSettings):
    instance_type: str = 'PDP'
    num_vehicles: int = 50
    vehicle_capacity: int = 33000
    energy_capacity: int = 390
    battery_consumption_rate: float = 1.3
    solver_settings: SolverConfig = SolverConfig()


class EPDPConfig(BaseSettings):
    instance_type: str = 'EPDP'
    num_vehicles: int = 50
    battery_consumption_rate: float = 1.3
    vehicle_capacity: int = 33000
    energy_capacity: int = 390
    charging_capacity: int = 300
    stations_to_use: list[int] = [264, 265, 268, 287, 301, 302, 313, 317, 324, 328, 329, 334, 213, 214, 225, 226, 240, 242, 244, 245, 251, 253, 254]
    #     251,
    #     245,
    #     302,
    #     214,
    #     274,
    #     317,
    #     265,
    #     226,
    #     262,
    #     329,
    #     278,
    #     328,
    #     253,
    #     268,
    #     298,
    #     324,
    #     242,
    #     287,
    #     334,
    #     225,
    #     257,
    #     301,
    #     254,
    #     240,
    #     213,
    #     337,
    #     327,
    #     244,
    #     264,
    #     313
    # ]

    solver_settings: SolverConfig = SolverConfig()


class Config(BaseSettings):
    pdp_config: PDPConfig = PDPConfig()
    epdp_config: EPDPConfig = EPDPConfig()


class ConfigLoader:
    def __init__(self, config_file=None, env_prefix=None):
        self.config = {}
        self.env_prefix = env_prefix

        if config_file:
            self.load_from_file(config_file)

        if env_prefix:
            self.load_from_env(env_prefix)

    def load_from_file(self, config_file):
        with open(config_file, 'r') as file:
            self.config.update(yaml.safe_load(file))

    def load_from_env(self, env_prefix):
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                self.config[key[len(env_prefix):]] = value

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def update(self, custom_config):
        self.config.update(custom_config)
