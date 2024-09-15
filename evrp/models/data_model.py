from pydantic import BaseModel
from typing import List, Tuple, Dict


class NodeInfo(BaseModel):
    unique_index: int
    source: str
    location: tuple


class PDPInstance(BaseModel):
    depot: int
    distance_matrix: List[List[int]]
    time_matrix: List[List[int]]
    demands: List[int]
    time_windows: List[Tuple[int, int]]
    pickups_deliveries: List[List[int]]
    service_time: List[int]
    num_vehicles: int
    vehicle_capacities: List[int]
    # optional?
    energy_capacity: int
    consumption_rate: float
    node_to_index_map: Dict[int, NodeInfo]

    def __getitem__(self, item):
        return getattr(self, item)


class EPDPInstance(BaseModel):
    depot: int
    station_index: List[int]
    distance_matrix: List[List[int]]
    time_matrix: List[List[int]]
    demands: List[int]
    time_windows: List[Tuple[int, int]]
    pickups_deliveries: List[List[int]]
    service_time: List[int]
    num_vehicles: int
    vehicle_capacities: List[int]
    energy_capacity: int
    charging_capacity: int
    consumption_rate: float
    node_to_index_map: Dict[int, NodeInfo]

    def __getitem__(self, item):
        return getattr(self, item)
