import os
import pandas as pd
import numpy as np
import json
from typing import Union

from ..config.config import PDPConfig, EPDPConfig
from ..models.data_model import PDPInstance, EPDPInstance


def load_instance(config: Union[PDPConfig, EPDPConfig],
                  problem_path: str,
                  unique_locations_path: str,
                  distance_matrix_path: str,
                  time_matrix_path: str,
                  charging_stations_path: str = None) -> Union[PDPInstance, EPDPInstance]:
    """
    Load and prepare an instance for the Pickup and Delivery Problem (PDP) or Energy Pickup and Delivery Problem (EPDP).

    Parameters:
    config (Union[PDPConfig, EPDPConfig]): Configuration for the problem instance.
    problem_path (str): Path to the CSV file containing bookings data.
    unique_locations_path (str): Path to the CSV file containing unique depot/customers/stans data.
    distance_matrix_path (str): Path to the JSON file containing the distance matrix.
    time_matrix_path (str): Path to the JSON file containing the time matrix.
    charging_stations_path (str, optional): Path to the CSV file containing charging stations data. Default is None.

    Returns:
    Union[PDPInstance, EPDPInstance]: An instance of PDP or EPDP populated with the loaded data in the format required
    by ortools.

    Raises:
    FileNotFoundError: If the distance matrix or time matrix files are not found.
    ValueError: If the configuration instance type is unsupported.
    """

    # Load the bookings data
    print(f"Loading instance from: {os.path.abspath(problem_path)}")
    bookings_mapping = pd.read_csv(problem_path)
    unique_locations = pd.read_csv(unique_locations_path)

    # Load the distance matrix
    try:
        with open(distance_matrix_path, 'r') as f:
            unique_distance_matrix = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Distance matrix file not found: {distance_matrix_path}")

    # Load the time matrix
    try:
        with open(time_matrix_path, 'r') as f:
            unique_duration_matrix = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Time matrix file not found: {time_matrix_path}")

    # Load the charging stations data if provided
    if charging_stations_path:
        stations_mapping = pd.read_csv(charging_stations_path)
        # optionally sample from them to reduce total number of station
        # stations_mapping = stations_mapping.sample(n=30, random_state=11).reset_index()
        # stations_mapping = filter_relevant_stations(bookings_mapping, stations_mapping, unique_distance_matrix, max_stations=30)
    else:
        stations_mapping = None

    # Filter stations to use based on hardcoded list in the config
    stations_multiply_factor = 2
    indices_to_multiply = [213, 302, 324, 242, 328, 329]
    #
    if config.instance_type == 'EPDP' and stations_mapping is not None:
        stations_mapping = stations_mapping[stations_mapping['Index'].isin(config.stations_to_use)].reset_index(drop=True)

    # Multiply used stations
    # stations_mapping = pd.concat([stations_mapping] * stations_multiply_factor, ignore_index=True)

    # Filter the rows to be duplicated
    rows_to_duplicate = stations_mapping[stations_mapping['Index'].isin(indices_to_multiply)]

    # Duplicate the rows by concatenating them multiple times
    duplicated_rows = pd.concat([rows_to_duplicate] * stations_multiply_factor, ignore_index=True)

    # Combine the duplicated rows with the original DataFrame
    # Keeping only unique indices in the original dataframe to avoid duplication
    original_rows = stations_mapping[~stations_mapping['Index'].isin(indices_to_multiply)]
    stations_mapping = pd.concat([original_rows, duplicated_rows], ignore_index=True).reset_index(drop=True)

    # Create node2index mapping
    node_to_index_map = create_node2index_mapping(bookings_mapping, stations_mapping, unique_locations,
                                                  epdp=(config.instance_type == 'EPDP'))

    num_nodes = len(node_to_index_map)
    num_bookings = len(bookings_mapping)
    num_customers = 2 * len(bookings_mapping)

    data = {}

    data['depot'] = 0

    data['distance_matrix'], data['time_matrix'] = create_distance_time_matrices(num_nodes,
                                                                                 unique_distance_matrix,
                                                                                 unique_duration_matrix,
                                                                                 node_to_index_map)

    # Convert time matrix to 30-minute units
    for i in range(len(data['time_matrix'])):
        for j in range(len(data['time_matrix'][i])):
            data['time_matrix'][i][j] = int(round(data['time_matrix'][i][j] * 2))  # Converting hours to 30-minute units

    data['demands'] = [0] + [int(weight) for weight in bookings_mapping['GrossWeight']] + [-int(weight) for weight in
                                                                                           bookings_mapping[
                                                                                               'GrossWeight']]

    pickup_times = [(row['PickupStartTime'], row['PickupEndTime']) for _, row in bookings_mapping.iterrows()]
    delivery_times = [(row['DeliveryStartTime'], row['DeliveryEndTime']) for _, row in bookings_mapping.iterrows()]
    data['time_windows'] = [(0, 47)] * (num_customers + 1)  # + pickup_times + delivery_times
    data['pickups_deliveries'] = [[i + 1, i + 1 + num_bookings] for i in range(num_bookings)]
    data['service_time'] = [0] + [0] * num_customers

    data['num_vehicles'] = config.num_vehicles
    data['vehicle_capacities'] = [int(config.vehicle_capacity)] * config.num_vehicles
    data['energy_capacity'] = config.energy_capacity
    data['consumption_rate'] = config.battery_consumption_rate

    # Additional parameters for EPDP
    if config.instance_type == 'EPDP':
        if stations_mapping is None:
            raise ValueError("Charging stations data is required for EPDP but not provided.")

        data['station_index'] = [node for node, details in node_to_index_map.items() if details['source'] == 'station']

        num_stations = len(data['station_index'])
        data['charging_capacity'] = config.charging_capacity

        # Calculate service time for stations
        for i, station in stations_mapping.iterrows():
            power_kw = station['PowerKW']
            service_time = round(300 / power_kw * 2)
            data['service_time'].append(int(service_time))

        data['demands'].extend([0] * num_stations)
        data['time_windows'].extend([(0, 47)] * num_stations)

    # Convert the DataFrame to a dictionary with integer keys
    data['node_to_index_map'] = node_to_index_map

    assert len(data['time_matrix']) == len(data['distance_matrix'])
    assert len(data['time_matrix']) == num_nodes
    assert num_nodes == len(data['time_windows'])
    assert num_nodes == len(data['demands'])

    if config.instance_type == 'PDP':
        return PDPInstance(**data)
    elif config.instance_type == 'EPDP':
        return EPDPInstance(**data)
    else:
        raise ValueError("Unsupported instance type")


def filter_relevant_stations(bookings_mapping: pd.DataFrame, stations_mapping: pd.DataFrame, distance_matrix: list,
                             max_stations: int = 30) -> pd.DataFrame:
    # Get unique customer indices from bookings
    customer_indices = set(bookings_mapping['FromIndex'].tolist() + bookings_mapping['ToIndex'].tolist())
    station_indices = stations_mapping['Index'].tolist()

    # Calculate the minimum distance from each station to any customer using the distance matrix
    station_distances = []
    for station_idx in station_indices:
        min_distance = min(distance_matrix[station_idx][customer_idx] for customer_idx in customer_indices)
        station_distances.append((station_idx, min_distance))

    # Sort stations by distance and select the closest ones
    station_distances.sort(key=lambda x: x[1])
    selected_stations = [idx for idx, _ in station_distances[:max_stations]]

    # Filter the stations_mapping DataFrame to keep only the selected stations
    return stations_mapping[stations_mapping['Index'].isin(selected_stations)].reset_index(drop=True)


def create_node2index_mapping(bookings_mapping, stations_mapping, unique_locations, epdp=True):
    num_bookings = len(bookings_mapping)
    num_stations = len(stations_mapping) if stations_mapping is not None else 0

    unique_locations = list(zip(unique_locations['Latitude'], unique_locations['Longitude']))
    node_to_index_map = {0: {'unique_index': 0, 'source': 'depot', 'location': unique_locations[0]}}

    for i in range(num_bookings):
        from_index = bookings_mapping.at[i, 'FromIndex']
        location = unique_locations[from_index]
        node_to_index_map[1 + i] = {'unique_index': from_index, 'source': 'pickup', 'location': location}

    for i in range(num_bookings):
        to_index = bookings_mapping.at[i, 'ToIndex']
        location = unique_locations[to_index]
        node_to_index_map[1 + num_bookings + i] = {'unique_index': to_index, 'source': 'delivery', 'location': location}

    if epdp:
        for i in range(num_stations):
            index = stations_mapping.at[i, 'Index']
            location = unique_locations[index]
            node_to_index_map[1 + 2 * num_bookings + i] = {'unique_index': index, 'source': 'station',
                                                           'location': location}

    return node_to_index_map


def create_distance_time_matrices(num_nodes, unique_distance_matrix, unique_duration_matrix, node_to_index_map):
    distance_matrix = np.zeros((num_nodes, num_nodes))
    time_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            unique_index_i = node_to_index_map[i]['unique_index']
            unique_index_j = node_to_index_map[j]['unique_index']

            distance = unique_distance_matrix[unique_index_i][unique_index_j] / 1000  # Convert to kilometers
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

            time = unique_duration_matrix[unique_index_i][unique_index_j] / 3600  # Convert to hours
            time_matrix[i, j] = time
            time_matrix[j, i] = time

    # Convert distance and time matrices to integers
    distance_matrix = distance_matrix.astype(int)
    # time_matrix = time_matrix.astype(int)

    return distance_matrix.tolist(), time_matrix.tolist()
