import os
import json
import yaml
import time
from tqdm import tqdm
import argparse
import pandas as pd
from dotenv import load_dotenv
from evrp.config.config import ConfigLoader
from util.routers import MapboxValhalla

MAX_COORDINATES_PER_REQUEST = 25
SECONDS_PER_REQUEST = 1  # Conservative interval

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def save_matrix(matrix, file_path):
    with open(file_path, 'w') as file:
        json.dump(matrix, file, indent=4)


def create_distance_duration_matrices(coordinates, router):
    size = len(coordinates)
    distance_matrix = [[0] * size for _ in range(size)]
    duration_matrix = [[0] * size for _ in range(size)]
    total_elements = 0

    # Calculate total expected requests
    total_requests = sum((i // (MAX_COORDINATES_PER_REQUEST - 1)) + 1 for i in range(size))

    with tqdm(total=total_requests, desc="Processing", unit="requests") as pbar:
        for i in range(size):
            if i == 0:
                # Special case for the first row
                sources = [0, 1]
                destinations = [0, 1]
                print(f'Query 1 Sources: {sources}, Destinations: {destinations}')
                query_coordinates = [coordinates[idx] for idx in set(sources + destinations)]
                matrix = router.matrix(profile='mapbox/driving', coordinates=query_coordinates, sources=[0], destinations=[0, 1])
                print(f"Matrix Distances: {matrix.distances}")
                print(f"Matrix Durations: {matrix.durations}")
                if len(matrix.distances) > 1 and len(matrix.distances[1]) > 0:
                    distance_matrix[0][1] = matrix.distances[0][1]
                    duration_matrix[0][1] = matrix.durations[0][1]
                    distance_matrix[1][0] = matrix.distances[1][0]
                    duration_matrix[1][0] = matrix.durations[1][0]
                    num_elements = len(sources) * len(destinations)
                    total_elements += num_elements
                    print(f'Query 1: {num_elements} elements')
                    pbar.set_postfix(elements=total_elements)
                    pbar.update(1)
                    time.sleep(SECONDS_PER_REQUEST)
                else:
                    print("Unexpected matrix dimensions for the first query.")
            else:
                sources = [i]
                start_index = 0
                while start_index < i + 1:
                    end_index = min(start_index + MAX_COORDINATES_PER_REQUEST - 1, i + 1)
                    destinations = list(range(start_index, end_index))
                    if len(destinations) == 1:
                        destinations.append((destinations[0] + 1) % size)
                    query_coordinates = [coordinates[idx] for idx in set(sources + destinations)]
                    source_indices = [query_coordinates.index(coordinates[src]) for src in sources]
                    dest_indices = [query_coordinates.index(coordinates[dest]) for dest in destinations]
                    print(f'Query {i + 1} Sources: {sources}, Destinations: {destinations}')
                    print(f'Coordinates being sent: {query_coordinates}')
                    matrix = router.matrix(profile='mapbox/driving', coordinates=query_coordinates, sources=source_indices, destinations=dest_indices)
                    print(f"Matrix Distances: {matrix.distances}")
                    print(f"Matrix Durations: {matrix.durations}")
                    distances = matrix.distances
                    durations = matrix.durations
                    if len(distances) > 0 and len(distances[0]) > 0:
                        for destination_index, destination in enumerate(destinations):
                            distance_matrix[i][destination] = distances[0][destination_index]
                            duration_matrix[i][destination] = durations[0][destination_index]
                        num_elements = len(sources) * len(destinations)
                        total_elements += num_elements
                        print(f'Query {i + 1}: {num_elements} elements')
                        pbar.set_postfix(elements=total_elements)
                        pbar.update(1)
                        time.sleep(SECONDS_PER_REQUEST)
                    else:
                        print(f"Unexpected matrix dimensions for query {i + 1}.")
                    start_index = end_index

    print(f'Total elements returned from all queries: {total_elements}')

    # Symmetrize the matrices
    for i in range(size):
        for j in range(i):
            distance_matrix[j][i] = distance_matrix[i][j]
            duration_matrix[j][i] = duration_matrix[i][j]

    return distance_matrix, duration_matrix


def main():
    parser = argparse.ArgumentParser(description='Generate distance and duration matrices using Mapbox API.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--coordinates', type=str, required=True, help='Path to the CSV file with unique locations.')
    parser.add_argument('--dry-run', action='store_true', help='Run a dry test with a small dummy list of coordinates.')
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("MAPBOX_API_KEY")
    config_loader = ConfigLoader(config_file=args.config)
    config = config_loader.config

    distance_matrix_path = config['paths']['distance_matrix']
    duration_matrix_path = config['paths']['duration_matrix']

    # Safeguard: Check if distance and duration matrix files already exist
    if not args.dry_run:
        # Safeguard: Check if distance and duration matrix files already exist
        if os.path.exists(distance_matrix_path) or os.path.exists(duration_matrix_path):
            print(f"Distance matrix file '{distance_matrix_path}' or duration matrix file '{duration_matrix_path}' already exists. Exiting to avoid overwriting.")
            return

    if args.dry_run:
        # Use dummy coordinates for dry run
        coordinates = [
            [13.388860, 52.517037],
            [13.397634, 52.529407],
        ]
    else:
        # Read our depot/customers/stations unique coords
        df = pd.read_csv(args.coordinates)
        coordinates = df[['Longitude', 'Latitude']].values.tolist()

    router = MapboxValhalla(api_key)
    distance_matrix, duration_matrix = create_distance_duration_matrices(coordinates, router)

    if args.dry_run:
        print("Distance Matrix (Dry Run):")
        print(json.dumps(distance_matrix, indent=4))
        print("Duration Matrix (Dry Run):")
        print(json.dumps(duration_matrix, indent=4))
    else:
        save_matrix(distance_matrix, distance_matrix_path)
        save_matrix(duration_matrix, duration_matrix_path)
        print(f"Distance matrix saved to {distance_matrix_path}")
        print(f"Duration matrix saved to {duration_matrix_path}")

    # Print the size of the matrices
    matrix_size = len(distance_matrix)
    print(f"Matrix size: {matrix_size} x {matrix_size}")


if __name__ == "__main__":
    main()
