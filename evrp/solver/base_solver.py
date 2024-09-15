from typing import Union
import json
import folium
from folium import Popup, Tooltip, plugins
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from abc import ABC, abstractmethod
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from branca.element import Template, MacroElement

from ..models.data_model import PDPInstance, EPDPInstance
from ..config.config import SolverConfig
from ..util.call_back import SolutionCallback


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def get_curve_points(p1, p2):
    """Generate points for a curve between two points."""
    lat1, lon1 = p1
    lat2, lon2 = p2

    curvature = 0.2 if (lon1 > lon2) else -0.2

    midpoint = [(lat1 + lat2) / 2, (lon1 + lon2) / 2]
    midpoint[1] += curvature  # adjust longitude for curvature

    curve_points = []
    for t in np.linspace(0, 1, 20):  # increase the number for more smoothness
        x = (1 - t) * (1 - t) * lat1 + 2 * (1 - t) * t * midpoint[0] + t * t * lat2
        y = (1 - t) * (1 - t) * lon1 + 2 * (1 - t) * t * midpoint[1] + t * t * lon2
        curve_points.append((x, y))
    return curve_points


class Solver(ABC):

    def __init__(self, data: Union[PDPInstance, EPDPInstance]):
        # Model parameter
        self.data = data
        self.solution = None
        self.manager = None
        self.routing = None
        self.node_to_index_map = data.node_to_index_map
        self.solution_log = None  # populated during solve_model() call and tracks each solution objective
        self.solution_data = None  # store routes etc. from solution

    @abstractmethod
    def create_model(self):
        pass

    def solve_model(self,
                    settings: SolverConfig,
                    log_search=True,
                    initial_solution_filepath=None):

        first_solution_strategy_mapping = {
            'AUTOMATIC': routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
            'PATH_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            'PATH_MOST_CONSTRAINED_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
            # 'EVALUATOR_STRATEGY': routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY,
            'SAVINGS': routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            # 'SWEEP': routing_enums_pb2.FirstSolutionStrategy.SWEEP,
            # 'CHRISTOFIDES': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
            'ALL_UNPERFORMED': routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
            'BEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
            'PARALLEL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            'LOCAL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
            'GLOBAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
            'LOCAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
            'FIRST_UNBOUND_MIN_VALUE': routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE
        }

        local_search_metaheuristic_mapping = {
            'AUTOMATIC': routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
            'GREEDY_DESCENT': routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
            'GUIDED_LOCAL_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
            'SIMULATED_ANNEALING': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            'TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
            'GENERIC_TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH
        }

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        # Setting first solution heuristic.
        search_parameters.first_solution_strategy = first_solution_strategy_mapping.get(
            settings.first_solution_strategy, routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

        # Set local search metaheuristic
        search_parameters.local_search_metaheuristic = local_search_metaheuristic_mapping.get(
            settings.local_search_metaheuristic, routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
        search_parameters.time_limit.seconds = settings.time_limit
        search_parameters.log_search = log_search

        solution_callback = SolutionCallback(self.routing)
        self.routing.AddAtSolutionCallback(solution_callback)

        if initial_solution_filepath is not None:
            print("Initial Solution generated")
            initial_solution = self.create_initial_solution(initial_solution_filepath)
            self.routing.CloseModelWithParameters(search_parameters)
            initial_solution = self.routing.ReadAssignmentFromRoutes(initial_solution, True)
            print(initial_solution)
            self.solution = self.routing.SolveFromAssignmentWithParameters(
                initial_solution, search_parameters
            )
        else:
            # Solve the problem.
            print("Solving without initial")
            self.solution = self.routing.SolveWithParameters(search_parameters)
            if self.solution:
                print('Solution found')
            else:
                print('Solution Not Found')
        # Store the solution details in the Solver instance
        self.solution_log = solution_callback.solution_log

    def create_initial_solution(self, file_path):
        with open(file_path, 'r') as f:
            initial_solution = json.load(f)

        routes = initial_solution['routes']
        dropped_nodes = set(initial_solution['dropped_nodes'])
        pickups_deliveries = self.data['pickups_deliveries']
        station_indices = set(self.data['station_index'])
        initial_routes = {}

        # Drop routes that do not have any nodes
        initial_routes = {route_key: node_list for route_key, node_list in routes.items() if node_list != []}

        for node in list(dropped_nodes):  # Use list to allow modification during iteration
            pickup_delivery_pair = next((pair for pair in pickups_deliveries if pair[0] == node or pair[1] == node),
                                        None)
            if pickup_delivery_pair:
                pickup, delivery = pickup_delivery_pair

                if pickup in dropped_nodes and delivery in dropped_nodes:
                    def find_closest_station(node1, node2):
                        mid_point = (self.data['distance_matrix'][node1][node2] / 2)
                        available_stations = [station for station in station_indices if
                                              station in dropped_nodes]
                        closest_station = min(available_stations, key=lambda station: abs(
                            mid_point - self.data['distance_matrix'][node1][station] -
                            self.data['distance_matrix'][node2][station]))
                        return closest_station

                    station1 = find_closest_station(0, pickup, )
                    dropped_nodes.remove(station1)
                    station2 = find_closest_station(pickup, delivery)
                    dropped_nodes.remove(station2)
                    station3 = find_closest_station(pickup, delivery)
                    dropped_nodes.remove(station3)
                    station4 = find_closest_station(delivery, 0)
                    dropped_nodes.remove(station4)

                    # Select stations for the new route
                    route = [station1, pickup, station2, station3, delivery, station4]
                    initial_routes[str(max(int(k) for k in initial_routes.keys()) + 1)] = route
                    print(f"New route: {route}")

                    # Remove pickup and delivery from dropped nodes
                    dropped_nodes.remove(pickup)
                    dropped_nodes.remove(delivery)

        initial_routes = list(initial_routes.values())
        return initial_routes

    def get_solution_data(self, with_energy_constraints=False):
        # Extracting solutionn data from
        if self.solution is None:
            print('Solution not existing')

        self.solution_data = {
            'routes': {},
            'routes_node2index': {},
            'final_objective': self.solution.ObjectiveValue(),
            'final_var': {},
            'cumul_vars': {},
            'dropped_nodes': [],
            'dropped_index': [],
            'solution_log': self.solution_log
        }

        if isinstance(self.data, EPDPInstance):
            self.solution_data['station_nodes'] = self.data['station_index'] # node
            station_index = [int(self.data['node_to_index_map'][node].unique_index) for node in self.data['station_index']]
            self.solution_data['station_index'] = station_index

        dimensions = ['Time', 'Distance', 'Capacity']
        if with_energy_constraints:
            dimensions.append('Energy')

        all_nodes = set(range(self.manager.GetNumberOfNodes()))
        visited_nodes = set()

        for vehicle_id in range(self.manager.GetNumberOfVehicles()):
            index = self.routing.Start(vehicle_id)
            visited_nodes.add(index)
            route = []
            route_node2index = []
            # Skip depot node
            index = self.solution.Value(self.routing.NextVar(index))

            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                visited_nodes.add(node_index)
                route.append(node_index)
                route_node2index.append(int(self.data['node_to_index_map'][node_index].unique_index))

                cumul_vars = {}
                for dimension in dimensions:
                    cumul_var = self.routing.GetDimensionOrDie(dimension).CumulVar(index)
                    cumul_vars[dimension] = self.solution.Value(cumul_var)
                self.solution_data['cumul_vars'][node_index] = cumul_vars

                index = self.solution.Value(self.routing.NextVar(index))

            self.solution_data['routes'][vehicle_id] = route
            self.solution_data['routes_node2index'][vehicle_id] = route_node2index

            # Capture the final time and distance when the vehicle returns to the depot
            final_vars = {}
            for dimension in ['Time', 'Distance']:
                cumul_var = self.routing.GetDimensionOrDie(dimension).CumulVar(index)
                final_vars[dimension] = self.solution.Value(cumul_var)
            self.solution_data['final_var'][vehicle_id] = final_vars

        dropped_nodes = list(all_nodes - visited_nodes)
        dropped_index = [int(self.data['node_to_index_map'][node].unique_index) for node in dropped_nodes]

        self.solution_data['dropped_nodes'] = dropped_nodes
        self.solution_data['dropped_index'] = dropped_index

    def save_solution(self, output_path):
        if not hasattr(self, 'solution_data'):
            raise ValueError("Solution data has not been extracted. Call get_solution_data() first.")

        with open(output_path, 'w') as output_file:
            json.dump(self.solution_data, output_file, indent=4)
        print(f"Solution saved to {output_path}")

    def read_solution(self, input_path):
        if self.solution_data is not None:
            print("Overwrites solution_data")
        self.solution_data = load_json(input_path)
        print(f'Reading solution from: {input_path}')

    def print_solution(self, with_energy_constraints=False):
        """Prints solution on console."""
        status = self.routing.status()
        if self.solution is None:
            raise RuntimeError(f"No solution available. Solver status: {status}")

        print(f'Objective: {self.solution.ObjectiveValue()}')

        # Display dropped nodes.
        dropped_nodes = []
        for node in range(self.routing.Size()):
            if self.routing.IsStart(node) or self.routing.IsEnd(node):
                continue
            if self.solution.Value(self.routing.NextVar(node)) == node:
                if isinstance(self.data, EPDPInstance) or (with_energy_constraints and 'station_index' in self.data):
                    if node not in self.data['station_index']:
                        dropped_nodes.append(self.manager.IndexToNode(node))
                else:
                    dropped_nodes.append(self.manager.IndexToNode(node))

        if dropped_nodes:
            print("Dropped customer nodes: ", ", ".join(map(str, dropped_nodes)))
        else:
            print("No dropped customer nodes.")

        # Display routes
        time_dimension = self.routing.GetDimensionOrDie('Time')
        distance_dimension = self.routing.GetDimensionOrDie('Distance')

        total_time = 0
        total_distance = 0
        total_load = 0

        if isinstance(self.data, EPDPInstance) or with_energy_constraints:
            energy_dimension = self.routing.GetDimensionOrDie('Energy')
            total_energy = 0

        for vehicle_id in range(self.manager.GetNumberOfVehicles()):
            index = self.routing.Start(vehicle_id)
            if self.routing.IsEnd(self.solution.Value(self.routing.NextVar(index))):  # Vehicle does not leave depot
                continue
            plan_output = f'Route for vehicle {vehicle_id}:\n'
            route_load = 0

            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                distance_var = distance_dimension.CumulVar(index)
                route_load += self.data["demands"][node_index]

                if isinstance(self.data, EPDPInstance) or with_energy_constraints:
                    energy_var = energy_dimension.CumulVar(index)
                    plan_output += '{0} Time({1},{2}) Distance:{3} Load:{4} Energy:{5} -> '.format(
                        node_index,
                        self.solution.Min(time_var) / 2, self.solution.Max(time_var) / 2,
                        self.solution.Value(distance_var),
                        route_load,
                        self.solution.Value(energy_var))
                else:
                    plan_output += '{0} Time({1},{2}) Distance:{3} Load:{4} -> '.format(
                        node_index,
                        self.solution.Min(time_var) / 2, self.solution.Max(time_var) / 2,
                        self.solution.Value(distance_var),
                        route_load)

                index = self.solution.Value(self.routing.NextVar(index))

            # For each Route
            time_var = time_dimension.CumulVar(index)
            distance_var = distance_dimension.CumulVar(index)

            if isinstance(self.data, EPDPInstance) or with_energy_constraints:
                energy_var = energy_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2}) Distance:{3}km Load:{4}kg Energy:{5}kWh \n '.format(
                    self.manager.IndexToNode(index),
                    self.solution.Min(time_var) / 2, self.solution.Max(time_var) / 2,
                    self.solution.Value(distance_var),
                    route_load,
                    self.solution.Value(energy_var))
                total_energy += self.solution.Value(energy_var)
            else:
                plan_output += '{0} Time({1},{2}) Distance:{3}km Load:{4}kg \n '.format(
                    self.manager.IndexToNode(index),
                    self.solution.Min(time_var) / 2, self.solution.Max(time_var) / 2,
                    self.solution.Value(distance_var),
                    route_load)

            plan_output += f'Time of the route: {self.solution.Min(time_var) / 2}h\n'
            plan_output += f"Distance of the route: {self.solution.Value(distance_var)}km\n"
            print(plan_output)

            # For all truck routes
            total_distance += self.solution.Value(distance_var)
            total_load += route_load
            total_time += self.solution.Min(time_var) / 2

        print('Total time of all routes: {}h'.format(total_time))
        print('Total distance of all routes: {}km'.format(total_distance))
        if isinstance(self.data, EPDPInstance) or with_energy_constraints:
            print('Total energy of all routes: {}kWh'.format(total_energy))

    def plot_solution(self, output_path=None):
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes with positions
        for node, details in self.node_to_index_map.items():
            G.add_node(node, pos=(details.location[0], details.location[1]))

        # Define colors for each vehicle route
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'purple', 'brown', 'pink']

        # Add edges (routes) for each vehicle
        for vehicle_id in range(self.data['num_vehicles']):
            index = self.routing.Start(vehicle_id)
            route = []
            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                route.append(node)
                next_index = self.solution.Value(self.routing.NextVar(index))
                next_node = self.manager.IndexToNode(next_index)
                if next_node != self.routing.End(
                        vehicle_id) and next_node != node:  # Avoid drawing edge to the end node and self-loops
                    G.add_edge(node, next_node, color=colors[vehicle_id % len(colors)])
                index = next_index

        # Get positions for all nodes
        pos = nx.get_node_attributes(G, 'pos')

        # Draw the nodes with different colors for stations
        if isinstance(self.data, EPDPInstance):
            node_colors = ['lightpink' if node in self.data.station_index else 'lightblue' for node in G.nodes()]
        else:
            node_colors = ['lightblue' for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
        # Draw the labels
        nx.draw_networkx_labels(G, pos)

        # Get edge colors
        edges = G.edges()
        edge_colors = [G[u][v]['color'] for u, v in edges]

        # Draw the edges
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color=edge_colors)

        # Show the plot
        plt.title('Vehicle Routes')
        # plt.show()

        # save
        if output_path:
            plt.savefig(output_path)

    def plot_map_solution(self, output_path=None):
        # Create a base map
        start_coords = [self.data['node_to_index_map'][0].location[0], self.data['node_to_index_map'][0].location[1]]
        m = folium.Map(location=start_coords, tiles="Cartodb Positron", zoom_start=7)

        # Define colors for each vehicle route, station nodes, and depot
        colors = ['green', 'cadetblue', 'pink', 'purple', 'orange', 'darkgreen', 'darkblue', 'darkred', 'darkpurple',
                  'lightblue', 'lightgreen', 'lightred', 'gray', 'lightgray', 'beige']

        station_color = 'lightred'
        regular_node_color = 'blue'
        depot_color = 'black'
        dropped_customer_color = 'lightgray'

        # Add markers for dropped customers
        for node in self.solution_data['dropped_nodes']:
            if not isinstance(self.data, EPDPInstance) or node not in self.solution_data['station_index']:
                node_info = self.node_to_index_map[node]
                lat, lon = node_info.location[0], node_info.location[1]
                popup_content = f"Dropped Customer: {node}<br>Coordinates: ({lat}, {lon})"
                popup = Popup(popup_content, parse_html=True)
                tooltip = Tooltip(f"Dropped Customer: {node}")
                folium.Marker(
                    location=(lat, lon),
                    popup=popup,
                    tooltip=tooltip,
                    icon=folium.Icon(color=dropped_customer_color)
                ).add_to(m)

        # Add nodes with positions and edges (routes) for each vehicle
        for vehicle_id, route in self.solution_data['routes'].items():
            route_nodes = [0] + route + [0]
            prev_lat, prev_lon = 0, 0
            # If routes exist, default = [0,0]
            if len(route_nodes) > 2:
                for i, node in enumerate(route_nodes):
                    is_depot = (i == 0 or i == len(route_nodes) - 1)
                    offset = 0 if is_depot else (i * 0.0001)
                    node_info = self.node_to_index_map[node]
                    lat, lon = node_info.location[0] + offset, node_info.location[1] + offset

                    is_station = node in self.solution_data['station_index'] if isinstance(self.data,
                                                                                           EPDPInstance) else False
                    is_depot = (i == 0 or i == len(route_nodes) - 1)
                    color = depot_color if is_depot else (station_color if is_station else regular_node_color)
                    popup_content = f"Node: {node}<br>Coordinates: ({lat}, {lon})<br>"
                    tooltip_content = f"Node: {node}"
                    popup = Popup(popup_content, parse_html=True)
                    tooltip = Tooltip(tooltip_content)

                    folium.Marker(
                        location=(lat, lon),
                        popup=popup,
                        tooltip=tooltip,
                        icon=folium.Icon(color=color)
                    ).add_to(m)

                    # Add edge
                    if i > 0:
                        curve_points = get_curve_points((prev_lat, prev_lon), (lat, lon))
                        edge_color = colors[int(vehicle_id) % len(colors)]
                        line = folium.PolyLine(curve_points, color=edge_color, weight=2).add_to(m)
                        # plugins.AntPath(curve_points, color=colors[int(vehicle_id) % len(colors)], weight=2).add_to(m)

                        # Add arrows along the curve
                        folium.plugins.PolyLineTextPath(
                            line,
                            '      ➔      ',  # Text to be repeated along the line
                            repeat=True,
                            offset=5,
                            attributes={'fill': edge_color, 'font-weight': 'bold',
                                        'font-size': '12'}
                        ).add_to(m)

                    prev_lon, prev_lat = lon, lat

        # Save the map if output_path is provided
        if output_path:
            m.save(output_path)
            print(f"Map saved to {output_path}")
            return m
        else:
            return m
