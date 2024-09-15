from ortools.constraint_solver import routing_enums_pb2, pywrapcp


class CallBack:

    def __init__(self, data, manager):
        self.manager = manager
        self.data = data

    def distance_callback(self, from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        distance = self.data["distance_matrix"][from_node][to_node]
        # print(f'distance: {distance}')
        return distance

    def demand_callback(self, from_index):
        """Returns the demand of the node."""
        from_node = self.manager.IndexToNode(from_index)
        return self.data['demands'][from_node]

    def time_callback(self, from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        travel_time = self.data['time_matrix'][from_node][to_node]
        service_time = self.data['service_time'][to_node]
        return travel_time + service_time

    def energy_callback(self, from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        distance = self.data["distance_matrix"][from_node][to_node]
        energy_consumption = distance * self.data['consumption_rate']
        energy_consumption = int(energy_consumption)
        if from_node == 0:
            return 390 - energy_consumption

        else:
            return -energy_consumption


class SolutionCallback:
    def __init__(self, routing: pywrapcp.RoutingModel):
        self.routing = routing
        self.current_min_objective = float('inf')
        self.solution_counter = 0

        self.solution_log = {}

    def __call__(self):
        time = self.routing.solver().WallTime()

        current_value = self.routing.CostVar().Value()
        if current_value < self.current_min_objective:
            self.current_min_objective = current_value

        self.solution_log[self.solution_counter] = {
            'objective_value': current_value,
            'min_objective_value': self.current_min_objective,
            'time': time,
        }

        self.solution_counter += 1
