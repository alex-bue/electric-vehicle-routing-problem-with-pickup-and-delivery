from ortools.constraint_solver import pywrapcp
from ..util.call_back import CallBack
from .base_solver import Solver
from ..models.data_model import PDPInstance


class PDPSolver(Solver):
    def __init__(self, data: PDPInstance):
        super().__init__(data)

    def create_model(self, with_energy_constraints=False):

        # Create the routing index manager
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.data['distance_matrix']),
            self.data['num_vehicles'],
            self.data['depot'])

        # Create Routing Model
        self.routing = pywrapcp.RoutingModel(self.manager)

        # Create Callback functions
        call_back = CallBack(self.data, self.manager)

        # Create and register a transit callback
        distance_callback_index = self.routing.RegisterTransitCallback(call_back.distance_callback)

        # Add distance constraints
        self.routing.AddDimension(
            distance_callback_index,
            0,  # no slack
            10000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            'Distance')
        distance_dimension = self.routing.GetDimensionOrDie('Distance')

        # Add vehicle capacity constraints
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(call_back.demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Add time window constraints
        time_callback_index = self.routing.RegisterTransitCallback(call_back.time_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
        self.routing.AddDimension(
            time_callback_index,
            50,  # allow waiting time
            50,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            'Time')
        time_dimension = self.routing.GetDimensionOrDie('Time')

        # Add time window constraints for each location except depot
        depot_idx = self.data['depot']
        for location_idx, time_window in enumerate(self.data['time_windows']):
            if location_idx == depot_idx:
                continue
            index = self.manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(self.data['num_vehicles']):
            index = self.routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                self.data['time_windows'][depot_idx][0],
                self.data['time_windows'][depot_idx][1])

        # Instantiate route start and end times to produce feasible times.
        for i in range(self.data['num_vehicles']):
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.Start(i)))
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.End(i)))

        # Energy constraints
        if with_energy_constraints:
            energy_callback_index = self.routing.RegisterTransitCallback(call_back.energy_callback)
            self.routing.AddDimension(
                energy_callback_index,
                0,
                self.data["energy_capacity"],
                True,
                'Energy')

            # Allow dropping customer nodes in case of energy constraints
            for pickup, delivery in self.data['pickups_deliveries']:
                pickup_index = self.manager.NodeToIndex(pickup)
                delivery_index = self.manager.NodeToIndex(delivery)
                self.routing.AddDisjunction([pickup_index, delivery_index], 10000, 2)

        # Define Transportation Requests.
        for request in self.data['pickups_deliveries']:
            pickup_index = self.manager.NodeToIndex(request[0])
            delivery_index = self.manager.NodeToIndex(request[1])
            self.routing.AddPickupAndDelivery(pickup_index, delivery_index)
            # constraints: pickup and delivery should be done by the same truck
            self.routing.solver().Add(
                self.routing.VehicleVar(pickup_index) == self.routing.VehicleVar(
                    delivery_index))
            # constraints: pickup should be done before delivery
            self.routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <=
                distance_dimension.CumulVar(delivery_index))
