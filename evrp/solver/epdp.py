from ortools.constraint_solver import pywrapcp
from ..util.call_back import CallBack
from ..models.data_model import EPDPInstance
from .base_solver import Solver


class EPDPSolver(Solver):
    def __init__(self, data: EPDPInstance):
        super().__init__(data)

    def create_model(self):

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
            100,  # allow waiting time
            100,  # maximum time per vehicle
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

        # Energy constraint
        energy_callback_index = self.routing.RegisterTransitCallback(call_back.energy_callback)
        self.routing.AddDimension(
            energy_callback_index,
            self.data["energy_capacity"],
            self.data["energy_capacity"],
            False,
            'Energy')
        energy_dimension = self.routing.GetDimensionOrDie('Energy')

        for i in range(len(self.data['distance_matrix'])):
            index = self.manager.NodeToIndex(i)
            # if i != self.data['depot']:
            #     energy_dimension.CumulVar(index).SetMin(20)
            if i not in self.data['station_index']:  # Locations
                energy_dimension.SlackVar(index).SetValue(0)

        # Allow dropping station nodes
        for station in self.data['station_index']:
            index = self.manager.NodeToIndex(station)
            self.routing.AddDisjunction([index], 0)

        # Allow dropping customer nodes in case of energy constraints
        for pickup, delivery in self.data['pickups_deliveries']:
            pickup_index = self.manager.NodeToIndex(pickup)
            delivery_index = self.manager.NodeToIndex(delivery)
            self.routing.AddDisjunction([pickup_index, delivery_index], 100, 2)

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
