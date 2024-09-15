import sys
import json
import yaml
from importlib.resources import files
from argparse import ArgumentParser, FileType, Namespace

from evrp.config.config import Config, PDPConfig, EPDPConfig
from evrp.util.instance_loader import load_instance
from evrp.solver.pdp import PDPSolver
from evrp.solver.epdp import EPDPSolver


def solve_pdp(args: Namespace, config: PDPConfig) -> None:
    print("Configuration used:")
    print(json.dumps(config.model_dump(), indent=4))

    # Load the instance data
    data = load_instance(
        config=config,
        problem_path=files('evrp.data.instances').joinpath(f'instance_{args.date}.csv'),
        unique_locations_path=files('evrp.data').joinpath('unique_locations.csv'),
        distance_matrix_path=files('evrp.data').joinpath('distance_matrix.json'),
        time_matrix_path=files('evrp.data').joinpath('duration_matrix.json')
    )

    # Create and solve the PDP model
    solver = PDPSolver(data=data)
    solver.create_model()
    solver.read_solution(f'../solutions/pdp/solution_pdp_{args.date}.json')

    # solver.solve_model(settings=config.solver_settings, log_search=True)
    # solver.get_solution_data()
    #
    # if args.output:
    #     solver.save_solution(args.output)

    # solver.print_solution(args.with_energy_constraints)
    # solver.plot_solution(output_path=args.plot_output)
    solver.plot_map_solution(output_path=args.map_output)


def solve_epdp(args: Namespace, config: EPDPConfig) -> None:
    print("Configuration for EPDP:")
    print(json.dumps(config.model_dump(), indent=4))

    # Load the instance data
    data = load_instance(
        config=config,
        problem_path=files('evrp.data.instances').joinpath(f'instance_{args.date}.csv'),
        unique_locations_path=files('evrp.data').joinpath('unique_locations.csv'),
        distance_matrix_path=files('evrp.data').joinpath('distance_matrix.json'),
        time_matrix_path=files('evrp.data').joinpath('duration_matrix.json'),
        charging_stations_path=files('evrp.data').joinpath('stations_mapping.csv')
    )

    # Create and solve the EPDP model
    solver = EPDPSolver(data=data)
    solver.create_model()
    solver.read_solution(f'solutions/epdp/solution_epdp_{args.date}.json')

    # solver.solve_model(settings=config.solver_settings,
    #                    log_search=True)  # , initial_solution_filepath='solutions/0418epdp.json')
    #
    # if args.output:
    #     solver.get_solution_data(with_energy_constraints=True)
    #     solver.save_solution(args.output)

    # solver.print_solution(with_energy_constraints=True)
    # solver.plot_solution(output_path=args.plot_output)
    solver.plot_map_solution(output_path=args.map_output)


def dump_config(args: Namespace, config: Config) -> None:
    """Dump the currently active config, either default or parsed from args."""

    def cfg_str_representer(dumper, in_str):
        if "\n" in in_str:  # use '|' style for multiline strings
            return dumper.represent_scalar("tag:yaml.org,2002:str", in_str, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", in_str)

    yaml.representer.SafeRepresenter.add_representer(str, cfg_str_representer)
    yaml.safe_dump(config.dict(), args.output, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = ArgumentParser(description="Solve PDP or EPDP problems.")
    parser.add_argument("-v", "--version", action="version", version="1.0")

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-c", "--config",
        type=FileType("rt", encoding="utf-8"),
        help="Path to the configuration file. Default settings can be dumped using the `dump-config` command.",
    )
    parent_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for the solution."
    )

    parent_parser.add_argument(
        "--plot_output",
        "-po",
        type=str,
        help="Output file path for the plot."
    )

    parent_parser.add_argument(
        "--map_output",
        "-mo",
        type=str,
        help="Output file path for the map."
    )

    parent_parser.add_argument(
        "--date",
        type=str,
        default="2023-04-18",
        help="Date for filtering bookings in the format dd/mm/yyyy."
    )

    # Solver settings arguments
    parent_parser.add_argument(
        "--first_solution_strategy",
        type=str,
        help="First solution strategy for the solver."
    )
    parent_parser.add_argument(
        "--local_search_metaheuristic",
        type=str,
        help="Local search metaheuristic for the solver."
    )
    parent_parser.add_argument(
        "--time_limit",
        type=int,
        help="Time limit for the solver."
    )
    parent_parser.add_argument(
        "--lns_time_limit",
        type=int,
        help="LNS time limit for the solver."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # PDP parser
    pdp_parser = subparsers.add_parser("pdp", parents=[parent_parser], help="Pickup and Delivery Problem (PDP)")
    pdp_parser.add_argument(
        "--with_energy_constraints",
        action="store_true",
        help="Include energy constraints (only for PDP)."
    )

    # EPDP parser
    epdp_parser = subparsers.add_parser("epdp", parents=[parent_parser],
                                        help="Energy Pickup and Delivery Problem (EPDP)")

    # Dump config parser
    dump_parser = subparsers.add_parser("dump-config", help="Dump default config to a file")
    dump_parser.add_argument(
        "--output",
        "-o",
        type=FileType("wt", encoding="utf-8"),
        default=sys.stdout,
        help="Output file path for the config."
    )

    args = parser.parse_args()

    config = Config()

    if args.command in ['pdp', 'epdp'] and args.config:
        user_config = yaml.safe_load(args.config)
        config = Config(
            pdp_config=PDPConfig(**user_config.get('pdp_config', {})),
            epdp_config=EPDPConfig(**user_config.get('epdp_config', {}))
        )

    if args.command in ['pdp', 'epdp']:
        # Override solver settings with CLI arguments
        if args.first_solution_strategy:
            if args.command == 'pdp':
                config.pdp_config.solver_settings.first_solution_strategy = args.first_solution_strategy
            elif args.command == 'epdp':
                config.epdp_config.solver_settings.first_solution_strategy = args.first_solution_strategy
        if args.local_search_metaheuristic:
            if args.command == 'pdp':
                config.pdp_config.solver_settings.local_search_metaheuristic = args.local_search_metaheuristic
            elif args.command == 'epdp':
                config.epdp_config.solver_settings.local_search_metaheuristic = args.local_search_metaheuristic
        if args.time_limit:
            if args.command == 'pdp':
                config.pdp_config.solver_settings.time_limit = args.time_limit
            elif args.command == 'epdp':
                config.epdp_config.solver_settings.time_limit = args.time_limit
        if args.lns_time_limit is not None:
            if args.command == 'pdp':
                config.pdp_config.solver_settings.lns_time_limit = args.lns_time_limit
            elif args.command == 'epdp':
                config.epdp_config.solver_settings.lns_time_limit = args.lns_time_limit

        if args.command == 'pdp':
            solve_pdp(args, config.pdp_config)
        elif args.command == 'epdp':
            solve_epdp(args, config.epdp_config)
    elif args.command == 'dump-config':
        dump_config(args, config)


if __name__ == '__main__':
    main()
