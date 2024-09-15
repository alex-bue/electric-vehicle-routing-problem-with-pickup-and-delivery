The code was stripped of sensitive data so it is not actually runnable since the required data is missing. It serves as a view only repo to showcase the logic used to get to our thesis results. This core logic (mainly build around Google OR-Tools) lives inside the `evrp` package.

## Installation

To install the package in editable mode, allowing you to make changes to the code and immediately see the effects without reinstalling the package, follow these steps:

1. Open a terminal.

2. Navigate to the root directory of the project where `setup.py` is located.

4. Setup virtual environment and install dependencies.

    ```bash
    pip install -r requirements.txt
    ```

5. Install package in editable mode:

   ```bash
   pip install -e .
   ```

### Usage

The `evrp` can be executed via the command line with the following commands and options.

#### Commands

- `pdp`: Pickup and Delivery Problem
- `epdp`: Energy Pickup and Delivery Problem
- `dump-config`: Dump the default configuration


#### Example PDP Command

```bash
evrp pdp -o solutions/output.json --date 18/04/2023 --with_energy_constraints
```

#### Options:
- `-o, --output`: Output file path for the solution (default: stdout)
- `--plot_output`: Output file path for the plot
- `--date`: Date for filtering bookings (`dd/mm/yyyy`)
- `--with_energy_constraints`: Include energy constraints (only for PDP)

#### Configuration

You can provide a configuration file before invoking `pdp` or `epdp` to override the default model values:

```bash
evrp -c path/to/config.yaml pdp -o solutions/output.json --date 18/04/2023 --with_energy_constraints
```

The default configuration can be dumped and saved by specifying the `--output` flag:

```bash
evrp dump-config -o config.yaml
```
