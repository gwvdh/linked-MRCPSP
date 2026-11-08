

# Resource-Constrained Scheduling

Generate process instances, solve them with Gurobi, and export result tables.
Run commands from the repository root. Gurobi, `gurobipy`, and a valid license are required.

The default SQLite database is `database.db`. To choose another, put `--database PATH` before the command:

```bash
python main.py --database experiments.db COMMAND [OPTIONS]
```

## Commands

```bash
# Generate one instance. Add --create-pending to queue solver runs.
python main.py generate [GENERATOR OPTIONS] [--create-pending]

# Solve queued runs for one instance, all instances, or a dataset.
python main.py run INSTANCE_ID [--force]
python main.py run-all [--force]
python main.py run-dataset DATASET_ID [--force]

# Alias for run-dataset
python main.py run-all-in-dataset DATASET_ID [--force]

# Generate a dataset; --n-instances is per scarcity (default 10).
python main.py create-dataset NAME [OPTIONS]

# Print LaTeX tables for one instance or dataset.
python main.py table (--instance-id ID | --dataset-id ID) [OPTIONS]
```

Example workflow:

```bash
python main.py generate --seed 42 --create-pending --models PDT CP --scarcities 0.4 0.8
python main.py run 1
python main.py table --instance-id 1
```

## Options

- Generator options: `--number-of-processes` (2), `--arrival-rate-min` (0.2), `--arrival-rate-max` (0.4), `--max-phases` (3), `--min-base-duration` (2.0), `--max-base-duration` (5.0), `--min-resource-ratio` (0.6), `--resource-ratio-center` (1.5), `--resource-ratio-spread` (1.0), `--timeout` (600), and `--seed` (unset). One arrival rate is sampled uniformly from the configured interval for each instance. The minimum duration times minimum resource ratio must be at least 1.
- `--models`: one or more of `PDT PDDT SDT SDDT OODDT OOPDT OOPDDT MSEQCT CP`. Defaults to all models where supported.
- `--scarcities`: one or more values; defaults to `0.2` through `1.0` where supported.
- `--objective {flow-time,makespan}`: defaults to `flow-time`.
- `--force`: rerun rows already marked solved.
- `--xml-files FILE[,FILE...] ...`: optional XML template pools, one per phase. Comma-separated files form a pool; omitted templates use the bundled default.
- `create-dataset` also accepts `--description TEXT` and `--n-instances N` (default 10 per scarcity).
- `table` requires exactly one of `--instance-id ID` or `--dataset-id ID`.

Run `python main.py --help` or `python main.py COMMAND --help` for full help. Generated files are stored under `data/<instance-id>/`.


