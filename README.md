# Agent-PINN

Model implementations and instructions for running the forecasting experiments.

## Models

`src/run_direct_pinn_forecasting_30seeds_formal.py` implements Agent-PINN (M1, K50/K75/K100), the no-overlap PINN baseline (M0), CNN, N-BEATS and InformerLite. The PINN objective combines reconstruction, ODE-residual and training-boundary losses.

## Setup

Use Python 3.13 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Input

Data are not distributed in this repository. Place your input CSVs in `data/`:

- `quarterly_counts_and_overlap_K50_2018Q1_2025Q4.csv`
- `quarterly_counts_and_overlap_K75_2018Q1_2025Q4.csv`
- `quarterly_counts_and_overlap_K100_2018Q1_2025Q4.csv`

Required columns:

```text
quarter,CNN_count,GNN_count,Transformer_count,O_CNN_GNN,O_CNN_Transformer,O_GNN_Transformer
```

The experiment configuration expects 32 chronological quarters, 2018Q1 through 2025Q4, with the first 28 used for training and the final four for testing.

## Run

Check the command without training:

```bash
python scripts/run_experiment.py --dry-run
```

Run a short execution check after supplying the inputs:

```bash
python scripts/run_experiment.py --smoke --cpu
```

Run the full configuration (30 seeds, three field pairs):

```bash
python scripts/run_experiment.py
```

Add `--cpu` to force CPU execution. Hyperparameters and training budgets are specified in `configs/formal_30seed.json`; the short execution check uses `configs/smoke.json`. Generated outputs are written to `outputs/`.
