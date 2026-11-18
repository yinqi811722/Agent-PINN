# Agent-PINN

Model implementations, aggregate quarterly input data, and instructions for running the forecasting experiments.

## Models

`src/run_direct_pinn_forecasting_30seeds_formal.py` implements Agent-PINN (M1, K50/K75/K100), the no-overlap PINN baseline (M0), CNN, N-BEATS and InformerLite. The PINN objective combines reconstruction, ODE-residual and training-boundary losses.

## Setup

Clone the repository and use Python 3.13:

```bash
git clone https://github.com/yinqi811722/Agent-PINN.git
cd Agent-PINN
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The `data/` directory includes the three aggregate CSVs used as forecasting inputs:

- `quarterly_counts_and_overlap_K50_2018Q1_2025Q4.csv`
- `quarterly_counts_and_overlap_K75_2018Q1_2025Q4.csv`
- `quarterly_counts_and_overlap_K100_2018Q1_2025Q4.csv`

Each file contains 32 chronological quarters from 2018Q1 to 2025Q4. K denotes the number of core concepts used to compute overlap. Publication-count columns are identical across the three files; overlap columns depend on K.

| Column | Meaning |
|---|---|
| `quarter` | Calendar quarter |
| `CNN_count`, `GNN_count`, `Transformer_count` | Quarterly publication counts for each field |
| `O_CNN_GNN`, `O_CNN_Transformer`, `O_GNN_Transformer` | Pairwise technical-overlap values for the specified K |

Field-specific publication totals are CNN 103,363, GNN 22,889 and Transformer 26,192. These are sums of each field's quarterly counts, not a deduplicated total across fields.

### Temporal split

The first 28 quarters (2018Q1–2024Q4) are used for training and the final four (2025Q1–2025Q4) for testing. No separate validation set is used. Normalization statistics are estimated on the training period only. During forecasting, overlap is fixed at the last training-quarter value (`future_o_mode=last`); test-quarter overlap values in the CSVs are not used as forecast inputs.

### Source and retrieval

Publication counts originate from Web of Science Core Collection, using Topic (`TS`) searches over 2018Q1–2025Q4:

**CNN**

```text
TS=("Convolutional Neural Network*")
OR TS=("Convolutional Neural Networks")
```

**GCN/GNN**

```text
TS=("Graph Neural Network*")
OR TS=("Graph Convolutional Network*")
OR TS=("GNN")
OR TS=("GCN")
```

**Transformer**

```text
TS=("Transformer")
AND TS=("Deep Learning" OR "Neural Network*" OR "Attention Mechanism"
OR "Natural Language Processing" OR "Computer Vision"
OR "Vision Transformer*" OR "Large Language Model*")
```

Records are assigned to calendar quarters by formal publication date and deduplicated using the WoS accession number. Missing dates are marked `unknown_date` and are not assigned to Q1. The GCN/GNN query covers graph-neural-network-related literature, including graph convolutional networks; GCN and GNN are not treated as synonymous terms.

This repository distributes aggregate forecasting inputs. Raw WoS records, literature full texts and the graph-extraction application are not included. The supplied code runs the forecasting stage from these inputs.

## Run

Check the command without training:

```bash
python scripts/run_experiment.py --dry-run
```

Run a short execution check with the included data:

```bash
python scripts/run_experiment.py --smoke --cpu
```

Run the full configuration (30 seeds, three field pairs):

```bash
python scripts/run_experiment.py
```

Add `--cpu` to force CPU execution. Hyperparameters and training budgets are specified in `configs/formal_30seed.json`; the short execution check uses `configs/smoke.json`. Generated outputs are written to `outputs/`.

The short check verifies execution with reduced training budgets; it does not reproduce the full experiment's accuracy. The full configuration trains 30 seeds over three field pairs and seven model settings. All K settings are reported; selecting the lowest test Macro-NMAE retrospectively is exploratory.
