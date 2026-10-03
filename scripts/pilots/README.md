# Pilot Scripts

These scripts contain the calibration experiments reported in Appendix A. They use classical local search and exact subgraph repair only; no QAOA is run.

## Prerequisites

The project `src` modules must be importable. For example:

```bash
PYTHONPATH=src python scripts/pilots/pilot_repairability.py
```

Required project modules: `maxcut`, `local_search`, `neighbourhood`, `exact_repair`, `boundary`.

External packages: `numpy`, `networkx`.

## 1. `pilot_repairability.py`

Supports Appendix A, Table A.1.

Measures the unconditioned positive-headroom rate for different graph families, graph sizes, and neighbourhood sizes using single draw selection.

```bash
PYTHONPATH=src python scripts/pilots/pilot_repairability.py
```

Settings:

* graph seeds: 0–7
* neighbourhoods per local optimum: 20
* graph families: 3-regular and ER
* graph sizes: 50 and 100
* neighbourhood sizes: 10 and 12
* ER probability: `3 / (n - 1)`

Outputs:

* `pilot_repairability_table.csv`
* `pilot_repairability_rows.json`

## 2. `neighbourhood_selection_check.py`

Supports Appendix A, Table A.2.

Compares single-draw selection with best of 6 selection using exact improvement or boundary norm.

```bash
PYTHONPATH=src python scripts/pilots/neighbourhood_selection_check.py
```

Settings:

* graph seeds: 0–4
* draws per local optimum: 10
* candidates: 6
* 3-regular: `n = 100`, `k = 12`
* ER: `n = 50`, `k = 12`

Output:

* `neighbourhood_selection_check.csv`

## Reproducibility

Both scripts use the same seed policy:

```text
initial assignment seed = graph_seed + 1000
local search seed       = graph_seed
neighbourhood/draw seed = graph_seed * 100 + index
```

Using the same settings and seeds reproduces the reported calibration results.

## Calibration Outcome

The pilot results led to the settings used in the main experiments:

* `k = 12`
* 3-regular graphs with `n = 100`
* ER graphs with `n = 50`
* best of 6 neighbourhood selection based on exact improvement

Boundary norm selection was tested but not adopted. The unconditioned single draw repairability rates are reported separately in Appendix A.

