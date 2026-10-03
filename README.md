# Boundary-Field QAOA as a Repair Operator for Max-Cut Local Search

MSc Artificial Intelligence Dissertation
King's College London, 2025/26

Author: Amir Lorvand
Supervisor: Dr. Kathleen Steinhofel

## Overview

This repository contains the implementation and experiments for the dissertation:

**Boundary-Field QAOA as a Repair Operator for Max-Cut Local Search: An Empirical Study of Transferred-Angle, Warm-Started Repair**

The project studies QAOA as a small repair operator for Max-Cut local search. Classical 1-flip local search is first used to reach a local optimum. A small neighbourhood is then selected and repaired while the rest of the graph remains fixed.

The effect of the frozen exterior is represented using boundary field terms in the QAOA cost Hamiltonian. The repair is warm started from the current classical solution and uses transferred fixed angles rather than optimising the angles separately for every subproblem.

The experiments investigate three main questions:

* whether boundary fields improve repair compared with the same QAOA operator without boundary terms.
* whether transferred fixed angles remain effective after boundary field terms are added.
* how much of the exact repair improvement is captured as the shot budget increases.

All experiments use exact statevector simulation. No claim of quantum advantage or speed up is made.

## Project Structure

```text
QLS_MAXCUT_PROJECT/
│
├── src/
│   ├── angles.py
│   ├── boundary.py
│   ├── exact_repair.py
│   ├── local_search.py
│   ├── maxcut.py
│   ├── neighbourhood.py
│   ├── qaoa.py
│   ├── repair.py
│   ├── sampling.py
│   └── warm_start.py
│
├── scripts/
│   ├── analysis/
│   ├── experiments/
│   ├── figures/
│   └── pilots/
│
├── tests/
│   ├── test_angles.py
│   ├── test_boundary.py
│   ├── test_exact_repair.py
│   ├── test_local_search.py
│   ├── test_maxcut.py
│   ├── test_neighbourhood.py
│   ├── test_qaoa.py
│   ├── test_repair.py
│   ├── test_sampling.py
│   └── test_warm_start.py
│
├── results/
├── analysis_results/
├── figures/
└── README.md
```

### `src/`

Contains the main implementation of the repair method.

* `maxcut.py` — Max-Cut objective and cut calculations.
* `local_search.py` — classical steepest-improvement 1-flip local search.
* `neighbourhood.py` — repair neighbourhood generation and edge partitioning.
* `boundary.py` — boundary field construction and subproblem Hamiltonian.
* `exact_repair.py` — brute force exact repair used for neighbourhood selection and the exact repair ceiling.
* `qaoa.py` — statevector QAOA implementation and expected cost calculations.
* `warm_start.py` — preparation of the warm start initial state.
* `angles.py` — loading, rescaling, and deploying transferred QAOA angles.
* `sampling.py` — sampling, candidate evaluation, and repair acceptance.
* `repair.py` — complete repair call and multi round repair procedure.

### `scripts/experiments/`

Contains the main experimental runners.

The main experiment evaluates the boundary aware and no-boundary QAOA repair arms using paired repair calls across the tested graph families and shot budgets.

The QAOA-OPT experiment separately optimises QAOA angles using COBYLA and is used as the reference for the transferred-angle comparison.

### `scripts/analysis/`

Contains the analysis scripts used to produce the numerical summaries for the experimental hypotheses.

The H1 and H3 analyses first aggregate repeated repair calls at graph level before statistical analysis. Positive headroom cases are used for the boundary and captured-gain comparisons.

The H2 analysis compares transferred angle expected cut with the expected cut reached by per subproblem angle optimisation.

### `scripts/figures/`

Contains the scripts used to generate the figures reported in Chapter 5.

These include:

* Figure 5.1 — boundary field advantage.
* Figure 5.2 — captured gain distribution.
* Figure 5.3 — captured gain against shot budget.
* Figure 5.4 — transferred angle quality and optimisation cost.
* Figure 5.5 — boundary field strength against boundary advantage.

Generated figures are saved in `figures/`.

### `scripts/pilots/`

Contains the classical calibration experiments reported in Appendix A.

These scripts do not run QAOA.

* `pilot_repairability.py` measures the unconditioned positive headroom rate for different graph sizes and neighbourhood sizes.
* `neighbourhood_selection_check.py` compares single draw neighbourhood selection with best of six selection using exact improvement or boundary norm.

The pilot results were used to select:

* neighbourhood size `k = 12`.
* 3-regular graphs with `n = 100`.
* ERdos-Renyi graphs with `n = 50`.
* best of six neighbourhood selection using exact improvement.

See `scripts/pilots/README.md` for further details.

### `tests/`

Contains the verification test suite described in Appendix B.

The tests cover the full implementation, including the Max-Cut objective, local search, neighbourhood selection, boundary reconstruction, exact repair, QAOA state evolution, warm start ordering, angle rescaling, sampling, acceptance, and the complete repair procedure.

The final verification suite contains 84 tests.

## Experimental Settings

The main experiments use:

```text
Graph families:        3-regular and ERdos-Renyi
3-regular size:        n = 100
ERdos-Renyi size:      n = 50
ER probability:        3 / (n - 1)
Neighbourhood size:    k = 12
Candidate selection:   best of 6 by exact improvement
QAOA depth:            p = 1
Shot budgets:          32, 128, 512, 2048
Warm start epsilon:    0.1
Graph instances:       20 per family
Repair calls:          8 per graph
```

The transferred p = 1 angles are based on the fixed angle values of Wurtz and Lykov and are rescaled according to the effective Hamiltonian coefficient scale.

QAOA-OPT uses COBYLA with the transferred angles as the first starting point and two additional random restarts. Its quality measure is the exact statevector expected cut.

## Installation

Python 3.12 was used for the experiments.

A virtual environment is recommended.

For example:

```bash
python -m venv myenv
source myenv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The project uses packages including:

```text
numpy
networkx
scipy
matplotlib
pytest
```

The local `myenv/` directory is not required to reproduce the project and should not be distributed with the repository.

## Running the Main Experiment

Run commands from the project root with `src` on `PYTHONPATH`.

For example:

```bash
PYTHONPATH=src python scripts/experiments/run_experiment.py
```

The main outputs are written to `results/`, including:

```text
results_rows.csv
results_rows.jsonl
run_manifest.json
```

The manifest records the experimental settings, seed policy, and environment information required for reproducibility.

## Running the QAOA-OPT Experiment

```bash
PYTHONPATH=src python scripts/experiments/run_qaoa_opt.py
```

Outputs include:

```text
results/qaoa_opt_rows.csv
results/qaoa_opt_manifest.json
```

This experiment provides the per subproblem optimised angle reference used for H2.

## Analysis

Run the analysis scripts from the project root.

For example:

```bash
PYTHONPATH=src python scripts/analysis/analyse_results.py results/results_rows.csv
```

and:

```bash
PYTHONPATH=src python scripts/analysis/analyse_h2.py
```

Generated analysis summaries are stored in:

```text
analysis_results/
```

## Figures

The Chapter 5 figures can be regenerated from the saved experimental results.

For the H1 and H3 figures:

```bash
python scripts/figures/make_figures.py results/results_rows.csv
```

For the H2 quality-cost figure:

```bash
python scripts/figures/make_figure_5_4.py results/qaoa_opt_rows.csv
```

Generated PNG and PDF files are written to:

```text
figures/
```

## Pilot Experiments

The Appendix A calibration experiments can be reproduced using:

```bash
PYTHONPATH=src python scripts/pilots/pilot_repairability.py
```

and:

```bash
PYTHONPATH=src python scripts/pilots/neighbourhood_selection_check.py
```

The pilot scripts use fixed graph, initial assignment, local search, and neighbourhood seeds. See `scripts/pilots/README.md` for the complete calibration settings.

## Verification Tests

Run the complete verification suite from the project root:

```bash
PYTHONPATH=src pytest tests/
```

The final implementation passes all 84 verification tests.

## Reproducibility

Fixed integer seeds are used throughout the experiments.

The main experimental runners save their settings and environment information in manifest files. Individual repair results also record the graph, neighbourhood, experimental arm, shot budget, exact repair ceiling, boundary statistics, deployed angles, and other diagnostics required for later analysis.

The pilot experiments use their own documented deterministic seed policy.

Saved result files are included so that the statistical analysis and figures can be regenerated without rerunning the complete statevector experiments.

## Notes

The implementation is designed for research and simulation rather than deployment on quantum hardware.

Exact repair performs exhaustive search over the selected neighbourhood and is therefore used only because the repair neighbourhood is small. It acts as a calibration and evaluation ceiling rather than as a competing scalable solver.

The QAOA simulations are noiseless statevector simulations, so the reported results should not be interpreted as evidence of practical quantum advantage.



