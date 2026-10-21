# Thesis Simulation Project

**Thesis Reference**
For the comprehensive theoretical background, cardiovascular mobility assumptions, and clinical implications driving this codebase, please refer to the primary manuscript: 
* Volpi, Davide. "Flow-Guided Cardiovascular Physiological Mapping from Noisy Multimodal Data." Master's thesis, Facultat d'Informàtica de Barcelona (FIB), Universitat Politècnica de Catalunya (UPC) - BarcelonaTech, 2026.

**Related Publications**
If you utilize this framework or its datasets in your own research, please acknowledge the foundational works:
* D. Volpi, F. Lemic, G. Encinas-Lago, E. Shitiri, S. Abadal, E. Alarcón, and X. Costa-Pérez, "Reliable Medical Condition Detection from Noisy Location-aware Multimodal Cardiovascular IoBNT Telemetry," in Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '26), Nov. 3-6, 2026, pp. 1-12. *(To appear)*
* D. Volpi, E. Shitiri, E. Alarcón, S. Abadal, and F. Lemic. "Diagnostic Monitoring and Adaptive Drug Delivery from Noisy Location-Aware Multimodal IoBNT Telemetry," in IEEE Transactions on Medical Imaging (IEEE TMI). *(To be submitted)*

## Project Overview

This repository implements an end-to-end artificial intelligence framework for the Internet of Bio-Nano Things (IoBNT). It translates raw, flow-guided in-vivo telemetry into actionable clinical insights by systematically mitigating cardiovascular stochasticity, nanoscale hardware imperfections, and localization errors. 

The architecture is split across two distinct Git branches to handle parallel clinical objectives:
* **Branch I (Diagnostic):** Processes noisy sensor data through parallel estimation streams—a 1-D Kalman filter for raw measurements and a second-order, position-and-velocity Kalman filter for aggregated measurements—culminating in a spatial Naive Bayes classifier that aims to identify underlying medical conditions while suppressing false diagnoses.
* **Branch II (Therapeutic):** Explores the conceptual use of the nanonetwork as an autonomous in-vivo liquid biopsy. Using a simulated digital-patient environment, it utilizes a Deep Recurrent Q-Network (DRQN) controller with an LSTM core to anticipate delayed physiological reactions and manage precision chemotherapy dosing within modeled organ-safety limits.

## Requirements

- Python 3.10 is the version used by the checked-in virtual environment.
- Dependencies are listed in the `requirements.txt` for the checked-out branch. After switching branches, verify the requirements file and install that branch's dependencies in the active environment before running its code.
- Branch I simulation and tuning pipelines use multiprocessing and can require substantial memory and runtime.
- NS-3 is an external prerequisite only when generating new network traces.

## Setup

Run commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If using an existing compatible environment, activate it instead of creating a new one. Recheck and install the checked-out branch's dependencies after switching branches; the two branches may have different requirements.

## Branch I — Nanonetwork Simulation and Disease Detection

Branch I models nanobot sampling across a vascular network, simulates physiological disease progression, and evaluates detection of anomalous pH and temperature trends using Kalman-filter-based detectors. It supports raw and statistically aggregated streams, static thresholds and adaptive z-score thresholds, and generalized or targeted clinical evaluation cohorts. These are simulation and evaluation results, not evidence of clinical validation or guaranteed real-world diagnostic performance.

### Branch I workflow

#### 1. Generate NS-3 network traces (optional)

`generate_simulations.py` runs the `toy_simulation` NS-3 program five times. Its default configuration expects NS-3 at `~/ns-allinone-3.36.1/ns-3.36.1` and writes traces to `~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces/`.

```bash
python generate_simulations.py
```

Before generating the Branch I disease datasets, make the five trace files available as `raw_ns3_traces/1.csv` through `raw_ns3_traces/5.csv` under the repository root. `dataset_generation.py` reads from that repository-local directory, so copy the generated files there if the NS-3 output directory differs. The NS-3 paths are constants in `generate_simulations.py` and may need adjustment for your machine.

#### 2. Build disease datasets

```bash
python dataset_generation.py
```

This creates raw and aggregated Parquet datasets for pH, temperature, and combined pH/temperature disease scenarios under `datasets/`. The datasets include disease labels, target zones, and ground-truth onset and peak times.

#### 3. Tune detector parameters

Run generalized static-threshold optimization:

```bash
python parameter_tuning.py
```

Run generalized adaptive z-score optimization:

```bash
python z_parameter_tuning.py
```

Both scripts tune raw and aggregated detectors for pH, temperature, and combined modalities. These are long-running multiprocessing jobs. They write Optuna databases in the repository root and serialized studies under `tuning_results/`.

The targeted tuning scripts process both post-operative and diabetes cohorts:

```bash
python targeted_parameter_tuning.py
python targeted_zscore_tuning.py
```

These tuning entry points remove their matching old Optuna database files and overwrite their corresponding output pickle files before starting. Back up results you want to retain before rerunning them.

#### 4. Calibrate Bayesian likelihoods (optional)

```bash
python probs_extraction.py
```

This extracts empirical detection likelihoods from generated datasets and available tuning studies, then writes `calibrated_likelihoods.txt`. It expects the relevant Parquet datasets and serialized optimization studies to exist.

#### 5. Evaluate detectors

Run the standard evaluation suites:

```bash
python eval.py
```

Run the Bayesian-classifier evaluation suites:

```bash
python bayesian_eval.py
```

Both scripts run generalized and targeted evaluations for available static and z-score tuning results. They write terminal output to `full_evaluation_results.txt`. The Bayesian evaluation reads `calibrated_likelihoods.txt` when present and otherwise falls back to default theoretical likelihood values.

### Branch I files

| File or directory | Purpose |
| --- | --- |
| `generate_simulations.py` | Generates baseline NS-3 network traces |
| `dataset_generation.py` | Builds labeled raw and aggregated disease datasets from traces |
| `simulation.py` | Nanobot sampling and energy-constrained simulation engine |
| `diseases.py` | Disease progression simulation over network traces |
| `drift_detection.py` | Raw and aggregated Kalman-filter anomaly detectors |
| `parameter_tuning.py` | Generalized static-threshold Optuna tuning |
| `z_parameter_tuning.py` | Generalized z-score Optuna tuning |
| `targeted_parameter_tuning.py` | Static-threshold tuning for post-operative and diabetes cohorts |
| `targeted_zscore_tuning.py` | Z-score tuning for post-operative and diabetes cohorts |
| `evaluation.py` | Evaluation suites and clinical metrics |
| `bayesian_evaluation.py` | Evaluation suites with spatial Bayesian classification |
| `eval.py` | Runs non-Bayesian generalized and targeted evaluations |
| `bayesian_eval.py` | Runs Bayesian generalized and targeted evaluations |
| `probs_extraction.py` | Extracts calibrated likelihoods for the Bayesian classifier |
| `plots.py` | Simulation, detector, and Optuna visualizations |
| `utils.py` | Data mappings, sensor statistics, and energy-model helpers |
| `raw_ns3_traces/` | Expected location for the five baseline trace CSV files |
| `datasets/` | Generated raw and aggregated Parquet datasets |
| `tuning_results/` | Serialized Optuna studies and tuning outputs |
| `Random.ipynb`, `Tests.ipynb` | Exploratory analysis and tests |

`simulation.py` is a reusable module; its `__main__` block does not start a standalone simulation. Its default input is `position_data.csv`. The disease-dataset generation workflow instead passes each `raw_ns3_traces/<n>.csv` file to `DiseaseSimulator`.

## Branch II — Predictive Chemotherapy Simulation

Branch II contains a digital-patient simulation and a Deep Recurrent Q-Network (DRQN) workflow for exploring chemotherapy dosing strategies. It combines simulated pH and temperature observations with pharmacokinetic, pharmacodynamic, tumor-growth, and toxicity models. Its results are limited to the simulated environment and do not establish clinical safety, efficacy, or real-world deployment.

The files and data described in this section belong to the Branch II checkout and are not present in the current Branch I checkout. Switch to the Branch II Git branch before running its commands.

### Branch II workflow

The standard training and evaluation workflow uses `datasets/1_week_sampling_schedule.parquet`, which must exist on the Branch II checkout. Training and evaluation use a CUDA-enabled GPU when available, or CPU otherwise.

#### 1. Train a cancer-specific DRQN

For example:

```bash
python train.py --cancer Glioblastoma --episodes 1000
```

Available cancer profile names:

- `Hepatocellular_Carcinoma`
- `Colorectal_Cancer`
- `Lung_Cancer`
- `Renal_Cell_Carcinoma`
- `Glioblastoma`

Training uses the dataset at `datasets/1_week_sampling_schedule.parquet`. If a matching configuration exists in `config/`, its optimized parameters are loaded; otherwise, the script uses its defaults.

Optional training modes:

```bash
# Run 20 Optuna trials and save the best parameters
python train.py --cancer Glioblastoma --optimize

# Run five independent training runs
python train.py --cancer Glioblastoma --stability_check
```

Hyperparameter optimization and stability checks can be computationally expensive.

#### 2. Evaluate a trained model

Run the diagnostic stress-test evaluation for one profile:

```bash
python evaluate.py --cancer Glioblastoma
```

Run the 100-episode frozen evaluation:

```bash
python evaluate.py --cancer Glioblastoma --frozen
```

The frozen evaluation reports simulation results across typical and stress scenarios, including growth-rate, resistance, and toxicity scenarios. `--all` runs evaluations across all profiles and requires a trained model file for each profile.

#### 3. Compare against clinical baselines

```bash
python baselines.py --cancer Glioblastoma
```

This compares Standard of Care and Adaptive Therapy policies across the scenario set.

#### 4. Run the spatial-expansion study

```bash
python spatial_expansion.py --cancer Glioblastoma
```

This study trains and evaluates models using progressively larger spatial observation windows. The current implementation defines spatial stages for `Glioblastoma` and `Renal_Cell_Carcinoma` only.

### Branch II files

| File or directory | Purpose |
| --- | --- |
| `train.py` | DRQN definition, training, hyperparameter optimization, and stability runs |
| `environment.py` | Gymnasium environment for the predictive chemotherapy task |
| `simulation.py` | Digital-patient pharmacokinetic and tumor-response simulation |
| `utils.py` | Data, sensor, and simulation helpers |
| `dataset_generation.py` | Converts a raw NS-3 trace into the sampling schedule |
| `generate_simulations.py` | Starts the NS-3 trace-generation workflow |
| `evaluate.py` | AI stress tests and frozen evaluation |
| `baselines.py` | Standard-of-Care and Adaptive Therapy comparisons |
| `spatial_expansion.py` | Spatial observation-window study |
| `model_study.py` | LSTM ablation and latent-space analysis |
| `datasets/` | Sampling schedule used by training and evaluation |
| `models/` | Saved model weights |
| `config/` | Per-profile optimized training parameters |
| `results/` | Evaluation summaries, reports, and plots |

### Branch II dataset regeneration

NS-3 is not required for the standard Branch II training and evaluation commands.

To regenerate the dataset, use the Branch II versions of `generate_simulations.py` and `dataset_generation.py`. The NS-3 script assumes an external NS-3 installation at `~/ns-allinone-3.36.1/ns-3.36.1` and raw trace files under `~/EXPERIMENTS/Sim_01_Toy_Tests/raw_ns3_traces`. Review these paths and your NS-3 setup before running the regeneration workflow.

### Branch II outputs

Scripts create or update files under `models/`, `config/`, and `results/`. Examples include:

- `models/<Cancer>_drqn_best.pt`
- `config/<Cancer>_best_params.json`
- `results/<Cancer>_stress_report.txt`
- `results/<Cancer>_stress_dashboard.png`
- `results/<Cancer>_clinical_baselines_summary.csv`
- `results/spatial_expansion/<Cancer>_stage_<N>_eval.csv`

## General notes

- Run commands from the repository root because scripts use relative input and output paths.
- The two branches contain different implementations and dataset requirements; follow the workflow for the branch currently checked out.
- On Branch II, `evaluate.py --all` expects a trained model for every cancer profile.