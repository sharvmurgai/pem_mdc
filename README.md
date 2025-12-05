# PEM-UDE Epidemiological Modeling & Identifiability Analysis

This repository implements a **Scientific Machine Learning (SciML)** workflow to model epidemiological dynamics using **Universal Differential Equations (UDEs)**. It uses a **Prediction Error Method (PEM)** to train the model and performs a **Minimally Disruptive Curve (MDC)** analysis to quantify the structural identifiability (robustness) of the learned parameters.

## Project Structure

* **`pem_mdc_train.jl`** (The Training Engine)
    * Defines the `SEInsIsIaDR` compartmental model.
    * Uses a Neural Network to learn the time-varying transmission rate $\kappa(t)$.
    * Trains an ensemble of models using the Prediction Error Method (PEM).
    * **Output:** Creates a timestamped results directory (e.g., `pem_mdc_train_2024-12-04_...`).
* **`pem_mdc_analyze.jl`** (The Robustness Tester)
    * A post-processing CLI tool.
    * Loads a trained model checkpoint.
    * Performs **Minimally Disruptive Curve (MDC)** analysis to test if the learned $\kappa(t)$ is unique or if alternative shapes exist within a specific error budget.
* **`utils_1NN.jl`** *(Dependency)*
    * Contains helper functions for data loading and processing.
* **`avg_output.dat`** *(Data)*
    * The input time-series data for the model.

---

## Quick Start

1. Installation
Ensure you have Julia installed. Instantiate the environment to install required packages (`DifferentialEquations`, `Lux`, `Optimization`, `SciMLSensitivity`, etc.):

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'

2. Step 1: Train the Model

Run the training script to fit the Neural ODE to your data. This script will run an ensemble of trainings (default N_RUNS=1 for testing, adjust in file for production).
Bash

julia --project pem_mdc_train.jl

What happens:

    A new folder is created: pem_mdc_train_YYYY-MM-DD_HHMMSS.

    The model trains using ADAM (coarse) and LBFGS (fine) optimizers.

    Checkpoints and prediction plots are saved to this folder.

3. Step 2: Analyze Robustness (MDC)

Once training is complete, use the analysis script to stress-test the results. You must provide the path to the output folder generated in Step 1.

Usage:
Bash

julia --project pem_mdc_analyze.jl <OUTROOT> <DELTA_FRAC> <SEED>

Example: To test seed 1 with a 1% error budget (finding a curve that is 1% worse than optimal but maximally different):
Bash

julia --project pem_mdc_analyze.jl pem_mdc_train_2025-12-04_103000 0.01 1

Output:

    Checks run_1 inside the folder.

    Generates mdcurve_kappa_overlay.png: A plot comparing the original learned curve vs. the "disruptive" alternative.

    Saves mdcurve_summary.csv: Metrics showing the "distance" between the curves.
