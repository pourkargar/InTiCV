# An Integrated Time Series Modeling and Computer Vision Framework for Predictive Structure Characterization of Extruded Plant-based Meat Products
InTiCV: (In)tegrated (Ti)me Series Modeling and (C)omputer (V)ision

## Overview

This repository contains a comprehensive multimodal machine learning framework for modeling and predicting food extrusion processes and product quality assessment. The project integrates time-series process data with computer vision-based texture analysis to predict critical process parameters and final product quality scores.

## Project Structure

The codebase is organized into three main components:

### PART 1: RNN Extruder Models
Deep learning models for time-series prediction of extrusion process parameters using recurrent neural networks.

- **LSTM_Script.ipynb** / **GRU_Script.ipynb**: Training scripts for LSTM and GRU models to predict die temperature and specific mechanical energy (SME)
- **TimeseriesSequence.py**: Custom data generator for time-series sequence batching
- **utils.py**: Evaluation metrics (NRMSE, MAE, MB) and visualization functions
- **Evals_&_plots.ipynb**: Model evaluation and performance visualization
- **saved_models/**: Pre-trained LSTM and GRU models
- **simulations/**: Model prediction outputs
- **training_hists/**: Training history logs

**Key Features:**
- Predicts die temperature and SME from process variables (RPM, water feed, temperatures, etc.)
- Time-series data split by experimental groups
- Early stopping and model checkpointing for optimal performance

### PART 2: Image Scoring
Computer vision-based texture quality assessment using unsupervised and semi-supervised learning approaches.

- **Unsupervised_CV_scoring.ipynb**: Image texture scoring using edge detection and Gabor features without labeled data
- **Semi_Supervised_CV_scoring.ipynb**: Supervised refinement using XGBoost with texture and edge features
- **Multi_Modal_Data_gen.ipynb**: Integration of process data with image-based scores
- **utils.py**: Image processing functions (edge extraction, shape detection, Gabor filtering)
- **Evals_&_Plots.ipynb**: Performance evaluation comparing scoring methods
- **data/Image/**: Product images for time-based and final product analysis
- **data/Scores/**: Generated quality scores from various methods

**Key Features:**
- Automated texture quality scoring (porous vs. fibrous)
- Multi-scale edge detection and texture analysis
- Binary classification and regression-based quality metrics
- Integration of process parameters with visual features

### PART 3: Multimodal
Ensemble learning models combining time-series process data and image-based quality scores for comprehensive prediction.

- **Multimodal_Training.ipynb**: Training XGBoost and LightGBM models with multimodal data
- **utils.py**: Time-series padding and evaluation utilities
- **Evals_&_Plots.ipynb**: Performance visualization and comparison
- **data/multimodal_data.csv**: Integrated dataset with process parameters and image scores
- **Simulations/**: Multimodal model predictions

**Key Features:**
- Forecasts die temperature, SME, and texture scores simultaneously
- Uses Darts library for time-series forecasting with covariates
- Feed composition encoding for different experimental formulations
- Ensemble models for robust predictions across multiple outputs

## Environment and Dependencies (uv)

This repository now uses a `uv`-managed Python environment via `pyproject.toml`.

### 1. Install uv

Use one of the official methods:

```bash
pip install uv
```

or

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Sync the project environment

From the repository root:

```bash
uv sync --dev
```

This creates a local virtual environment (`.venv`) and installs all pinned runtime dependencies plus notebook tooling.


### Dependency source of truth

- Runtime dependencies are defined in `pyproject.toml` under `[project.dependencies]`.
- Notebook/dev tools are in `pyproject.toml` under `[dependency-groups].dev`.
- Python version is tracked in `.python-version`.


## Usage

### 1. Train RNN Models
Navigate to `PART1 - RNN Extruder Models/` and run:
- `LSTM_Script.ipynb` for LSTM model training
- `GRU_Script.ipynb` for GRU model training
- Use `Evals_&_plots.ipynb` for model evaluation

### 2. Generate Image Scores
Navigate to `PART2- Image Scoring/` and run:
1. `Unsupervised_CV_scoring.ipynb` for initial computer vision-based scoring
2. `Semi_Supervised_CV_scoring.ipynb` for refined scoring with supervised learning
3. `Multi_Modal_Data_gen.ipynb` to create the multimodal dataset
4. `Evals_&_Plots.ipynb` for evaluation

### 3. Train Multimodal Models
Navigate to `PART3 - Multimodal/` and run:
- `Multimodal_Training.ipynb` for training ensemble models
- `Evals_&_Plots.ipynb` for visualization and comparison

## Data

- **Extrusion Process Data**: Time-series measurements of extruder parameters (RPM, temperature zones, water feed, bulk density, etc.)
- **Image Data**: Product images captured during and after extrusion for texture analysis
- **Quality Scores**: Expert-labeled and computer vision-generated texture quality scores

## Models

- **LSTM/GRU**: Sequence-to-sequence models for multi-step process parameter forecasting
- **XGBoost**: Gradient boosting for image feature-based quality prediction
- **LightGBM/CatBoost**: Ensemble methods for multimodal time-series forecasting with covariates
- **Computer Vision**: Edge detection (Canny), shape recognition (Hough transforms), and Gabor texture filters

## System Specifications

The notebooks in this repository were developed and tested on the following system configuration:

### Hardware
- **Operating System**: Windows 11 Pro 
- **Processor**: Dual Intel Xeon Gold 5220R @ 2.20GHz
  - Total Cores: 48 physical cores (96 logical processors)
  - Architecture: x64-based PC
- **RAM**: 128 GB (130,693 MB) Total Physical Memory
- **GPU**: 3x NVIDIA RTX A4500

### Software Environment
- **Python**: 3.12.6
- **CUDA**: Compatible with TensorFlow 2.18.0 
- **Development Environment**: Jupyter Lab 4.2.5

**Note**: The code should run on systems with lower specifications, though training times may vary. GPU acceleration is recommended for deep learning model training (PART 1) but not strictly required. 


## Publication

This code is associated with the following publication:


```
[Author Last Name], [Initials]. ([Year]). [Title of the paper]. [Journal Name], [Volume]([Issue]), [Page range]. https://doi.org/[DOI]
```

---

## Contact

For questions or collaboration inquiries, please contact dbpourkargar@ksu.edu.
