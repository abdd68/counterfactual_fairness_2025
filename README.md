# Counterfactual Fairness 2025

This repository contains the implementation of the paper: **"Towards Counterfactual Fairness through Auxiliary Variables"**, presented at ICLR 2025.

## Overview

Balancing fairness and predictive accuracy in machine learning models, especially when sensitive attributes such as race, gender, or age are considered, has been a significant challenge. Counterfactual fairness ensures that predictions remain consistent across counterfactual variations of sensitive attributes, addressing societal biases. However, existing approaches often overlook intrinsic information about sensitive features, limiting their ability to achieve fairness while maintaining performance.

To address this, we introduce **EXOgenous Causal reasoning (EXOC)**, a novel causal reasoning framework motivated by exogenous variables. EXOC leverages auxiliary variables to uncover intrinsic properties that give rise to sensitive attributes. Our framework explicitly defines auxiliary and control nodes to promote counterfactual fairness and control information flow within the model. Evaluations on synthetic and real-world datasets demonstrate that EXOC outperforms state-of-the-art approaches in achieving counterfactual fairness without sacrificing accuracy.

## Repository Structure

- `dataset/`: Contains synthetic and real-world datasets used for evaluation.
- `baselines.py`: Implementation of baseline models for comparison.
- `main.py`: Main script to run experiments.
- `models.py`: Definition of the EXOC framework and other model architectures.
- `utils.py`: Utility functions for data processing and evaluation.

## Requirements

- Python 3.8 or higher
- Required Python packages are listed in `requirements.txt`.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/abdd68/counterfactual_fairness_2025.git
   cd counterfactual_fairness_2025
   ```

2. **Create a virtual environment**:

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the data**:

Ensure that the datasets are properly formatted and placed in the `dataset/` directory.

2. **Run experiments**:

Use the `main.py` script to train models and evaluate performance:

    python main.py --seed 56 -a 1.2 -cuda 2 --dataset law --synthetic 1 -spp 1


3. **Results**:

After training, results and metrics will be saved in the `results/` directory for analysis.

## Citation

If you find this work useful, please cite our paper:

```
@inproceedings{tian2025exoc,
  title={Towards Counterfactual Fairness through Auxiliary Variables},
  author={Bowei Tian and Ziyao Wang and Shwai He and Wanghao Ye and Guoheng Sun and Yucong Dai and Yongkai Wu and Ang Li},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}