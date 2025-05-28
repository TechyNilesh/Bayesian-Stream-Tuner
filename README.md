# Bayesian Stream Tuner (BST)

This is the code repository for "Bayesian Stream Tuner: Dynamic Hyperparameter Optimization for Real-Time Data Streams" published in KDD 2025.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TechyNilesh/Bayesian-Stream-Tuner.git
cd bayesian-stream-tuner
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Repository Structure

- `StreamTunner/`: Core implementation files
  - `BayesianStreamTuner.py`: Main BST algorithm implementation
  - `BLR.py`: Bayesian Linear Regression implementation
  - `MESSPT.py`: MESSPT algorithm implementation
  - `SSPT.py`: SSPT algorithm implementation
  - `RandomStreamSearch.py`: Random Search implementation

- `data_cls/`: Classification datasets path
- `data_reg/`: Regression datasets path

### Experiment Scripts
- `run_cls_benchmark_multi.py`: Classification experiments
- `run_reg_benchmark_multi.py`: Regression experiments
- `bst_component_analysis.py`: Component ablation study
- `bst_parameter_analysis.py`: Parameter sensitivity analysis

## Datasets

The experiments in our paper use the following datasets, available in the respective data folders:

### Classification Datasets (in `data_cls/`)
- `electricity.arff`: Real-world dataset containing 45,312 instances with 8 features for predicting electricity price changes
- `Forestcover.arff`: Dataset with 581,012 instances and 54 features for predicting forest cover types
- `new_airlines.arff`: Contains 539,383 instances with 7 features for flight delay prediction
- `nomao.arff`: Quality classification dataset with 34,465 instances and 118 features
- `SEA_Mixed_5.arff`: Synthetic data stream with mixed concept drift patterns (500,000 instances)
- `SEA_Abrubt_5.arff`: Synthetic data stream with abrupt concept drift (500,000 instances)
- `RBFm_100k.arff`: Random RBF generator with mixed drift types (100,000 instances)
- `RTG_2abrupt.arff`: Rotating hyperplane generator with gradual drift (100,000 instances)
- `sine_stream_with_drift.arff`: Synthetic data with abrupt concept drift based on sine function (50,000 instances)
- `HYPERPLANE_01.arff`: Stream classification with hyperplane rotation (500,000 instances)

### Regression Datasets (in `data_reg/`)
- `bike.arff`: Real-world dataset with 17,379 instances for bike rental prediction
- `diamonds.arff`: Price prediction dataset with 53,940 instances and price-related features
- `health_insurance.arff`: Contains 22,272 instances for predicting insurance costs
- `physiochemical_protein.arff`: 45,730 instances for protein structure prediction
- `sarcos.arff`: Robot dynamics dataset with 48,933 instances and 21 input features
- `FriedmanGra.arff`: Synthetic data with gradual concept drift (100,000 instances)
- `FriedmanGsg.arff`: Synthetic data with gradual sigmoid drift (100,000 instances)
- `FriedmanLea.arff`: Synthetic data with local evolution drift (100,000 instances)
- `fried.arff`: Combined Friedman data with mixed drift types (40,768 instances)
- `hyperA.arff`: Regression version with abrupt drift (500,000 instances)

## Reproducing Results

To reproduce the main experimental results:

1. Run classification experiments:
```bash
python run_cls_benchmark_multi.py
```

2. Run regression experiments:
```bash
python run_reg_benchmark_multi.py
```

3. Run component analysis:
```bash
python bst_component_analysis.py
```

4. Run parameter analysis:
```bash
python bst_parameter_analysis.py
```

Results will be saved in their respective directories (RESULTS_MULTI/, RESULTS_REG_MULTI/, etc.).

## Citation

If you use this code in your research, please cite our paper:

**BibTeX:**
```bibtex
@inproceedings{verma2025bayesian,
  title={Bayesian Stream Tuner: Dynamic Hyperparameter Optimization for Real-Time Data Streams},
  author={Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard and Bahri, Maroua},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages={1--12},
  year={2025},
  organization={ACM},
  address={New York, NY, USA},
  doi={10.1145/3711896.3736852},
  url={https://doi.org/10.1145/3711896.3736852}
}
```

## Note
For any questions or issues, please contact the authors or create an issue in this repository.

