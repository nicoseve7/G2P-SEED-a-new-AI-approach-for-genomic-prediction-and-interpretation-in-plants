# G2P-SEED: a new AI approach for genomic prediction and interpretation in plants
Source code developed for my MSc thesis on genomic prediction and model
interpretation in apple (*Malus domestica*).

The project investigates three agronomic traits:

- Harvest date
- Acidity
- Red overcolor

The workflow combines genomic, environmental, and phenotypic information
using machine learning and neural network models, together with SHAP-based
interpretation and Genetic Algorithm-based genomic feature selection.

The main objective is not only to predict phenotypic traits, but also to
identify genomic regions and SNPs that contribute consistently to model
predictions across multiple data splits.

## Main workflow
The repository contains three main analysis components:

1. Common genomic preprocessing
2. Harvest date analysis
3. Acidity and red overcolor analysis

The overall workflow includes:

- phenotype quality control and preprocessing;
- genotype preprocessing and genomic principal component analysis;
- weather and soil preprocessing;
- Gradient Boosting-based SNP preselection;
- multimodal neural network modelling;
- SHAP-based model interpretation;
- genomic-region ranking;
- Genetic Algorithm-based SNP and region selection;
- biological annotation of selected genomic regions.

## Reference deep-learning model
A multimodal neural network architecture proposed by Jung et al. (2025)
is used as a reference model for the Harvest date analysis.

The original neural network architecture provided by Jung et al. is retained,
including separate branches for weather, genomic principal components,
SNP markers, and soil information.

The implementation is adapted to the dataset and preprocessing pipeline
used in this thesis. In particular:

- 20 genomic principal components are used instead of the 58 PCs employed
  in the original study;
- SNPs are selected independently within each cross-validation split using
  Gradient Boosting;
- the selected SNPs therefore have comparable dimensionality to the original
  implementation but are specific to the dataset analysed here;
- weather inputs are reconstructed from the environments available in this
  study using 300 daily observations and three meteorological variables
  (temperature, humidity, and radiation);
- soil information is included only for the Jung et al. reference model;
- model performance is evaluated using the cross-validation strategy adopted
  in this thesis.

The reference architecture is implemented in:

`02_harvest_date/06_deep_learning_baseline/`

The neural network developed in this thesis is implemented separately in:

`02_harvest_date/07_neural_network/`

## Repository structure
```text
apple-genomic-prediction/
│
├── data/
│   └── raw/
│
├── 01_common_genomic_preprocessing/
│
├── 02_harvest_date/
│   ├── 01_data_audit/
│   ├── 02_phenotype_preprocessing/
│   ├── 03_environment_preprocessing/
│   ├── 04_input_preparation/
│   ├── 05_gradient_boosting/
│   ├── 06_deep_learning_baseline/
│   ├── 07_neural_network/
│   ├── 08_shap/
│   └── 09_genetic_algorithm/
│
├── 03_acidity_color_over/
│   ├── 01_data_audit/
│   ├── 02_phenotype_preprocessing/
│   ├── 03_gradient_boosting/
│   ├── 04_neural_network/
│   ├── 05_shap/
│   └── 06_genetic_algorithm/
│
├── run_preprocessing.py
├── run_harvest.py
├── run_new_traits.py
├── requirements.txt
└── README.md
