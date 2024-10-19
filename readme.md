# Comprehensive machine learning comparisons based on the gut microbiome to assess the response of patients with non-small-cell lung cancer to immunotherapy

## Authors

- **Ziwei Yu**
- **Yang Dong**
- **Jinhuan Liu**
- **Xiao Wu** *(Corresponding Author)*  
  Email: [Wuxiao990222@163.com](mailto:Wuxiao990222@163.com)

## Affiliation

1. Department of Respiratory, Qingdao Central Hospital, University of Health and Rehabilitation Sciences, Qingdao, China, 266000

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Data Availability](#data-availability)
- [Contact](#contact)

## Introduction

This project focuses on assessing the performance of various machine learning algorithms in predicting the clinical response of non-small-cell lung cancer (NSCLC) patients to immunotherapy based on their gut microbiome composition. The study evaluates eight different models to determine which algorithms provide the most accurate predictions.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Ensure that your feature data and metadata are in CSV format and placed in the project directory. Update the file paths in the `main.py` script if necessary.

- **Feature Data**: `features_genus.csv`, `features_species.csv`
- **Metadata**: `metadata.csv`

The `metadata.csv` file should include a `SampleID` column and a `Clinical_Response` column containing labels such as 'PR', 'PD', and 'SD'.

### Running the Script

```bash
python main.py
```

This command will execute the training and evaluation of the specified machine learning models on the provided datasets. The results, including ROC curves and confusion matrices, will be saved in the designated output directories.


## Results

After running the script, the following outputs will be generated for each model and dataset:

- **ROC Curves**: Saved as `.png` images in the respective results directories.
- **ROC Data**: Saved as `.xlsx` files containing False Positive Rate (FPR) and True Positive Rate (TPR) data.
- **Confusion Matrices**: Printed to the console and can be saved or analyzed further as needed.

## Data Availability

The shotgun metagenomic sequences used in this study can be accessed through the National Center for Biotechnology Information (NCBI) under the accession number **PRJNA751792**. All machine learning code is available in this repository.

## Contact

For any questions or further information, please contact **Xiao Wu** at [Wuxiao990222@163.com](mailto:Wuxiao990222@163.com).
