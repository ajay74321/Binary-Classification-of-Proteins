## Binary-Classification-of-Proteins
## Introduction
This project focuses on developing machine learning models to classify protein sequences into positive (+1) or negative (-1) categories. Protein sequences, represented as chains of amino acids, carry vital biological information that can be leveraged for predictive tasks in computational biology. Unlike structured numerical data, protein sequences are inherently symbolic and require specialized feature engineering techniques such as amino acid composition, dipeptide composition, physico-chemical properties, and residue repeat information. These features capture both the chemical nature and sequential patterns of proteins, enabling the construction of robust classifiers.
### 📊 File Description

The dataset provided for this project consists of three CSV files:

#### 1. `train.csv`
- **Columns**:  
  - `label` → Target variable, where `+1` denotes a positive protein sequence and `-1` denotes a negative protein sequence.  
  - `Sequence` → The amino acid sequence of the protein.  
- **Purpose**: Used to train and validate the classification models.  

#### 2. `test.csv`
- **Columns**:  
  - `ID` → Unique identifier for each protein sequence.  
  - `Sequence` → The amino acid sequence of the protein.  
- **Purpose**: Used for generating predictions. The labels for these sequences are unknown and must be predicted by the model.  

#### 3. `sample.csv`
- **Columns**:  
  - `ID` → Unique identifier (copied from test set).  
  - `label` → Predicted class label, either `+1` (positive) or `-1` (negative).  
- **Purpose**: Defines the required submission format for the competition.  

The sequences are composed of **standard amino acids** represented by single-letter codes:  
`A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V`.  

Any rare or unknown characters are standardized during preprocessing.  
### ⚙️ Requirements  

- `pandas`  
- `numpy`  
- `scikit-learn`  
- `autogluon`  

You can install all dependencies using:  

```bash
pip install -r requirements.txt
```
## Data 
The training and test data used in this project are provided in the dataset folder.

## Note
This project was part of a kaggle competition.



