# CGL

Collaborative Graph Learning with Auxiliary Text for Temporal Event Prediction in Healthcare

# Requirements

## Packages

- python 3
- numpy
- sklearn
- pandas
- nltk
- tensorflow

## Prepare the environment

1. Install all required softwares and packages.

```bash
pip install -r requirements.txt
```

2. In the python console, download the `stopwords` and `punkt` corpus required by `nltk`.

```python
import nltk


nltk.download('stopwords')
nltk.download('punkt')
```

## Download the MIMIC-III dataset

Go to [https://mimic.physionet.org/gettingstarted/access/](https://mimic.physionet.org/) for access. Once you have the authority for the dataset, download the dataset and extract the csv files to `data/mimic3/raw/` in this project.

# Preprocess the dataset

```bash
python run_preprocess.py
```

# Train the model

1. For the medical code prediction task.

```bash
python train_codes.py
```

2. For the heart failure prediction task.

```bash
python train_hf.py
```