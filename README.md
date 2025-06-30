# Estimating Stressors in Driving

## Installation

We recommend using a conda environment with ``Python >= 3.12`` :

```bash
conda create -n [env-name] python=3.12
conda activate [env-name]
```

Clone the repository and install the dependencies:

```bash
git clone https://github.com/usc-sail/driving-stressors
pip install -r requirements.txt
```

## Input Data Structure

The data to train the model are loaded through ``process_trina33.py`` and pre-processed through ``features_new.py``.

## Running the models

The machine learning pipeline can be run with:

```bash
python trainer.py
```

Most parameters can be tuned through the ``config.yaml`` file.

Model evaluation and statistics for a specific experimental setup through:

```bash
python get_scores.py
python evaluation.py
```
