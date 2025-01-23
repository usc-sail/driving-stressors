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

## Data Structure

The data for this code is located in ``erie`` server, under ``/media/data/toyota/processed_data/``.

* ``trina_33/`` contains the full csv files of the Toyota dataset, with annotations embedded.
* ``trina_33_samples/`` contains pkl dicts with segmented time-series for model input.
* ``trina_33_samples_f/`` contains pkl dicts with further pre-processed input time-series.

Any (sub)directory that is not mentioned here can be ignored.

## Running the models

The deep learning pipeline is included in the files starting with ``nn_``. Run with:

```bash
python script.py
```

Most parameters can be tuned through the ``config.yaml`` and ``nn_dataset.py`` files.
