<img src="https://journals.aps.org/prd/article/10.1103/PhysRevD.97.014021/figures/2/medium" width=42% align="right" />

# Calorimeter Shower ID with Deep learning

### Steps

1. Download the data from [here](https://data.mendeley.com/datasets/pvn3xc3wy5/1). Save the individual files to a directory, say, `/path/to/data` (you now should have `/path/to/data/{gammma, eplus, piplus}.hdf5`).
2. Edit the configuration file `config.json` (or make a copy) to point to this directory, and edit the config to point to a location where you want model metadata & logging to occur (say, `/path/to/save/things`).
3. From the directory, run `python trainer.py config.json`, and profit!

### Requirements

Just run `pip install -r requirements.txt` (or, `pip install -r requirements-gpu.txt` if you've got a CUDA-enabled graphics card).

_[update 10/13/20]_ This requires some old software, so please consider using a virtual environment. The specific versions of Keras and TensorFlow matter.

We recommend using Python 3. If you need to use Python 2, please downgrade the TensorFlow version to 1.15.0.

* Keras==2.0.8
* Keras-contrib (from our fork, on branch [`densenet-mod`](https://github.com/hep-lbdl/keras-contrib/tree/densenet-mod))
* Pandas
* Numpy
* Scikit learn
* h5py
* TensorFlow<=1.15.4 (make sure to install the GPU version if you can)
