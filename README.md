<img src="https://journals.aps.org/prd/article/10.1103/PhysRevD.97.014021/figures/2/medium" width=42% align="right" />

# Calorimeter Shower ID with Deep learning

### Steps

1. Download the data from [here](https://data.mendeley.com/datasets/pvn3xc3wy5/1). Save the individual files to a directory, say, `/path/to/data` (you now should have `/path/to/data/{gammma, eplus, piplus}.hdf5`).
2. Edit the configuration file `config.json` (or make a copy) to point to this directory, and edit the config to point to a location where you want model metadata & logging to occur (say, `/path/to/save/things`).
3. From the directory, run `python trainer.py config.json`, and profit!

### Requirements

* Keras==2.2.0
* Keras-contrib==2.0.8
* Pandas
* Numpy
* Scikit learn
* h5py

