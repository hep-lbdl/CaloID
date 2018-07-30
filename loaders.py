
import logging
from functools import partial
import h5py
import numpy as np

from calodata.features import extract_features, extract_dataframe

import os

from functools import partial


concat = partial(np.concatenate, axis=0)


def load_calodata(fpaths):
    '''
    Returns:
    --------
        data: a list of 3 numpy arrays, representing the energy deposition in
            each layer for a group of showers contained in the file 'fpath'
    '''
    for fpath in fpaths:
        with h5py.File(fpath, 'r') as h5:
            try:
                data = [concat((data[i], h5['layer_{}'.format(i)][:]))
                        for i in range(3)]
            except NameError:
                data = [h5['layer_{}'.format(i)][:] for i in range(3)]
    return data


def load_all_data(basedir, class_one='piplus', class_two='eplus',
                  ending='_angle_position_5deg_xy.h5'):

    # CLASS_ONE = 'gamma'
    # CLASS_TWO = 'eplus'

    # CLASS_ONE = 'piplus'
    # CLASS_TWO = 'eplus'

    import glob
    logger = logging.getLogger(__name__)

    c1path = glob.glob(
        os.path.join(os.path.abspath(basedir),
                     '{}{}'.format(class_one, ending))
    )
    logger.info('Extracting data for {} from {}'.format(class_one, c1path))
    c1 = load_calodata(c1path)

    c2path = glob.glob(
        os.path.join(os.path.abspath(basedir),
                     '{}{}'.format(class_two, ending))
    )
    logger.info('Extracting data for {} from {}'.format(class_two, c2path))
    c2 = load_calodata(c2path)

    data = list(map(concat, zip(c1, c2)))

    labels = np.array([1] * c1[0].shape[0] + [0] * c2[0].shape[0])

    logger.info('Number of {} events = {}'.format(class_one, c1[0].shape[0]))
    logger.info('Number of {} events = {}'.format(class_two, c2[0].shape[0]))

    features = extract_features(data)  # shower shapes
    # features_df = extract_dataframe(data)
    # shower_shapes = features_df.keys()

    features = features / np.abs(features).max(axis=0)[np.newaxis, :]
    # random shuffle
    np.random.seed(0)
    ix = np.array(range(len(labels)))
    np.random.shuffle(ix)

    # number of examples to train on
    nb_train = int(0.7 * len(ix))

    # train test split
    ix_train = ix[:nb_train]
    ix_test = ix[nb_train:]

    features_train = features[ix_train]
    data_train = [np.expand_dims(d[ix_train], -1) / 1000. for d in data]
    labels_train = labels[ix_train]

    features_test = features[ix_test]
    data_test = [np.expand_dims(d[ix_test], -1) / 1000. for d in data]
    labels_test = labels[ix_test]

    raveled_train = np.concatenate([d.reshape(d.shape[0], -1)
                                    for d in data_train], axis=-1)

    raveled_test = np.concatenate([d.reshape(d.shape[0], -1)
                                   for d in data_test], axis=-1)

    return {
        'ix_train': ix_train,
        'ix_test': ix_test,
        'features_train': features_train,
        'data_train': data_train,
        'labels_train': labels_train,
        'features_test': features_test,
        'data_test': data_test,
        'labels_test': labels_test,
        'raveled_train': raveled_train,
        'raveled_test': raveled_test
    }
