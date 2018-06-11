""" 
file: models.py
description: Definition of clf models for densenet, etc.
"""
import json
import logging
import os

import numpy as np

from sklearn.metrics import roc_curve
from keras.layers.merge import concatenate, multiply
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import (Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization,
                          LocallyConnected2D, Activation, ZeroPadding2D,
                          Dropout, Lambda, Flatten, Input, add)

from keras_contrib.applications.densenet import DenseNet as build_densenet


logger = logging.getLogger(__name__)


def assign_identifier():
    import uuid
    return str(uuid.uuid4())


def build_densenet_model(data):

    shapes = [d.shape[1:] for d in data]

    x = [Input(shape=sh) for sh in shapes]

    dnet_layer0 = build_densenet(weights=None, input_shape=(3, 96, 1),
                                 nb_dense_block=1, include_top=False)
    dnet_layer1 = build_densenet(weights=None, input_shape=(12, 12, 1),
                                 nb_dense_block=1, include_top=False)
    dnet_layer2 = build_densenet(weights=None, input_shape=(12, 6, 1),
                                 nb_dense_block=1, include_top=False)

    dnet_merged = [dnet_layer0, dnet_layer1, dnet_layer2]

    features = [f(xi) for f, xi in zip(dnet_merged, x)]

    y = Dense(1, activation='sigmoid')(Dense(64, activation='relu')(
        concatenate(features)
    ))

    return Model(x, y)


def build_shower_shape_model(data, bn=True, dropout_rate=0.0, skip=False):

    apply_bn = lambda x: BatchNormalization()(x) if bn else lambda x: x
    x = Input(shape=(data.shape[1], ))

    h = Dense(512)(x)

    if skip:
        h_skip = h

    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(1024)(h)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(2048)(h)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(1024)(h)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(128)(h)
    if skip:

        h = concatenate([h, h_skip])

    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = Dense(1)(h)
    y = Activation('sigmoid')(h)

    feature_dnn = Model(x, y)

    return feature_dnn


def build_raveled_model(data, bn=True, dropout_rate=0.0):

    apply_bn = lambda x: BatchNormalization()(x) if bn else lambda x: x

    x = Input(shape=(data.shape[1], ))

    h = Dense(512)(x)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(1024)(h)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(2048)(h)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(1024)(h)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = apply_bn(h)

    h = Dense(128)(h)
    h = Dropout(dropout_rate)(LeakyReLU()(h))

    h = Dense(1)(h)
    y = Activation('sigmoid')(h)

    raveled_dnn = Model(x, y)
    return raveled_dnn


def build_lagan_style_model(data, lcn=True, bn=True, dropout_rate=0.0):
    """
    Takes configuration information about the input (shapes) and builds a model
    and returns it
    """
    def build_model(image):
        '''
        Build LAGAN-style discriminator
        '''

        layer_op = LocallyConnected2D if lcn else Conv2D

        apply_bn = lambda x: BatchNormalization()(x) if bn else lambda x: x

        x = Conv2D(64, (2, 2), padding='same')(image)

        x = apply_bn(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = ZeroPadding2D((1, 1))(image)
        x = layer_op(8 * 4, (3, 3), padding='valid', strides=(1, 2))(x)

        x = apply_bn(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = ZeroPadding2D((1, 1))(x)
        x = layer_op(16 * 4, (2, 2), padding='valid')(x)

        x = apply_bn(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = ZeroPadding2D((1, 1))(x)
        x = layer_op(32 * 4, (2, 2), padding='valid', strides=(1, 2))(x)

        x = apply_bn(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Flatten()(x)

        return x

    shapes = [d.shape[1:] for d in data]
    logger.info('found shapes for tensors: {}'.format(shapes))

    x = [Input(shape=sh) for sh in shapes]

    h = concatenate(map(build_model, x))

    h = Dense(256)(h)
    h = Activation('relu')(h)
    h = Dropout(dropout_rate)(h)

    y = Dense(1, activation='sigmoid')(h)

    return Model(x, y)


def train_caloclf_model(model_fn, data_train, labels_train, data_test,
                        labels_test, model_hparams, training_hparams):

    logger = logging.getLogger(__name__)

    assert 'class_one' in training_hparams
    assert 'class_two' in training_hparams
    assert 'adam_lr' in training_hparams
    assert 'batch_size' in training_hparams
    assert 'basedir' in training_hparams

    fn_name = model_fn.__name__

    meta = {}

    basedir = training_hparams['basedir']

    identifier = assign_identifier()
    logger.info('assigned identifier = {}'.format(identifier))

    identifier = '{}-{}'.format(fn_name, identifier)

    meta_file = os.path.join(basedir, '{}-meta.json'.format(identifier))
    logger.info('will write experiment tracking to {}'.format(meta_file))

    chkpt = os.path.join(basedir, '{}-chkpt.h5'.format(identifier))
    logger.info('will write model checkpoints to {}'.format(chkpt))

    final = os.path.join(basedir, '{}-final.h5'.format(identifier))
    logger.info('will write final model weights to {}'.format(final))

    yhat_file = os.path.join(basedir, '{}-predictions.h5'.format(identifier))
    logger.info('will write final predictions to {}'.format(yhat_file))

    meta.update({
        'chkpt_file': chkpt,
        'final_file': final,
        'yhat_file': yhat_file,
        'class_one': training_hparams['class_one'],
        'class_two': training_hparams['class_two']
    })

    image_dnn = model_fn(data_train, **model_hparams)

    meta.update({
        'model_fn': chkpt,
        'model_hparams': model_hparams,
        'training_hparams': training_hparams
    })

    image_dnn.compile(
        Adam(lr=training_hparams['adam_lr']),
        'binary_crossentropy',
        metrics=['acc']
    )

    callbacks = [
        EarlyStopping(verbose=True, patience=12, monitor='val_loss'),
        ModelCheckpoint(chkpt, monitor='val_loss', verbose=True,
                        save_best_only=True),
    ]

    try:
        image_dnn.fit(
            data_train, labels_train,
            callbacks=callbacks,
            verbose=True,
            validation_split=0.3,
            batch_size=training_hparams['batch_size'],
            epochs=100
        )

    except KeyboardInterrupt:
        logger.warning('ending early')

    image_dnn.load_weights(chkpt)
    image_dnn.save_weights(final)
    image_dnn.load_weights(final)

    yhat_image_dnn = image_dnn.predict(data_test, verbose=True,
                                       batch_size=512).ravel()

    accuracy = np.mean((yhat_image_dnn > 0.5) == labels_test)

    fpr, tpr, _ = roc_curve(
        labels_test,
        abs(1 - yhat_image_dnn),
        pos_label=0
    )

    rej = 1 / fpr

    working_points = [0.60, 0.70, 0.80, 0.90, 0.96, 0.97, 0.98, 0.99, 0.9999]

    all_ops = {wp: rej[np.argmin(abs(tpr - wp))] for wp in working_points}

    meta.update({
        'metrics': {
            'acc': accuracy,
            'operating_points': all_ops
        }
    })

    np.save(yhat_file, yhat_image_dnn)

    logger.info('writing to meta location = {}'.format(meta_file))

    with open(meta_file, 'w') as fp:
        json.dump(meta, fp, indent=4, sort_keys=True)
