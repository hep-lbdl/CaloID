""" 
file: trainer.py
description: Train a series of caloclf models from a config spec.
"""

import json
import logging
import os
import random
import sys
from itertools import product

from sklearn.model_selection import ParameterGrid

from loaders import load_all_data
from models import (
    train_caloclf_model,
    build_lagan_style_model,
    build_shower_shape_model,
    build_densenet_model,
    build_raveled_model,
)

if __name__ == "__main__":
    logger = logging.getLogger(
        "%s.%s"
        % (__package__, os.path.splitext(os.path.split(__file__)[-1])[0])
    )

    logging.basicConfig(level=logging.DEBUG)

    cfg = json.load(open(sys.argv[1]))

    # {
    #     "basedir": "./data/",
    #     "ending": "_angle_position_5deg_xy.h5",
    #     "jobs": [{
    #         "data_prefix": "data_",
    #         "model": {
    #             "lcn": true,
    #             "bn": true,
    #             "dropout_rate": 0.0
    #         },
    #         "training": {
    #             "class_one": "gamma",
    #             "class_two": "eplus",
    #             "adam_lr": 0.0001,
    #             "batch_size": 512
    #         }
    #     }]
    # }

    def safe_to_list(v):
        if isinstance(v, list):
            return v
        return [v]

    BASEDIR = cfg["data"]["basedir"]
    ENDING = cfg["data"]["ending"]

    logger.info(
        "Reading from basedir = {}, ending = {}".format(BASEDIR, ENDING)
    )

    data = {}

    for job_id, job in enumerate(cfg["jobs"]):

        logger.info("loading job {}/{}".format(job_id + 1, len(cfg["jobs"])))

        training_hparams = {
            k: safe_to_list(v) for k, v in job["training"].items()
        }
        model_hparams = {k: safe_to_list(v) for k, v in job["model"].items()}
        model_fn = eval(job["model_fn"])

        # these should be single lists
        CLASS_ONE = training_hparams["class_one"]
        CLASS_TWO = training_hparams["class_two"]

        assert len(CLASS_ONE) == 1
        assert len(CLASS_TWO) == 1

        CLASS_ONE = CLASS_ONE[0]
        CLASS_TWO = CLASS_TWO[0]

        # number of random hyperparameter combinations to choose from
        nb_selections = job["random_selection"]
        nb_proc = job["proc"]

        logger.info("training {} vs. {}".format(CLASS_ONE, CLASS_TWO))
        logger.info("checking for prebuilt data...")

        key = CLASS_ONE + CLASS_TWO
        if key not in data:
            logger.info("data not pre-built, making data...")

            data[key] = load_all_data(
                basedir=BASEDIR,
                class_one=CLASS_ONE,
                class_two=CLASS_TWO,
                ending=ENDING,
            )
        else:
            logger.info("data was pre-built, continuing")

        data_train = data[key][job["data_prefix"] + "train"]
        data_test = data[key][job["data_prefix"] + "test"]

        labels_train = data[key]["labels_train"]
        labels_test = data[key]["labels_test"]

        t_hparam_grid = ParameterGrid(training_hparams)
        m_hparam_grid = ParameterGrid(model_hparams)

        logger.info("selecting {} combinations".format(nb_selections))

        candidates = list(product(list(t_hparam_grid), list(m_hparam_grid)))

        hparam_grid = random.sample(population=candidates, k=nb_selections)

        # if nb_proc > 1:
        for count, (t_hparams, m_hparams) in enumerate(hparam_grid):
            logger.info(
                "performing grid element {} of {}".format(
                    count + 1, nb_selections
                )
            )

            train_caloclf_model(
                model_fn=model_fn,
                data_train=data_train,
                labels_train=labels_train,
                data_test=data_test,
                labels_test=labels_test,
                model_hparams=m_hparams,
                training_hparams=t_hparams,
            )
