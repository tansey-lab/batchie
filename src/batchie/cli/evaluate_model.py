import argparse
import logging

import numpy as np

from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie.common import EXPERIMENT_SPACE
from batchie.core import BayesianModel, ThetaHolder
from batchie.data import Screen, ExperimentSpace
from batchie.models.main import predict_viability_all, ModelEvaluation

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for evaluating model performance by predicting over an observed 'test screen'."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--screen",
        help="A batchie Screen in hdf5 format with all plates observed.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--thetas",
        help="A batchie ThetaHolder in hdf5 format.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--output",
        help="Output ModelEvaluation object in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        help="Seed to use for PRNG.",
        type=int,
        default=0,
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    screen = Screen.load_h5(args.screen)

    theta_holder: ThetaHolder = ThetaHolder(n_thetas=1)

    theta_holders = [theta_holder.load_h5(x) for x in args.thetas]

    thetas = theta_holder.concat(theta_holders)

    chain_ids = []
    for i, t in enumerate(theta_holders):
        chain_ids.extend([i] * t.n_thetas)

    chain_ids = np.array(chain_ids, dtype=int)

    predictions = predict_viability_all(
        screen=screen,
        thetas=thetas,
    ).T

    me = ModelEvaluation(
        observations=screen.observations,
        predictions=predictions,
        chain_ids=chain_ids,
        sample_names=screen.sample_names,
    )

    logger.info(f"Saving ModelEvaluation to {args.output}")

    me.save_h5(args.output)
