import argparse
import logging

import numpy as np

from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie.common import N_UNIQUE_SAMPLES, N_UNIQUE_TREATMENTS
from batchie.core import BayesianModel, ThetaHolder
from batchie.data import Screen
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
        "--model",
        help="Fully qualified name of the BayesianModel class to use.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Output ModelEvaluation object in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Model parameters",
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

    args.model_cls = introspection.get_class(
        package_name="batchie", class_name=args.model, base_class=BayesianModel
    )

    required_args = introspection.get_required_init_args_with_annotations(
        args.model_cls
    )

    if not args.model_param:
        args.model_params = {}
    else:
        args.model_params = cast_dict_to_type(args.model_param, required_args)

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    screen = Screen.load_h5(args.screen)

    args.model_params[N_UNIQUE_SAMPLES] = screen.sample_space_size
    args.model_params[N_UNIQUE_TREATMENTS] = screen.treatment_space_size

    model: BayesianModel = args.model_cls(**args.model_params)

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
