import argparse
from itertools import combinations
import json
import numpy as np
from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import (
    KVAppendAction,
    cast_dict_to_type,
    get_prng_from_seed_argument,
)
from batchie.core import (
    BayesianModel,
    ThetaHolder,
    Scorer,
    PlatePolicy,
)
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.data import Screen
from batchie.scoring.main import select_next_batch
from batchie.common import N_UNIQUE_SAMPLES, N_UNIQUE_TREATMENTS, SELECTED_PLATES_KEY


def get_parser():
    parser = argparse.ArgumentParser(description="calculate_distance_matrix.py")
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--data",
        help="A batchie Screen object in hdf5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--thetas",
        help="A batchie ThetaHolder object in hdf5 format.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--distance-matrix",
        help="A batchie ChunkedDistanceMatrix object in hdf5 format.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--batch-size",
        help="Number of plates to select in this batch.",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--model",
        help="Fully qualified name of the model class to use.",
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
        "--scorer",
        help="Fully qualified name of the scorer class to use.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scorer-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Scorer parameters",
    )

    parser.add_argument(
        "--policy",
        help="Fully qualified name of the PlatePolicy class to use.",
        type=str,
    )
    parser.add_argument(
        "--policy-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Policy parameters",
    )

    parser.add_argument(
        "--output",
        help="Location of output json file which will contain the plate ids selected.",
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

    args.scorer_cls = introspection.get_class(
        package_name="batchie", class_name=args.scorer, base_class=Scorer
    )

    required_args = introspection.get_required_init_args_with_annotations(
        args.scorer_cls
    )

    if not args.scorer_param:
        args.scorer_params = {}
    else:
        args.scorer_params = cast_dict_to_type(args.scorer_param, required_args)

    if args.policy is not None:
        args.policy_cls = introspection.get_class(
            package_name="batchie", class_name=args.policy, base_class=PlatePolicy
        )

        required_args = introspection.get_required_init_args_with_annotations(
            args.policy_cls
        )

        if not args.policy_param:
            args.policy_params = {}
        else:
            args.policy_params = cast_dict_to_type(args.policy_param, required_args)
    else:
        args.policy_cls = None
        args.policy_params = {}

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    screen = Screen.load_h5(args.data)

    args.model_params[N_UNIQUE_SAMPLES] = screen.n_unique_samples
    args.model_params[N_UNIQUE_TREATMENTS] = screen.n_unique_treatments

    model: BayesianModel = args.model_cls(**args.model_params)
    scorer: Scorer = args.scorer_cls(**args.scorer_params)
    theta_holder: ThetaHolder = model.get_results_holder(n_samples=1)
    thetas = theta_holder.concat([theta_holder.load_h5(x) for x in args.thetas])

    distance_matrix = ChunkedDistanceMatrix.concat(
        [ChunkedDistanceMatrix.load(x) for x in args.distance_matrix]
    )
    policy = None
    if args.policy is not None:
        policy = args.policy_cls(**args.policy_params)

    rng = get_prng_from_seed_argument(args)

    next_batch = select_next_batch(
        model=model,
        scorer=scorer,
        screen=screen,
        distance_matrix=distance_matrix,
        policy=policy,
        samples=thetas,
        batch_size=args.batch_size,
        rng=rng,
        progress_bar=args.progress,
    )

    result_object = {
        SELECTED_PLATES_KEY: [int(x.plate_id) for x in next_batch],
    }

    with open(args.output, "w") as f:
        json.dump(result_object, f, indent=4)
