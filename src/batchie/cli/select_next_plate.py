import argparse
import json
import logging

from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import (
    KVAppendAction,
    cast_dict_to_type,
    get_prng_from_seed_argument,
)
from batchie.common import SELECTED_PLATES_KEY
from batchie.core import (
    PlatePolicy,
)
from batchie.data import Screen
from batchie.scoring.main import ChunkedScoresHolder
from batchie.scoring.main import select_next_plate

logger = logging.getLogger(__name__)


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
        "--scores",
        help="One or more ChunkedScoresHolder objects in hdf5 format.",
        type=str,
        required=True,
        nargs="+",
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

    policy = None
    if args.policy is not None:
        policy = args.policy_cls(**args.policy_params)

    rng = get_prng_from_seed_argument(args)

    scores = ChunkedScoresHolder.concat(
        [ChunkedScoresHolder.load_h5(x) for x in args.scores]
    )

    logger.info(
        f"Loaded {len(scores.scores)} scores files, {scores.current_index} scores total"
    )

    next_plate = select_next_plate(
        screen=screen,
        scores=scores,
        policy=policy,
        rng=rng,
    )

    result_object = {
        SELECTED_PLATES_KEY: [int(next_plate.plate_id)],
    }

    with open(args.output, "w") as f:
        json.dump(result_object, f, indent=4)
