import argparse
import logging
from typing import Optional

import numpy as np

from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import (
    KVAppendAction,
    cast_dict_to_type,
    get_prng_from_seed_argument,
)
from batchie.core import (
    InitialRetrospectivePlateGenerator,
    RetrospectivePlateGenerator,
    RetrospectivePlateSmoother,
)
from batchie.data import (
    Screen,
    filter_dataset_to_treatments_that_appear_in_at_least_one_combo,
)
from batchie.retrospective import reveal_plates, mask_screen, create_holdout_set

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for revealing plates in a retrospective simulation,"
        "calculating the prediction error on the un-revealed plates thus far,"
        "and saving the results."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--data", help="A batchie Screen in hdf5 format.", type=str, required=True
    )
    parser.add_argument(
        "--training-output",
        help="Output training set batchie Screen in hdf5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test-output",
        help="Output test set batchie Screen in hdf5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--initial-plate-generator",
        help="Fully qualified name of the InitialRetrospectivePlateGenerator class to use.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--initial-plate-generator-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Initial plate generator parameters",
    )
    parser.add_argument(
        "--plate-generator",
        help="Fully qualified name of the RetrospectivePlateGenerator class to use.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plate-generator-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Plate generator parameters",
    )
    parser.add_argument(
        "--plate-smoother",
        help="Fully qualified name of the RetrospectivePlateSmoother class to use.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plate-smoother-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Plate smoother parameters",
    )
    parser.add_argument(
        "--holdout-fraction",
        help="Fraction of data to holdout for testing "
        "(proportion of experiments in the test set in the test/train spit).",
        type=float,
        default=0.1,
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

    if args.plate_generator is not None:
        args.plate_generator_cls = introspection.get_class(
            package_name="batchie",
            class_name=args.plate_generator,
            base_class=RetrospectivePlateGenerator,
        )

        required_args = introspection.get_required_init_args_with_annotations(
            args.plate_generator_cls
        )

        if not args.plate_generator_param:
            args.plate_generator_params = {}
        else:
            args.plate_generator_params = cast_dict_to_type(
                args.plate_generator_param, required_args
            )

    if args.initial_plate_generator is not None:
        args.initial_plate_generator_cls = introspection.get_class(
            package_name="batchie",
            class_name=args.initial_plate_generator,
            base_class=InitialRetrospectivePlateGenerator,
        )

        required_args = introspection.get_required_init_args_with_annotations(
            args.initial_plate_generator_cls
        )

        if not args.initial_plate_generator_param:
            args.initial_plate_generator_params = {}
        else:
            args.initial_plate_generator_params = cast_dict_to_type(
                args.initial_plate_generator_param, required_args
            )

    if args.plate_smoother is not None:
        args.plate_smoother_cls = introspection.get_class(
            package_name="batchie",
            class_name=args.plate_smoother,
            base_class=RetrospectivePlateSmoother,
        )

        required_args = introspection.get_required_init_args_with_annotations(
            args.plate_smoother_cls
        )

        if not args.plate_smoother_param:
            args.plate_smoother_params = {}
        else:
            args.plate_smoother_params = cast_dict_to_type(
                args.plate_smoother_param, required_args
            )

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    screen = Screen.load_h5(args.data)

    filtered_screen = filter_dataset_to_treatments_that_appear_in_at_least_one_combo(
        screen
    )
    logger.info(
        "Screen size dropped from {} -> {} after removing "
        "spurious single treatment experiments that do not "
        "appear in any combo experiment".format(screen.size, filtered_screen.size)
    )

    rng = get_prng_from_seed_argument(args)

    initial_plate_generator: Optional[InitialRetrospectivePlateGenerator] = None

    if args.initial_plate_generator is not None:
        initial_plate_generator: InitialRetrospectivePlateGenerator = (
            args.initial_plate_generator_cls(**args.initial_plate_generator_params)
        )

        initialized_screen = initial_plate_generator.generate_and_unmask_initial_plate(
            screen=filtered_screen, rng=rng
        )
    else:
        initialized_screen = mask_screen(screen=filtered_screen)

    if args.plate_generator is not None:
        plate_generator: RetrospectivePlateGenerator = args.plate_generator_cls(
            **args.plate_generator_params
        )

        initialized_screen_with_generated_plates = plate_generator.generate_plates(
            screen=initialized_screen, rng=rng
        )
    else:
        logger.warning(
            "No plate generator was provided, will use plate ids provided in input screen."
        )
        initialized_screen_with_generated_plates = initialized_screen

    if initial_plate_generator is None:
        logger.warning(
            "No initial plate generator was provided. Selecting a random plate to reveal."
        )
        random_first_plate = rng.choice(
            [
                plate
                for plate in initialized_screen_with_generated_plates.plates
                if not plate.is_observed
            ]
        )

        logger.warning("Will reveal plate {}".format(random_first_plate.plate_id))
        initialized_screen_with_generated_plates = reveal_plates(
            screen=initialized_screen_with_generated_plates,
            plate_ids=[random_first_plate.plate_id],
        )

    logger.info(
        "Created {} plates (before smoothing)".format(
            initialized_screen_with_generated_plates.n_plates
        )
    )

    if args.plate_smoother is None:
        logger.info("No plate smoother specified, skipping smoothing")
        smoothed_screen = initialized_screen_with_generated_plates
    else:
        logger.info("Applying {}".format(args.plate_smoother))
        plate_smoother: RetrospectivePlateSmoother = args.plate_smoother_cls(
            **args.plate_smoother_params
        )

        smoothed_screen = plate_smoother.smooth_plates(
            screen=initialized_screen_with_generated_plates, rng=rng
        )

        n_plates = smoothed_screen.n_plates
        avg_plate_size = smoothed_screen.size / n_plates
        plate_size_std = np.std([plate.size for plate in smoothed_screen.plates])

        logger.info(
            "After smoothing {} plates remain of average size {} with stddev in size of {}".format(
                n_plates, avg_plate_size, plate_size_std
            )
        )

    training_screen, test_screen = create_holdout_set(
        screen=smoothed_screen, fraction=args.holdout_fraction, rng=rng
    )

    training_screen.save_h5(args.training_output)
    test_screen.save_h5(args.test_output)
