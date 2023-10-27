import argparse
from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import (
    KVAppendAction,
    cast_dict_to_type,
    get_prng_from_seed_argument,
)
from batchie.core import InitialRetrospectivePlateGenerator, RetrospectivePlateGenerator
from batchie.data import Experiment
from batchie.retrospective import reveal_plates, mask_experiment
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for revealing plates in a retrospective experiment,"
        "calculating the prediction error on the un-revealed plates thus far,"
        "and saving the results."
    )
    log_config.add_logging_args(parser)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--initial-plate-generator", type=str, default=None)
    parser.add_argument(
        "--initial-plate-generator-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Initial plate generator parameters",
    )
    parser.add_argument("--plate-generator", type=str, required=True)
    parser.add_argument(
        "--plate-generator-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Plate generator parameters",
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

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    full_experiment = Experiment.load_h5(args.experiment)

    rng = get_prng_from_seed_argument(args)

    plate_generator: RetrospectivePlateGenerator = args.plate_generator_cls(
        **args.plate_generator_params
    )

    initial_plate_generator: Optional[InitialRetrospectivePlateGenerator] = None

    if args.initial_plate_generator is not None:
        initial_plate_generator: InitialRetrospectivePlateGenerator = (
            args.initial_plate_generator_cls(**args.initial_plate_generator_params)
        )

        experiment = initial_plate_generator.generate_and_unmask_initial_plate(
            experiment=full_experiment, rng=rng
        )
    else:
        experiment = mask_experiment(experiment=full_experiment)

    experiment = plate_generator.generate_plates(experiment=experiment, rng=rng)
    logger.info("Generated {} plates".format(len(experiment.plates)))

    if initial_plate_generator is None:
        logger.warning(
            "No initial plate generator was provided. Selecting a random plate to reveal."
        )

        logger.info([plate for plate in experiment.plates if not plate.is_observed])
        random_first_plate = rng.choice(
            [plate for plate in experiment.plates if not plate.is_observed]
        )

        logger.warning("Will reveal plate {}".format(random_first_plate))
        experiment = reveal_plates(
            full_experiment=full_experiment,
            masked_experiment=experiment,
            plate_ids=[random_first_plate.plate_id],
        )

    experiment.save_h5(args.output)
