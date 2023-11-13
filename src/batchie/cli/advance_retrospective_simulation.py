import argparse
import json
import os.path
import logging

from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie.core import BayesianModel, ThetaHolder, SimulationTracker
from batchie.data import Screen
from batchie.common import SELECTED_PLATES_KEY, N_UNIQUE_SAMPLES, N_UNIQUE_TREATMENTS
from batchie.retrospective import reveal_plates, calculate_mse


logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for revealing plates in a retrospective simulation,"
        "calculating the prediction error on a holdout test set,"
        "and saving the results."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--training-screen",
        help="A batchie Screen in hdf5 format with some plates observed.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test-screen",
        help="A batchie Screen in hdf5 format with all plates observed.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch-selection",
        help="A json file containing the next batch to reveal.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--screen-output",
        help="Output batchie Screen in hdf5 format with the next batch of experiments revealed.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--simulation-tracker-input",
        help="A batchie SimulationTracker in json format.",
        type=str,
        nargs="?",
        const=None,
        default=None,
    )
    parser.add_argument(
        "--simulation-tracker-output",
        help="An updated batchie SimulationTracker in json format.",
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

    training_screen = Screen.load_h5(args.training_screen)

    args.model_params[N_UNIQUE_SAMPLES] = training_screen.n_unique_samples
    args.model_params[N_UNIQUE_TREATMENTS] = training_screen.n_unique_treatments

    model: BayesianModel = args.model_cls(**args.model_params)
    theta_holder: ThetaHolder = model.get_results_holder(n_samples=1)
    thetas = theta_holder.concat([theta_holder.load_h5(x) for x in args.thetas])

    test_screen = Screen.load_h5(args.test_screen)

    if args.simulation_tracker_input and os.path.exists(args.simulation_tracker_input):
        simulation_tracker = SimulationTracker.load(args.simulation_tracker_input)
    else:
        logger.warning("No simulation tracker provided, will create blank one.")
        simulation_tracker = SimulationTracker(
            plate_ids_selected=[], losses=[], seed=args.seed
        )

    mse = calculate_mse(
        observed_screen=test_screen,
        thetas=thetas,
        model=model,
    )

    simulation_tracker.losses.append(mse)

    with open(args.batch_selection, "r") as f:
        next_batch = json.load(f)

    plates_to_reveal = next_batch[SELECTED_PLATES_KEY]
    advanced_screen = reveal_plates(training_screen, plates_to_reveal)

    simulation_tracker.plate_ids_selected.append(plates_to_reveal)

    advanced_screen.save_h5(args.screen_output)
    simulation_tracker.save(args.simulation_tracker_output)
