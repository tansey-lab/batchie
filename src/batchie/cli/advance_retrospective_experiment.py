import argparse
import json
import os.path
import logging

from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie.core import BayesianModel, ThetaHolder, ExperimentTracker
from batchie.data import Screen
from batchie.common import SELECTED_PLATES_KEY, N_UNIQUE_SAMPLES, N_UNIQUE_TREATMENTS
from batchie.retrospective import reveal_plates, calculate_mse


logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for revealing plates in a retrospective experiment,"
        "calculating the prediction error on the un-revealed plates thus far,"
        "and saving the results."
    )
    log_config.add_logging_args(parser)
    parser.add_argument("--unmasked-experiment", type=str, required=True)
    parser.add_argument("--masked-experiment", type=str, required=True)
    parser.add_argument("--batch-selection", type=str, required=True)
    parser.add_argument("--experiment-output", type=str, required=True)
    parser.add_argument(
        "--experiment-tracker-input", type=str, nargs="?", const=None, default=None
    )
    parser.add_argument("--experiment-tracker-output", type=str, required=True)
    parser.add_argument("--thetas", type=str, required=True, nargs="+")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Model parameters",
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

    masked_experiment = Screen.load_h5(args.masked_experiment)

    args.model_params[N_UNIQUE_SAMPLES] = masked_experiment.n_unique_samples
    args.model_params[N_UNIQUE_TREATMENTS] = masked_experiment.n_unique_treatments

    model: BayesianModel = args.model_cls(**args.model_params)
    theta_holder: ThetaHolder = model.get_results_holder(n_samples=1)
    thetas = theta_holder.concat([theta_holder.load_h5(x) for x in args.thetas])

    unmasked_experiment = Screen.load_h5(args.unmasked_experiment)

    if args.experiment_tracker_input and os.path.exists(args.experiment_tracker_input):
        experiment_tracker = ExperimentTracker.load(args.experiment_tracker_input)
    else:
        logger.warning("No experiment tracker provided, will create blank one.")
        experiment_tracker = ExperimentTracker(plate_ids_selected=[], losses=[], seed=0)

    mse = calculate_mse(
        full_experiment=unmasked_experiment,
        masked_experiment=masked_experiment,
        thetas=thetas,
        model=model,
    )

    experiment_tracker.losses.append(mse)

    with open(args.batch_selection, "r") as f:
        next_batch = json.load(f)

    plates_to_reveal = next_batch[SELECTED_PLATES_KEY]
    advanced_experiment = reveal_plates(
        unmasked_experiment, masked_experiment, plates_to_reveal
    )

    experiment_tracker.plate_ids_selected.append(plates_to_reveal)

    advanced_experiment.save_h5(args.experiment_output)
    experiment_tracker.save(args.experiment_tracker_output)
