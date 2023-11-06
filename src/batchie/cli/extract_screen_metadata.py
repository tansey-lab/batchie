import argparse
import json
from batchie import log_config
from batchie.data import Screen


def get_parser():
    parser = argparse.ArgumentParser(description="extract_screen_metadata.py")
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--screen", help="A batchie Screen in hdf5 format.", type=str, required=True
    )
    parser.add_argument(
        "--output",
        help="Output json file to save metadata to.",
        type=str,
        required=True,
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    experiment = Screen.load_h5(args.screen)

    n_observed_plates = 0
    n_unobserved_plates = 0

    for plate in experiment.plates:
        if plate.is_observed:
            n_observed_plates += 1
        else:
            n_unobserved_plates += 1

    result_object = {
        "n_unique_samples": experiment.n_unique_samples,
        "n_unique_treatments": experiment.n_unique_treatments,
        "size": experiment.size,
        "n_plates": experiment.n_plates,
        "n_unobserved_plates": n_unobserved_plates,
        "n_observed_plates": n_observed_plates,
    }

    with open(args.output, "w") as f:
        json.dump(result_object, f, indent=4)
