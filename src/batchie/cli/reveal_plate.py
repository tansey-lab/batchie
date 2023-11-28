import argparse
import logging

from batchie import log_config
from batchie.data import Screen
from batchie.retrospective import reveal_plates

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for revealing a plates in a retrospective simulation."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--screen",
        help="A batchie Screen in hdf5 format with some plates observed.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Where to save screen with the next plate revealed.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--plate-id",
        help="The plate(s) to reveal.",
        nargs="+",
        type=int,
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

    screen = Screen.load_h5(args.screen)

    advanced_screen = reveal_plates(screen, args.plate_id)

    advanced_screen.save_h5(args.output)
