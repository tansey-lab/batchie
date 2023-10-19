import argparse

from batchie import introspection
from batchie.core import BayesianModel
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie import log_config
from batchie import sampling


def get_parser():
    parser = argparse.ArgumentParser(description="train_model.py")
    log_config.add_logging_args(parser)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Model parameters",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--n-samples",
        help="Number of samples to save from the posterior distribution",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n-burnin",
        help="Number of steps to iterate before samples are saved",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--thin",
        help="Thinning factor for samples after burn-in is complete. "
        "A value of 2 means every seconds sample is saved, etc.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n-chains",
        help="Number of parallel instances of this model that are being run. "
        "This information is used for rng seed intialization.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chain-index",
        help="Index of this model in the series of parallel model runs.",
        type=int,
        default=0,
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    cls = introspection.get_class(
        package_name="batchie", class_name=args.model, base_class=BayesianModel
    )

    required_args = introspection.get_required_init_args_with_annotations(cls)

    args.model_params = cast_dict_to_type(args.model_param, required_args)

    args.model_cls = cls

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    model = args.model_cls(**args.model_params)


if __name__ == "__main__":
    main()
