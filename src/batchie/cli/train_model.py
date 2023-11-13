import argparse

from batchie import introspection
from batchie import log_config
from batchie import sampling
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie.common import N_UNIQUE_SAMPLES, N_UNIQUE_TREATMENTS
from batchie.core import BayesianModel, ThetaHolder
from batchie.data import Screen
import logging

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train the provided model by iteratively calling its #step method,"
        "conditioned on the provided data. Collect the model parameters and save"
        "to a file."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--data",
        help="A batchie Screen object in hdf5 format.",
        type=str,
        required=True,
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Model parameters",
    )
    parser.add_argument(
        "--output",
        help="Output file to save learned model parameters to.",
        type=str,
        required=True,
    )
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

    cls = introspection.get_class(
        package_name="batchie", class_name=args.model, base_class=BayesianModel
    )

    required_args = introspection.get_required_init_args_with_annotations(cls)

    if not args.model_param:
        args.model_params = {}
    else:
        args.model_params = cast_dict_to_type(args.model_param, required_args)

    args.model_cls = cls

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    data = Screen.load_h5(args.data)

    args.model_params[N_UNIQUE_SAMPLES] = data.n_unique_samples
    args.model_params[N_UNIQUE_TREATMENTS] = data.n_unique_treatments

    model: BayesianModel = args.model_cls(**args.model_params)

    samples_holder: ThetaHolder = model.get_results_holder(n_samples=args.n_samples)

    observed_subset = data.subset_observed()

    if observed_subset is not None:
        logger.info(
            "Will train model on {} observed experiments".format(observed_subset.size)
        )
        model.add_observations(observed_subset)
    else:
        logger.warning(
            "No observed experiments found. Model will be trained on no data "
            "(check your input screen if this is unexpected)"
        )

    results: ThetaHolder = sampling.sample(
        model=model,
        results=samples_holder,
        seed=args.seed,
        n_chains=args.n_chains,
        chain_index=args.chain_index,
        n_burnin=args.n_burnin,
        thin=args.thin,
        progress_bar=args.progress,
    )

    logger.info("Saving thetas to {}".format(args.output))

    results.save_h5(args.output)


if __name__ == "__main__":
    main()
