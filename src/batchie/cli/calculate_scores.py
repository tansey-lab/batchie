import argparse
import logging

from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie.common import N_UNIQUE_SAMPLES, N_UNIQUE_TREATMENTS
from batchie.core import BayesianModel, ThetaHolder, Scorer
from batchie.data import Screen
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.scoring.main import score_chunk

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="calculate_scores.py")
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
        "--n-chunks",
        help="Number of chunks to parallelize scoring over.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--chunk-index",
        help="Which of the n chunks to calculate in this invocation.",
        type=int,
        default=0,
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
        "--batch-plate-ids",
        help="The plate(s) already selected as part of this batch.",
        nargs="+",
        type=int,
        default=list(),
    )
    parser.add_argument(
        "--output",
        help="Location of output h5 file where scores will be saved.",
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

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    screen = Screen.load_h5(args.data)

    args.model_params[N_UNIQUE_SAMPLES] = screen.n_unique_samples
    args.model_params[N_UNIQUE_TREATMENTS] = screen.n_unique_treatments

    model: BayesianModel = args.model_cls(**args.model_params)

    model.add_observations(screen.subset_observed())

    scorer: Scorer = args.scorer_cls(**args.scorer_params)

    thetas_holder: ThetaHolder = model.get_results_holder(n_samples=1)

    thetas = thetas_holder.concat([thetas_holder.load_h5(x) for x in args.thetas])

    n_plates = sum(
        [
            1
            for p in screen.plates
            if not p.is_observed and p.plate_id not in args.batch_plate_ids
        ]
    )

    logger.info(
        f"Calculating chunk {args.chunk_index + 1} of {args.n_chunks} "
        f"over {n_plates} unobserved plates."
    )

    distance_matrix = ChunkedDistanceMatrix.concat(
        [ChunkedDistanceMatrix.load(x) for x in args.distance_matrix]
    )

    result = score_chunk(
        model=model,
        scorer=scorer,
        distance_matrix=distance_matrix,
        thetas=thetas,
        screen=screen,
        chunk_index=args.chunk_index,
        n_chunks=args.n_chunks,
        progress_bar=args.progress,
        batch_plate_ids=args.batch_plate_ids,
    )

    logger.info("Saving results to {}".format(args.output))

    result.save_h5(args.output)
