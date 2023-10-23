import argparse
from itertools import combinations

import numpy as np
from batchie import introspection
from batchie import log_config
from batchie.cli.argument_parsing import KVAppendAction, cast_dict_to_type
from batchie.common import N_UNIQUE_SAMPLES, N_UNIQUE_TREATMENTS
from batchie.core import DistanceMetric, BayesianModel, ThetaHolder
from batchie.data import Experiment
from batchie.distance_calculation import (
    calculate_pairwise_distance_matrix_on_predictions,
)


def get_parser():
    parser = argparse.ArgumentParser(description="calculate_distance_matrix.py")
    log_config.add_logging_args(parser)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--samples", type=str, required=True, nargs="+")
    parser.add_argument("--distance-metric", type=str, required=True)
    parser.add_argument(
        "--distance-metric-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Distance metric parameters",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model-param",
        nargs=1,
        action=KVAppendAction,
        metavar="KEY=VALUE",
        help="Model parameters",
    )
    parser.add_argument("--n-chunks", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    args.metric_cls = introspection.get_class(
        package_name="batchie",
        class_name=args.distance_metric,
        base_class=DistanceMetric,
    )

    required_args = introspection.get_required_init_args_with_annotations(
        args.metric_cls
    )

    if not args.distance_metric_param:
        args.metric_params = {}
    else:
        args.metric_params = cast_dict_to_type(
            args.distance_metric_param, required_args
        )

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

    data = Experiment.load_h5(args.data)

    args.model_params[N_UNIQUE_SAMPLES] = data.n_unique_samples
    args.model_params[N_UNIQUE_TREATMENTS] = data.n_unique_treatments

    model: BayesianModel = args.model_cls(**args.model_params)

    samples_holder: ThetaHolder = model.get_results_holder(n_samples=1)

    samples = samples_holder.concat([samples_holder.load_h5(x) for x in args.samples])

    distance_metric: DistanceMetric = args.metric_cls(**args.metric_params)

    if args.n_chunks > 1:
        idx_chunks = np.array_split(np.arange(samples.n_thetas), args.n_chunks)

        chunk_to_run = sorted(list(combinations(idx_chunks, 2)))[args.chunk_index]

        chunk_indices = tuple([np.array(chunk_to_run[0]), np.array(chunk_to_run[1])])
    else:
        chunk_indices = tuple(
            [np.arange(samples.n_thetas), np.arange(samples.n_thetas)]
        )

    result = calculate_pairwise_distance_matrix_on_predictions(
        model=model,
        samples=samples,
        distance_metric=distance_metric,
        data=data,
        chunk_indices=chunk_indices,
    )

    result.save(args.output)
