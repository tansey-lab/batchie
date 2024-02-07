import argparse
import json
import logging
import os

from batchie import log_config, plotting
from batchie.core import ThetaHolder
from batchie.data import Screen
from batchie.models.main import ModelEvaluation, correlation_matrix

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for analyzing and plotting the results of model fitting."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--model-evaluation",
        help="A batchie ModelEvaluation in hdf5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--screen",
        help="A batchie Screen in hdf5 format.",
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
        "--output-dir",
        help="Output directory.",
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

    return args


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_config.configure_logging(args)

    theta_holder: ThetaHolder = ThetaHolder(n_thetas=1)

    theta_holders = [theta_holder.load_h5(x) for x in args.thetas]

    thetas = theta_holder.concat(theta_holders)

    screen = Screen.load_h5(args.screen)

    me = ModelEvaluation.load_h5(args.model_evaluation)

    logger.info("Calculating correlation matrix")

    corr = correlation_matrix(screen, thetas)

    logger.info("Creating plots")

    plotting.plot_correlation_heatmap(
        corr, os.path.join(args.output_dir, "sample_prediction_correlation.pdf")
    )

    plotting.predicted_vs_observed_scatterplot(
        me, os.path.join(args.output_dir, "predicted_vs_observed_scatterplot.pdf")
    )

    plotting.predicted_vs_observed_scatterplot_per_sample(
        me,
        os.path.join(
            args.output_dir, "predicted_vs_observed_by_sample_scatterplot.pdf"
        ),
    )

    plotting.per_sample_violin_plot(
        me,
        os.path.join(args.output_dir, "per_sample_violin_plot.pdf"),
    )

    plotting.per_sample_violin_plot(
        me,
        os.path.join(args.output_dir, "per_sample_violin_plot__99th_percentiles.pdf"),
        percentile=99,
    )

    summary_statistics = {
        "mse": me.mse(),
        "mse_variance": me.mse_variance(),
        "inter_chain_mse_variance": me.inter_chain_mse_variance(),
    }

    with open(os.path.join(args.output_dir, "summary_statistics.json"), "w") as f:
        json.dump(summary_statistics, f, indent=4)
