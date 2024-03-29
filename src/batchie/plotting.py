import seaborn as sns
import numpy as np

from pandas import DataFrame
from matplotlib import pyplot as plt
from batchie.models.main import ModelEvaluation
from scipy.stats import spearmanr
from math import ceil


def plot_correlation_heatmap(correlation: DataFrame, output_fn: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(data=correlation, ax=ax, annot=True, fmt=".2f", cmap="coolwarm")

    # Rotate the axis labels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(output_fn)


def predicted_vs_observed_scatterplot(
    model_evaluation: ModelEvaluation, output_fn: str
):
    fig, ax = plt.subplots(figsize=(8, 6))

    rho, _ = spearmanr(model_evaluation.observations, model_evaluation.mean_predictions)

    sns.regplot(
        ax=ax,
        x=model_evaluation.observations,
        y=model_evaluation.mean_predictions,
        scatter_kws={"color": "black", "s": 3},
    )

    ax.annotate(
        f"ρ = {rho:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
    )

    plt.title("Observed vs Predicted on Held Out Data")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")

    fig.tight_layout()
    fig.savefig(output_fn)


def predicted_vs_observed_scatterplot_per_sample(
    model_evaluation: ModelEvaluation, output_fn: str, ncol=3
):
    df = DataFrame(
        {
            "X": model_evaluation.observations,
            "Y": model_evaluation.mean_predictions,
            "Category": model_evaluation.sample_names,
        }
    )
    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()

    unique_categories = df["Category"].unique()

    n_categories = len(unique_categories)

    fig, axs = plt.subplots(
        ceil(n_categories / ncol), ncol, figsize=(8, 4 * ceil(n_categories / ncol))
    )

    if len(axs.shape) == 1:
        axs = np.array([axs])

    for i, category in enumerate(unique_categories):
        ax = axs[i // ncol][i % ncol]
        category_data = df[df["Category"] == category]

        rho, _ = spearmanr(category_data["X"], category_data["Y"])

        sns.regplot(
            x="X",
            y="Y",
            data=category_data,
            ax=ax,
            scatter_kws={"color": "black", "s": 3},
        )

        # Annotating the plot with Spearman's rho
        ax.annotate(
            f"ρ = {rho:.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        )

        ax.set_title(f"{category}")
        ax.set_xlabel("obs")
        ax.set_ylabel("pred")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for idx, ax in enumerate(axs.flatten()):
        if idx > (n_categories - 1):
            ax.remove()

    fig.tight_layout()
    fig.savefig(output_fn)


def filter_df_groups_by_percentile(df: DataFrame, gbkey, val_key, percentile: float):
    return df[
        df.groupby(gbkey)[val_key].transform(lambda x: x > np.percentile(x, percentile))
    ]


def per_sample_violin_plot(
    model_evaluation: ModelEvaluation, output_fn: str, percentile=0.0
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    df = DataFrame()

    df["observed"] = model_evaluation.observations
    df["prediction"] = model_evaluation.mean_predictions
    df["variance"] = np.var(model_evaluation.predictions, axis=1)
    df["sample"] = model_evaluation.sample_names

    sns.violinplot(
        x="sample",
        y="observed",
        data=filter_df_groups_by_percentile(df, "sample", "observed", percentile),
        ax=ax1,
        hue="sample",
    )
    ax1.set_xlabel("Sample name")
    ax1.set_ylabel("Observed viability")

    sns.violinplot(
        x="sample",
        y="prediction",
        data=filter_df_groups_by_percentile(df, "sample", "prediction", percentile),
        ax=ax2,
        hue="sample",
    )
    ax2.set_xlabel("Sample name")
    ax2.set_ylabel("Mean predicted viability")

    sns.violinplot(
        x="sample",
        y="variance",
        data=filter_df_groups_by_percentile(df, "sample", "variance", percentile),
        ax=ax3,
        hue="sample",
    )
    ax3.set_xlabel("Sample name")
    ax3.set_ylabel("Variance of mean predicted viability")

    fig.tight_layout()
    fig.savefig(output_fn)
