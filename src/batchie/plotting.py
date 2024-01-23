import seaborn as sns

from pandas import DataFrame
from matplotlib import pyplot as plt
from batchie.models.main import ModelEvaluation


def plot_correlation_heatmap(correlation: DataFrame, output_fn: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(data=correlation, ax=ax, annot=True, fmt=".2f", cmap="coolwarm")

    # Rotate the axis labels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(output_fn)


def predicted_vs_observed_scatterplot(model_evaluation: ModelEvaluation):
    model_evaluation
