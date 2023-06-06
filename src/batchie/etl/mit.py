import numpy as np
import os, pickle
import pandas as pd
from scipy.special import logit, expit
from batchie.sparse_combo.models import SparseDrugCombo, ComboPredictorHolder
from batchie.sparse_combo.datasets import ComboDataset, dataset_from_dict
from batchie.sparse_combo.metrics import MSEDistance, MSEMetric
from batchie.sparse_combo.batcher import BatcherKPerCline


def load_data(
    path: str,
    cline_frac: float = 1.0,
    dd_frac: float = 1.0,
    fake_plates: str = "sarcoma",
    warm_start: bool = True,
    per_cell: int = 3,
    revealed_init_frac: float = 0.0,
    revealed_clines: int = -1,
    revealed_sarcoma_dd_frac: float = 0.0,
    combo_only: bool = True,
    holdout_frac: float = 0.1,
    min_plate_obs: int = 50,
    nmerge_pass: int = 1,
) -> tuple[ComboDataset, list[str]]:
    ## Load clean data if it exists
    clean_fname = os.path.join(path, "mit_clean.pkl")
    if os.path.exists(clean_fname):
        with open(clean_fname, "rb") as io:
            data = pickle.load(io)
    else:
        fname = os.path.join(path, "pone.0140310.s012.xlsx")
        data = load_and_process_mit(fname)
        with open(clean_fname, "wb") as io:
            pickle.dump(data, io, protocol=pickle.HIGHEST_PROTOCOL)

    ## Place into canonical order
    X = data["X"]
    X["viability"] = expit(data["y"])
    drug_lookup = dict(zip(X["drugdose1"].values, X["drug1"].values))
    drug_lookup.update(dict(zip(X["drugdose2"].values, X["drug2"].values)))
    dose_lookup = dict(zip(X["drugdose1"], X["dose1"]))
    dose_lookup.update(dict(zip(X["drugdose2"], X["dose2"])))

    dmax = np.maximum(X["drugdose1"], X["drugdose2"])
    dmin = np.minimum(X["drugdose1"], X["drugdose2"])
    X["drugdose1"] = dmin
    X["drugdose2"] = dmax
    X["drug1"] = np.vectorize(drug_lookup.get)(X["drugdose1"].values)
    X["drug2"] = np.vectorize(drug_lookup.get)(X["drugdose2"].values)
    X["dose1"] = np.vectorize(dose_lookup.get)(X["drugdose1"].values)
    X["dose2"] = np.vectorize(dose_lookup.get)(X["drugdose2"].values)

    ## Average replicates
    X_mean = X.groupby(["cline", "drugdose1", "drugdose2"]).mean().reset_index()
    y_mean = logit(X_mean["viability"].values)
    X_mean = X_mean.drop(columns=["viability"])
    X_mean[["drug1", "drug2", "dose1", "dose2"]] = X_mean[
        ["drug1", "drug2", "dose1", "dose2"]
    ].astype(int)

    dataset, selected = dataset_from_dict(
        X=X_mean,
        y=y_mean,
        cline_frac=cline_frac,
        dd_frac=dd_frac,
        fake_plates=fake_plates,
        warm_start=warm_start,
        per_cell=per_cell,
        revealed_init_frac=revealed_init_frac,
        revealed_clines=revealed_clines,
        revealed_sarcoma_dd_frac=revealed_sarcoma_dd_frac,
        combo_only=combo_only,
        holdout_frac=holdout_frac,
        min_plate_obs=min_plate_obs,
        nmerge_pass=nmerge_pass,
        **kwargs,
    )

    return (dataset, selected)


def load_and_process_mit(datapath: str, eps: float = 0.01, **kwargs):
    ## Excel sheets are large matrices -- Drugs x Cell lines
    singledf = pd.read_excel(datapath, sheet_name="Single Drug Data", usecols="B:AL")
    combodf = pd.read_excel(
        datapath, sheet_name="Combination Drug Data", usecols="B:AN"
    )

    ## Place into row-wise form
    singledf = singledf.melt(id_vars=["Drug"], var_name="CLINE", value_name="GROWTH")
    singledf = singledf[~singledf["GROWTH"].isna()]

    combodf = combodf.melt(
        id_vars=["Drug 1", "Drug 2", "Drug Combo"],
        var_name="CLINE",
        value_name="GROWTH",
    )
    combodf = combodf[~combodf["GROWTH"].isna()]

    ## Calculate dose-level data
    singledf["dose1"] = [0 if "low" in x else 1 for x in singledf["Drug"].values]
    combodf["dose1"] = [0 if "low" in x else 1 for x in combodf["Drug Combo"].values]
    combodf["dose2"] = [0 if "low" in x else 1 for x in combodf["Drug Combo"].values]

    ## Specify drugs correctly in single-drug experiments
    singledf["Drug 1"] = [
        x.replace(" low", "").strip().replace("Vismodegib", "vismodegib")
        for x in singledf["Drug"].values
    ]
    singledf["Drug 2"] = "None"
    singledf["dose2"] = -1

    ## Remove superfluous columns
    singledf = singledf[["Drug 1", "dose1", "Drug 2", "dose2", "CLINE", "GROWTH"]]
    combodf = combodf[["Drug 1", "dose1", "Drug 2", "dose2", "CLINE", "GROWTH"]]

    ## Concatenate
    df = pd.concat([combodf, singledf], ignore_index=True)

    ## Map cell lines + drugs to unique ordinals
    cellnames = np.unique(df["CLINE"])
    cellmap = {x: i for i, x in enumerate(cellnames)}
    df["cline"] = np.vectorize(cellmap.get)(df["CLINE"])

    drugnames = np.setdiff1d(np.union1d(df["Drug 1"], df["Drug 2"]), ["None"])
    drugmap = {x: i for i, x in enumerate(drugnames)}
    drugmap["None"] = -1

    df["drug1"] = np.vectorize(drugmap.get)(df["Drug 1"])
    df["drug2"] = np.vectorize(drugmap.get)(df["Drug 2"])

    ## Map drug dose pairs to unique ordinals
    drugdoses = sorted(
        list(
            set(
                list(zip(df["drug1"].values, df["dose1"].values))
                + list(zip(df["drug2"].values, df["dose2"].values))
            )
        )
    )
    drugdoses = drugdoses[1:]  ## First will be (-1,-1)
    drugdoses2id = {x: i for i, x in enumerate(drugdoses)}
    drugdoses2id[(-1, -1)] = -1
    df["drugdose1"] = [
        drugdoses2id[drug, dose]
        for drug, dose in zip(df["drug1"].values, df["dose1"].values)
    ]
    df["drugdose2"] = [
        drugdoses2id[drug, dose]
        for drug, dose in zip(df["drug2"].values, df["dose2"].values)
    ]

    ## Drop superfluous columns
    X = df[["cline", "drugdose1", "drugdose2", "drug1", "drug2", "dose1", "dose2"]]

    ## Viability = growth/100
    y = logit(np.clip(df["GROWTH"].values / 100, eps, 1 - eps))

    output = dict(
        y=y,
        X=X,
        drugdoses2id=drugdoses2id,
        drugs2id=drugmap,
        clines2id=cellmap,
        has_controls=True,
    )
    return output
