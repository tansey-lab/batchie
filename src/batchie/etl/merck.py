import numpy as np
import os, pickle
import pandas as pd
from scipy.special import logit, expit
from batchie.data import Dataset


def load_data(path: str) -> Dataset:
    combo_fname = os.path.join(path, "156849_1_supp_1_w2lrww.xls")
    single_fname = os.path.join(path, "156849_1_supp_0_w2lh45.xlsx")
    data = load_and_process_merck(combo_fname, single_fname)

    ## Place into canonical order
    X = data["X"]
    X["y_viability"] = expit(data["y"])
    drug_lookup = dict(zip(X["drugdose1"].values, X["drug1"].values))
    drug_lookup.update(dict(zip(X["drugdose2"].values, X["drug2"].values)))
    dose_lookup = dict(zip(X["drugdose1"], X["dose1"]))
    dose_lookup.update(dict(zip(X["drugdose2"], X["dose2"])))
    conc_lookup = dict(zip(X["dose1"], X["conc1"]))
    conc_lookup.update(dict(zip(X["dose2"], X["conc2"])))

    dmax = np.maximum(X["drugdose1"], X["drugdose2"])
    dmin = np.minimum(X["drugdose1"], X["drugdose2"])
    X["drugdose1"] = dmin
    X["drugdose2"] = dmax
    X["drug1"] = np.vectorize(drug_lookup.get)(X["drugdose1"].values)
    X["drug2"] = np.vectorize(drug_lookup.get)(X["drugdose2"].values)
    X["dose1"] = np.vectorize(dose_lookup.get)(X["drugdose1"].values)
    X["dose2"] = np.vectorize(dose_lookup.get)(X["drugdose2"].values)
    X["conc1"] = np.vectorize(conc_lookup.get)(X["dose1"].values)
    X["conc2"] = np.vectorize(conc_lookup.get)(X["dose2"].values)

    ## Average replicates
    X_mean = X.groupby(["cline", "drugdose1", "drugdose2"]).mean().reset_index()
    X_mean["y_logit"] = logit(X_mean["y_viability"].values)
    X_mean[["drug1", "drug2", "dose1", "dose2"]] = X_mean[
        ["drug1", "drug2", "dose1", "dose2"]
    ].astype(int)

    return Dataset(
        observations=X_mean.y_viability.to_numpy(),
        treatments=X_mean[["drugdose1", "drugdose2"]].to_numpy(),
        sample_ids=X_mean.cline.to_numpy(),
        plate_ids=X_mean.plate.to_numpy(),
    )


def load_and_process_merck(
    combo_datapath: str, single_datapath: str, eps: float = 0.01, **kwargs
):
    combo_df = pd.read_excel(combo_datapath)
    single_df = pd.read_excel(single_datapath)

    ## Average viabilities in each of the datasets
    single_df["viability"] = single_df[
        [
            "viability1",
            "viability2",
            "viability3",
            "viability4",
            "viability5",
            "viability6",
        ]
    ].mean(axis=1)
    combo_df["viability"] = combo_df[
        ["viability1", "viability2", "viability3", "viability4"]
    ].mean(axis=1)

    ## Remove superfluous columns
    combo_df = combo_df[
        [
            "cell_line",
            "drugA_name",
            "drugA Conc (µM)",
            "drugB_name",
            "drugB Conc (µM)",
            "viability",
        ]
    ]
    single_df = single_df[
        ["cell_line", "drug_name", "Drug_concentration (µM)", "viability"]
    ]

    ## Align single drug control dataset into the same format as combo dataset
    single_df = single_df.rename(
        columns={
            "drug_name": "drugA_name",
            "Drug_concentration (µM)": "drugA Conc (µM)",
        }
    )
    single_df["drugB_name"] = "None"
    single_df["drugB Conc (µM)"] = -1
    df = pd.concat([combo_df, single_df], ignore_index=True)
    df = df.rename(columns={"drugA Conc (µM)": "conc1", "drugB Conc (µM)": "conc2"})

    ## Map cell names, drug names and concentrations to ordinals
    cellname = np.unique(df["cell_line"])
    cellmap = {x: i for i, x in enumerate(cellname)}
    drugnames = np.setdiff1d(np.union1d(df["drugA_name"], df["drugB_name"]), ["None"])
    drugmap = {x: i for i, x in enumerate(drugnames)}
    drugmap["None"] = -1

    df["cline"] = np.vectorize(cellmap.get)(df["cell_line"].values)
    df["drug1"] = np.vectorize(drugmap.get)(df["drugA_name"].values)
    df["drug2"] = np.vectorize(drugmap.get)(df["drugB_name"].values)

    concs = np.setdiff1d(np.union1d(df["conc1"], df["conc2"]), [-1])
    dosemap = {x: i for i, x in enumerate(concs)}
    dosemap[-1] = -1
    df["dose1"] = np.vectorize(dosemap.get)(df["conc1"].values)
    df["dose2"] = np.vectorize(dosemap.get)(df["conc2"].values)

    ## Get unique drug/dose pairs
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

    ## Extract viability
    y = logit(np.clip(df["viability"].values, eps, 1.0 - eps))

    ## Drop superfluous columns
    X = df[
        [
            "cline",
            "drugdose1",
            "drugdose2",
            "drug1",
            "drug2",
            "dose1",
            "dose2",
            "conc1",
            "conc2",
        ]
    ]

    output = dict(
        y=y,
        X=X,
        drugdoses2id=drugdoses2id,
        drugs2id=drugmap,
        clines2id=cellmap,
        has_controls=True,
    )
    return output
