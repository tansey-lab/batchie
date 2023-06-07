import os

import numpy as np
import pandas as pd
from scipy.special import logit

from batchie.data import Dataset


def load_sarcoma(
    path: str,
    round_10_validation: bool = False,
    round_15_random_validation: bool = False,
    round_15_targeted_validation: bool = False,
) -> Dataset:
    df = pd.read_csv(os.path.join(path, "sarcoma.csv"))
    df = df[~df["Failed QC"]]
    df.drop(columns=["Failed QC", "Raw count"], inplace=True)
    df = df[
        (~df["Drug 1 (Kung Lab ID)"].isin(["high control", "low control"]))
        & (~df["Drug 2 (Kung Lab ID)"].isin(["high control", "low control"]))
    ]
    df.fillna(-1, inplace=True)

    if round_10_validation:
        df_inter = df[df["Iteration"] == "validation (round 10)"].copy()
        df_inter.replace("validation (round 10)", 11, inplace=True)
        df_inter = df_inter[
            df_inter["Conc 1 (uM)"].isin([-1.0, 0.1, 1.0])
            & df_inter["Conc 2 (uM)"].isin([-1.0, 0.1, 1.0])
        ]
        df_inter.drop(columns=["file"], inplace=True)

    if round_15_random_validation:
        df.replace("validation (random, round 15)", 16, inplace=True)

    if round_15_targeted_validation:
        df.replace("validation (targeted, round 15)", 17, inplace=True)

    df = df[
        ~df.Iteration.isin(
            [
                "validation (round 10)",
                "validation (random, round 15)",
                "validation (targeted, round 15)",
            ]
        )
    ]

    df["Iteration"] = pd.to_numeric(df["Iteration"])
    print(df.shape)

    ## Replicates happen starting at round 11
    df_uni = df[df["Iteration"] < 11].copy()
    df_uni.drop(columns=["file"], inplace=True)

    df_rep = df[df["Iteration"] >= 11].copy()

    ## Only average across appropriate format
    x = [j.split("_") for j in df_rep["file"].values]
    pformat1 = [j[3].strip() for j in x]
    z = [j[4].split("+") for j in x]
    pformat2 = [j[1].strip() for j in z]
    d1 = [j[0].strip() for j in z]
    d2 = [j[5] for j in x]
    df_rep["format"] = ["_".join(l) for l in zip(pformat1, d1, pformat2, d2)]

    df_rep.drop(columns=["file"], inplace=True)
    df_rep = (
        df_rep.groupby(
            [
                "Cell line",
                "Drug 1 (Kung Lab ID)",
                "Drug 2 (Kung Lab ID)",
                "Conc 1 (uM)",
                "Conc 2 (uM)",
                "Plate row",
                "Plate column",
                "Iteration",
                "format",
            ]
        )
        .mean()
        .reset_index()
    )
    df_rep.drop(columns=["format"], inplace=True)

    ## Combine everything back together
    if round_10_validation:
        df = pd.concat([df_uni, df_rep, df_inter], ignore_index=True)
    else:
        df = pd.concat([df_uni, df_rep], ignore_index=True)

    ## Place cell lines, drugs, concentrations and drug-doses into categorical 0,1,2...,etc order
    clines = np.unique(df["Cell line"])
    cline2id = {x: i for i, x in enumerate(clines)}
    drugs = np.union1d(df["Drug 1 (Kung Lab ID)"], df["Drug 2 (Kung Lab ID)"])
    drugs2id = {x: i for i, x in enumerate(drugs)}
    drugs2id["dmso"] = -1
    dose2id = {0.1: 0, 1.0: 1, -1: -1}

    id2drugs = {val: key for key, val in drugs2id.items()}
    id2dose = {val: key for key, val in dose2id.items()}

    clookup = np.vectorize(cline2id.get)
    dlookup = np.vectorize(drugs2id.get)
    doselookup = np.vectorize(dose2id.get)

    df["Cell line"] = clookup(df["Cell line"])
    df["Drug 1 (Kung Lab ID)"] = dlookup(df["Drug 1 (Kung Lab ID)"])
    df["Drug 2 (Kung Lab ID)"] = dlookup(df["Drug 2 (Kung Lab ID)"])
    df["dose1"] = doselookup(df["Conc 1 (uM)"].values)
    df["dose2"] = doselookup(df["Conc 2 (uM)"].values)
    df.rename(
        columns={
            "Cell line": "cline",
            "Drug 1 (Kung Lab ID)": "drug1",
            "Drug 2 (Kung Lab ID)": "drug2",
        },
        inplace=True,
    )

    drugdose1 = list(zip(df["drug1"], df["dose1"]))
    drugdose2 = list(zip(df["drug2"], df["dose2"]))

    drugdoses = sorted(list(set(drugdose1 + drugdose2)))
    drugdose2id = {x: i for i, x in enumerate(drugdoses[1:])}
    drugdose2id[(-1, -1)] = -1
    df["drugdose1"] = [drugdose2id[x] for x in drugdose1]
    df["drugdose2"] = [drugdose2id[x] for x in drugdose2]

    drugdoses2id = {
        (id2drugs[drug_id], id2dose[dose_id]): val
        for (drug_id, dose_id), val in drugdose2id.items()
    }
    ## Sort by iteration
    df.sort_values(by=["Iteration", "Plate"], ignore_index=True, inplace=True)

    uniq_iter_plate_values = (
        df[["Iteration", "Plate"]]
        .drop_duplicates()
        .sort_values(by=["Iteration", "Plate"], ignore_index=True)
        .to_records(index=False)
    )

    iter_plate_to_plate_id = {
        tuple(rec): idx for idx, rec in enumerate(uniq_iter_plate_values)
    }

    df["plate_id"] = df.apply(
        lambda r: iter_plate_to_plate_id[r["Iteration"], r["Plate"]], axis=1
    )

    ## Pull out viabilities
    df["y_logit"] = logit(np.clip(df["Viability"], a_min=0.01, a_max=0.99))

    return Dataset(
        observations=df.y_logit.to_numpy(),
        treatments=df[["drugdose1", "drugdose2"]].to_numpy(),
        sample_names=df.cline.to_numpy(),
        plate_names=df.plate_id.to_numpy(),
    )
