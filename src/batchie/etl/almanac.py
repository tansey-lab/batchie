import os

import numpy as np
import pandas as pd
from scipy.special import logit, expit

from batchie.data import Dataset


def load_data(path: str) -> Dataset:
    data = load_almanac(os.path.join(path, "ComboDrugGrowth_Nov2017.csv"))

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
        sample_names=X_mean.cline.to_numpy(),
        plate_names=X_mean.plate.to_numpy(),
    )


def load_almanac(fname, eps=0.01):
    dfraw = pd.read_csv(fname)
    dfraw["CLINE"] = dfraw.CELLNAME
    dfraw.replace("MDA-MB-231/ATCC", "MDA-MB-231", inplace=True)
    dfraw.replace("SF-539\x1a", "SF-539", inplace=True)

    df = dfraw.copy()
    study2id = {x: i for i, x in enumerate(df.STUDY.unique())}
    screener2id = {x: i for i, x in enumerate(df.SCREENER.unique())}
    study = np.array([study2id[x] for x in df.STUDY])
    screener = np.array([screener2id[x] for x in df.SCREENER])
    # dummy_num = df.NSC1.max() + 1  # this is just a hack
    dummy_num = -1
    df.loc[df.CONCINDEX2 == 0, "NSC2"] = dummy_num
    df["NSC1"] = df.NSC1.astype(int)
    df["NSC2"] = df.NSC2.astype(int)

    cline = df.CLINE.values
    clines = np.unique(cline)
    clines2id = {x: i for i, x in enumerate(clines)}

    nsc1 = df.NSC1.values
    nsc2 = df.NSC2.values
    drugs = np.setdiff1d(np.union1d(nsc1, nsc2), [dummy_num])
    drugs2id = {x: i for i, x in enumerate(drugs)}
    drugs2id[dummy_num] = -1  # len(drugs2id)  # assign dummy to the last one

    plate = df.PLATE.values
    unique_plates = (  # order plates historically
        df[["TESTDATE", "PLATE"]]
        .drop_duplicates()
        .sort_values(["TESTDATE", "PLATE"])
        .PLATE.values
    )
    plates2id = {x: i for i, x in enumerate(unique_plates)}

    dfraw["CONC1"] = dfraw["CONC1"].fillna(-1)
    dfraw["CONC2"] = dfraw["CONC2"].fillna(-1)

    # %% drugdoses
    conc1 = dfraw["CONC1"].values
    conc2 = dfraw["CONC2"].values

    doses = np.union1d(conc1, conc2)
    doses2id = {x: i for i, x in enumerate(doses)}
    doses2id[-1.0] = -1

    dose1 = np.vectorize(doses2id.get)(conc1)
    dose2 = np.vectorize(doses2id.get)(conc2)

    drugdoses = sorted(list(set(list(zip(nsc1, dose1)) + list(zip(nsc2, dose2)))))
    drugdoses2id = {x: (i - 1) for i, x in enumerate(drugdoses) if x[0] != dummy_num}
    drugdoses2id[
        dummy_num, -1
    ] = -1  # len(drugdoses2id)  # assign dummy to the last one

    ## Put everything in the correct format
    cline = np.vectorize(clines2id.get)(cline)
    plates = np.vectorize(plates2id.get)(plate)
    drug1 = np.vectorize(drugs2id.get)(nsc1)
    drug2 = np.vectorize(drugs2id.get)(nsc2)
    drugdose1 = np.array(
        [drugdoses2id[drug, dose] for drug, dose in zip(nsc1, dose1)], dtype=int
    )
    drugdose2 = np.array(
        [drugdoses2id[drug, dose] for drug, dose in zip(nsc2, dose2)], dtype=int
    )
    prob = (df.PERCENTGROWTH + 100) / 200
    prob = np.clip(prob.values, eps, 1.0 - eps)
    y = logit(prob)

    X = {
        "cline": cline,
        "drugdose1": drugdose1,
        "drugdose2": drugdose2,
        "plate": plates,
        "drug1": drug1,
        "drug2": drug2,
        "dose1": dose1,
        "dose2": dose2,
        "conc1": conc1,
        "conc2": conc2,
    }
    X = pd.DataFrame(X)

    ## Make sure everything lines up
    unique_dds = np.union1d(drugdose1, drugdose2)
    expected_dds = np.arange(len(unique_dds)) - 1
    assert np.all(unique_dds == expected_dds), "Missing drug doses"

    unique_clines = np.unique(cline)
    expected_clines = np.arange(len(unique_clines))
    assert np.all(unique_clines == expected_clines), "Missing cell lines"

    output = dict(
        y=y,
        study=study,
        screener=screener,
        X=X,
        drugdoses2id=drugdoses2id,
        drugs2id=drugs2id,
        clines2id=clines2id,
        plates2id=plates2id,
        has_controls=True,
    )
    return output
