import os.path

import numpy as np
import pandas as pd
from batchie.data import Experiment
from scipy.special import logit, expit


def load_data(path: str) -> Experiment:
    data = load_and_preprocess_gdsc_sq(path)

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

    return Experiment(
        observations=X_mean.y_viability.to_numpy().astype(float),
        treatment_names=X_mean[["drug1", "drug2"]].fillna("").to_numpy().astype(str),
        treatment_doses=X_mean[["dose1", "dose2"]].to_numpy().astype(float),
        sample_names=X_mean.sample_ids.to_numpy().astype(str),
        plate_names=X_mean.plate.to_numpy().astype(str),
    )


def load_and_preprocess_gdsc_sq(path, eps=0.01):
    df_orig = pd.read_csv(
        os.path.join(path, "Original_screen_All_tissues_raw_data.csv")
    )
    df_val = pd.read_csv(
        os.path.join(path, "Validation_screen_All_tissues_raw_data.csv")
    )
    df = pd.concat([df_orig, df_val], ignore_index=True)

    df = df[df["TAG"] != "UN-USED"]
    drugid2name = dict(zip(df["DRUG_ID"].values.astype(int), df["DRUG_NAME"].values))

    df_high_controls = df[(df["TAG"] == "NC-1")]
    df_high_controls = df_high_controls.groupby("BARCODE")[
        "INTENSITY"
    ].mean()  ## Maps barcodes to corresponding high_control

    df_low_controls = df[df["TAG"] == "B"]
    df_low_controls = df_low_controls.groupby("BARCODE")[
        "INTENSITY"
    ].mean()  ## Maps barcodes to corresponding low_control

    df = df[
        (df["TAG"] != "B") & (df["TAG"] != "NC-1") & (df["TAG"] != "NC-0")
    ]  ## Eliminate high/low controls
    ## TAG ending with -S indicates single drug control
    ## TAG ending with -C indicates combination

    cline_name = np.unique(df["CELL_LINE_NAME"].values)
    clines2id = {x: i for i, x in enumerate(cline_name)}
    drug_ids = np.unique(df["DRUG_ID"].values.astype(int))
    drugs2id = {x: i for i, x in enumerate(drug_ids)}
    drugs2id[4000] = -1  ## DMSO corresponds to single-drug control
    doses = np.unique(df["CONC"].values)
    doses2id = {x: i for i, x in enumerate(doses)}

    barcode_pos2cell_drugid_conc_intens = {}
    for bcode, pos, cell_name, tag, drugid, conc, intens in zip(
        df["BARCODE"].values,
        df["POSITION"].values,
        df["CELL_LINE_NAME"].values,
        df["TAG"].values,
        df["DRUG_ID"].values.astype(int),
        df["CONC"].values,
        df["INTENSITY"].values,
    ):
        x = barcode_pos2cell_drugid_conc_intens.get((bcode, pos), [])
        x.append((cell_name, tag, drugid, conc, intens))
        barcode_pos2cell_drugid_conc_intens[bcode, pos] = x

    valid_keys = [
        k for k, v in barcode_pos2cell_drugid_conc_intens.items() if len(v) == 2
    ]

    cline = []
    plate = []
    drug1 = []
    drug2 = []
    dose1 = []
    dose2 = []
    concentration1 = []
    concentration2 = []
    y = []
    for bcode, pos in valid_keys:
        (cname1, _, drugid1, conc1, intens1), (
            cname2,
            _,
            drugid2,
            conc2,
            intens2,
        ) = barcode_pos2cell_drugid_conc_intens[bcode, pos]
        if cname1 != cname2:
            print("Error. Cell names don't line up")
        if intens1 != intens2:
            print("Error. Intensity values don't line up")
        ## Set drug2 to be DMSO
        if drugid1 == 4000:
            drugid1 = drugid2
            conc1 = conc2
            drugid2 = 4000
            conc2 = np.nan

        if (
            drugid1 != drugid2
        ):  ## Filter out data where drug1 = drug2 (appears to be a data error)
            cline.append(clines2id[cname1])
            plate.append(bcode)
            drug1.append(drugs2id[drugid1])
            drug2.append(drugs2id[drugid2])
            concentration1.append(conc1)
            concentration2.append(conc2)

            if np.isnan(conc1):
                dose1.append(-1)
            else:
                dose1.append(doses2id[conc1])
            if np.isnan(conc2):
                dose2.append(-1)
            else:
                dose2.append(doses2id[conc2])
            y_val = (intens1 - df_low_controls[bcode]) / (
                df_high_controls[bcode] - df_low_controls[bcode]
            )
            y.append(y_val)

    cline = np.array(cline, dtype=int)
    plate = np.array(plate, dtype=int)
    drug1 = np.array(drug1, dtype=int)
    drug2 = np.array(drug2, dtype=int)
    dose1 = np.array(dose1, dtype=int)
    dose2 = np.array(dose2, dtype=int)
    y = np.array(y, dtype=np.float32)
    y = logit(np.clip(y, eps, (1.0 - eps)))
    plates2id = {x: x for x in np.unique(plate)}

    alldrugdoses = sorted(list(set(list(zip(drug1, dose1)) + list(zip(drug2, dose2)))))
    pairs2ddids = {x: i for i, x in enumerate(alldrugdoses[1:])}
    pairs2ddids[(-1, -1)] = -1
    drugdose1 = np.array(
        [pairs2ddids[drug, dose] for drug, dose in zip(drug1, dose1)], dtype=int
    )
    drugdose2 = np.array(
        [pairs2ddids[drug, dose] for drug, dose in zip(drug2, dose2)], dtype=int
    )

    X = {
        "cline": cline,
        "drugdose1": drugdose1,
        "drugdose2": drugdose2,
        "plate": plate,
        "drug1": drug1,
        "drug2": drug2,
        "dose1": dose1,
        "dose2": dose2,
        "conc1": concentration1,
        "conc2": concentration2,
    }
    X = pd.DataFrame(X)
    output = dict(
        y=y,
        X=X,
        drugdoses2id=pairs2ddids,
        drugs2id=drugs2id,
        drugid2name=drugid2name,
        clines2id=clines2id,
        plates2id=plates2id,
        doses2id=doses2id,
        has_controls=True,
    )
    return output
