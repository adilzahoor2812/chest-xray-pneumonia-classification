"""Prepare NIH ChestX-ray14 CSVs with stratified patient-level splits."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, RANDOM_SEED, SOURCE_CSV, TEST_CSV, TRAIN_CSV, VAL_CSV


def _patient_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with pneumonia-positive flag for stratification."""
    patient_labels = (
        df.groupby("Patient ID")["label"]
        .max()
        .rename("patient_label")
        .reset_index()
    )
    return patient_labels


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Could not find: {SOURCE_CSV}\n"
            "Download NIH ChestX-ray14, place Data_Entry_2017.csv and image "
            "folders under data/, then re-run this script."
        )

    df = pd.read_csv(SOURCE_CSV)

    required_columns = {"Finding Labels", "Patient ID", "Image Index"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    pneumonia_mask = df["Finding Labels"].str.contains(
        "Pneumonia",
        regex=False,
        na=False,
    )
    no_finding_mask = df["Finding Labels"].eq("No Finding")
    df = df[pneumonia_mask | no_finding_mask].copy()
    df["label"] = pneumonia_mask.loc[df.index].astype(int)

    print("Usable images:")
    print(df["label"].value_counts().rename(index={0: "No Finding", 1: "Pneumonia"}))

    patient_frame = _patient_label_frame(df)
    patient_ids = patient_frame["Patient ID"].to_numpy()
    stratify_labels = patient_frame["patient_label"].to_numpy()

    try:
        train_patients, remaining_patients, _, remaining_labels = train_test_split(
            patient_ids,
            stratify_labels,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=stratify_labels,
        )
        val_patients, test_patients = train_test_split(
            remaining_patients,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=remaining_labels,
        )
        print("\nUsed stratified patient-level splits.")
    except ValueError as error:
        print(f"\nStratified split failed ({error}); falling back to unstratified.")
        train_patients, remaining_patients = train_test_split(
            patient_ids,
            test_size=0.30,
            random_state=RANDOM_SEED,
        )
        val_patients, test_patients = train_test_split(
            remaining_patients,
            test_size=0.50,
            random_state=RANDOM_SEED,
        )

    train_df = df[df["Patient ID"].isin(train_patients)].copy()
    val_df = df[df["Patient ID"].isin(val_patients)].copy()
    test_df = df[df["Patient ID"].isin(test_patients)].copy()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    assert set(train_patients).isdisjoint(val_patients)
    assert set(train_patients).isdisjoint(test_patients)
    assert set(val_patients).isdisjoint(test_patients)

    for name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df),
    ]:
        print(f"\n{name} images: {len(split_df)}")
        print(
            split_df["label"]
            .value_counts()
            .rename(index={0: "No Finding", 1: "Pneumonia"})
        )

    print("\nNo patient overlap found.")
    print(f"Wrote {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")


if __name__ == "__main__":
    main()
