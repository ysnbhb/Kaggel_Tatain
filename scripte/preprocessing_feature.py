import pandas as pd
import numpy as np


def add_features(data):
    data = data.copy()


    data["passenger_type"] = (
        data["Name"]
        .str.split(",")
        .apply(lambda x: x[1].split(".")[0].strip())
    )
    title_map = {
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Dr": "Officer",   "Rev": "Officer",
        "Don": "Royalty",  "Sir": "Royalty", "Jonkheer": "Royalty",
        "Lady": "Royalty", "the Countess": "Royalty",
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    }
    data["passenger_type_grouped"] = data["passenger_type"].replace(title_map)

    age_median = (
        data.groupby(["passenger_type_grouped", "Pclass"])["Age"]
        .transform("median")
    )
    data["Age"] = data["Age"].fillna(age_median)
    data["Age"] = data["Age"].fillna(data["Age"].median())   # final fallback

    data["Age_cut"] = pd.qcut(data["Age"], 8, duplicates="drop")

    data["IsChild"] = (data["Age"] < 12).astype(int)


    data["Deck"] = data["Cabin"].str[0].fillna("U")   # U = Unknown
    data = data.drop(columns="Cabin")

    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # ── 5. FAMILY features ──────────────────────────────────────────────────
    data["Family_size"] = data["SibSp"] + data["Parch"]
    data["IsAlone"]     = (data["Family_size"] == 0).astype(int)   # 0 = alone

    data["FamilyCategory"] = pd.cut(
        data["Family_size"],
        bins=[-1, 0, 3, 20],
        labels=["Alone", "Small", "Large"],
    )

    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    data["Fare_per_person"] = data["Fare"] / (data["Family_size"] + 1)

    # Log-fare compresses right-skew and helps linear/SVM models
    data["Fare_log"] = np.log1p(data["Fare"])

    data["Fare_cut"] = pd.qcut(
        data["Fare"], 4, duplicates="drop",
        labels=["Low", "Mid", "High", "VeryHigh"],
    )

    data["Sex_Pclass"] = data["Sex"].astype(str) + "_" + data["Pclass"].astype(str)

    data["WomanOrChild"] = (
        (data["Sex"] == "female") | (data["IsChild"] == 1)
    ).astype(int)

    # data = data.drop(
    #     columns=["Name", "Ticket", "PassengerId", "passenger_type"],
    #     errors="ignore",
    # )

    return data