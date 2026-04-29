import pandas as pd
import numpy as np


def add_features(data ,is_train=True):
    data = data.copy()  

    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Age_cut"] = pd.qcut(data["Age"], 8, duplicates="drop")

    data["Deck"] = data["Cabin"].str[0].fillna("Unknown")
    data = data.drop(columns="Cabin")

    # Embarked
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # Family
    data["Family_size"] = data["SibSp"] + data["Parch"]
    data["IsAlone"] = (data["Family_size"] == 0).astype(int) 
    data["FamilyCategory"] = pd.cut(
        data["Family_size"],
        bins=[-1, 0, 3, 20],
        labels=["Alone", "Small", "Large"]
    )

    htype = data["Name"].str.split(",")
    data["passenger_type"] = htype.apply(lambda x: x[1].split(".")[0].strip())
    title_map = {
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Dr": "Officer",   "Rev": "Officer",
        "Don": "Royalty",  "Sir": "Royalty", "Jonkheer": "Royalty",
        "Lady": "Royalty", "the Countess": "Royalty",
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    }
    data["passenger_type_grouped"] = data["passenger_type"].map(title_map).fillna("Other")
    data = data.drop(columns="passenger_type") 

    # Fare
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Fare_per_person"] = data["Fare"] / (data["Family_size"] + 1)
    data["Fare_cut"] = pd.qcut(data["Fare"], 4, duplicates="drop",
                                labels=["Low", "Mid", "High", "VeryHigh"])
    if is_train:
        data = data.drop(columns=["Name", "Ticket", "PassengerId"], errors="ignore")

    return data
