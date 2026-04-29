import pandas as pd
import numpy as np


def add_features(data):
    data["Age"] = data["Age"].fillna(data["Age"].mean())

    data["Age_cut"] = pd.qcut(data["Age"], 8, duplicates="drop")

    data["Cabin"] = data["Cabin"].fillna("O")
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    data["Family_size"] = data["SibSp"] + data["Parch"]

    htype = data["Name"].str.split(",")
    data["passenger_type"] = htype.apply(lambda x: x[1].split(".")[0].strip())

    title_map = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Jonkheer": "Royalty",
        "Lady": "Royalty",
        "the Countess": "Royalty",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    data["passenger_type_grouped"] = data["passenger_type"].replace(title_map)

    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

    return data
