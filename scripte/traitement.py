import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from preprocessing_feature import add_features


def candidate_models():

    return [
        # Logistic Regression
        (
            "logreg",
            LogisticRegression(max_iter=2000, solver="liblinear"),
            {
                "model__C": [0.001, 0.01, 0.1, 1.0, 10, 100],
                "model__penalty": ["l1", "l2"],
            },
        ),
        # KNN
        (
            "knn",
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],  # Manhattan vs Euclidean
            },
        ),
        # SVM
        (
            "svm",
            SVC(probability=True),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1],
                "model__kernel": ["rbf", "poly"],
            },
        ),
        # Random Forest
        (
            "rf",
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
        # Gradient Boosting
        (
            "gb",
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
            },
        ),
    ]


def main():
    print("Loading data...")
    data = pd.read_csv("titanic/train.csv")
    X = data.drop(columns="Survived")
    y = data["Survived"]

    print("\nEngineering features...")
    X = add_features(X)
    print("value nan ===> ", X.isna().sum())
    print("\nSplitting data (80/20)...")
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best = None
    best_name = None
    best_score = -np.inf

    print("\n" + "=" * 60)
    print("STARTING GRID SEARCH (5-fold CV)")
    print("=" * 60)

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    binary_cols = [c for c in numeric_cols if X[c].nunique() == 2]
    numeric_cols = [c for c in numeric_cols if c not in binary_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", binary_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


    print("Loading test data...")
    test = pd.read_csv("titanic/test.csv")

    print("Engineering features...")
    X_test = add_features(test)

    
    for name, model, grid in candidate_models():
        print(f"\n[{name.upper()}] Grid search starting...")
        pipe = Pipeline([("pre", pre), ("model", model)])

        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

        gs.fit(X_train, y_train)

        print(f"[{name.upper()}] Best CV score: {gs.best_score_:.4f}")
        print(f"[{name.upper()}] Best params: {gs.best_params_}")
        
        print("Predicting...")
        preds = gs.best_estimator_.predict(X_test)
        out = pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": preds})
        out.to_csv(f"{name}.csv", index=False)

        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best = gs.best_estimator_
            best_name = name

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)

    print(f"\nBest model: {best_name.upper()}")
    print(f"Best CV accuracy: {best_score:.4f}")

    print("\nSaving model and results...")
    joblib.dump(best, "bestmodel.pkl")


if __name__ == "__main__":
    main()
