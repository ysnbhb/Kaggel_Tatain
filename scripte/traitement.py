import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier , ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from preprocessing_feature import add_features


def candidate_models():
    return [
 
        # ── Logistic Regression ─────────────────────────────────────────────
        # solver="saga" supports all penalties including elasticnet
        # class_weight="balanced" handles 60/40 Titanic imbalance
        (
            "logreg",
            LogisticRegression(
                max_iter=2000,
                solver="saga",
                class_weight="balanced",
                random_state=42,
            ),
            {
                "model__C":        [0.01, 0.1, 1.0, 10],
                "model__penalty":  ["l1", "l2"],
            },
        ),
 
        # ── KNN ────────────────────────────────────────────────────────────
        # Tighter n_neighbors (>11 rarely helps on ~700-row splits)
        # metric replaces p — manhattan often beats euclidean on mixed features
        (
            "knn",
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights":     ["uniform", "distance"],
                "model__metric":      ["euclidean", "manhattan"],
            },
        ),
 
        # ── SVM ────────────────────────────────────────────────────────────
        # Dropped poly kernel (degree not tuned = wasteful)
        # class_weight="balanced" important for imbalanced target
        # Tighter gamma range focused around "scale"
        (
            "svm",
            SVC(probability=True, class_weight="balanced", random_state=42),
            {
                "model__C":      [0.1, 1, 10, 50],
                "model__gamma":  ["scale", 0.01, 0.1],
                "model__kernel": ["rbf", "sigmoid"],
            },
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            {
                "model__n_estimators":      [300, 500],
                "model__max_depth":         [5, 10, 15, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf":  [1, 2],
                "model__max_features":      ["sqrt", "log2"],
            },
        ),
 
        # ── Extra Trees ────────────────────────────────────────────────────
        # NEW — faster than RF, more randomness = less overfit on small data
        # Often matches or beats RF on Titanic with less tuning
        (
            "et",
            ExtraTreesClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            {
                "model__n_estimators":     [300, 500],
                "model__max_depth":        [5, 10, 15],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features":     ["sqrt", "log2"],
            },
        ),
 
        # ── Gradient Boosting ──────────────────────────────────────────────
        # Added min_samples_leaf: prevents overfit on leaf nodes
        # Added max_features: adds randomness, often boosts generalization
        # Removed slow combos (n_estimators=300 + lr=0.01)
        (
            "gb",
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators":      [100, 200],
                "model__learning_rate":     [0.05, 0.1, 0.2],
                "model__max_depth":         [3, 4, 5],
                "model__subsample":         [0.8, 1.0],
                "model__min_samples_leaf":  [1, 5, 10],
                "model__max_features":      ["sqrt", None],
            },
        ),
    ]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["str"]).columns.tolist()

    binary_cols = [c for c in numeric_cols if X[c].nunique() == 2]
    numeric_cols = [c for c in numeric_cols if c not in binary_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", binary_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main():
    print("Loading data...")
    data = pd.read_csv("titanic/train.csv")
    X = data.drop(columns="Survived")
    y = data["Survived"]

    print("\nEngineering features...")
    X = add_features(X)

    missing = X.isna().sum()
    if missing.any():
        print("⚠️  Missing values after feature engineering:")
        print(missing[missing > 0])
    else:
        print("✅  No missing values after feature engineering.")

    print("\nSplitting data (80/20)...")
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_estimator = None
    best_name = None
    best_cv_score = -np.inf
    results = []

    print("\n" + "=" * 60)
    print("STARTING GRID SEARCH (5-fold CV)")
    print("=" * 60)

    for name, model, grid in candidate_models():
        print(f"\n[{name.upper()}] Grid search starting...")

        pre = build_preprocessor(X_train)
        pipe = Pipeline([("pre", pre), ("model", model)])

        gs = GridSearchCV(
            pipe,
            grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)

        cv_score = gs.best_score_
        print(f"[{name.upper()}] Best CV score : {cv_score:.4f}")
        print(f"[{name.upper()}] Best params   : {gs.best_params_}")

        results.append({"model": name, "cv_accuracy": cv_score, "params": gs.best_params_})

        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_estimator = gs.best_estimator_
            best_name = name

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)

    print(f"\n🏆 Best model      : {best_name.upper()}")
    print(f"   Best CV accuracy: {best_cv_score:.4f}")

    # Fix #3: evaluate on holdout set that was previously ignored
    holdout_acc = best_estimator.score(X_holdout, y_holdout)
    print(f"   Holdout accuracy: {holdout_acc:.4f}")

    # Summary table
    print("\n── All models summary ──────────────────────────────")
    summary = pd.DataFrame(results)[["model", "cv_accuracy"]].sort_values(
        "cv_accuracy", ascending=False
    )
    print(summary.to_string(index=False))

    print("\nSaving best model → bestmodel.pkl")
    joblib.dump(best_estimator, "bestmodel.pkl")
    print("✅  Done.")


if __name__ == "__main__":
    main()