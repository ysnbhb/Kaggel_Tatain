import joblib
import pandas as pd
from preprocessing_feature import add_features

def main():
    print("Loading trained model...")
    model = joblib.load("bestmodel.pkl")

    print("Loading test data...")
    test = pd.read_csv("titanic./test.csv")

    print("Engineering features...")
    X_test = add_features(test)
    print("value nan ===> ", X_test.isna().sum())

    print("Predicting...")
    preds = model.predict(X_test)

    print("Saving predictions...")
    out = pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": preds})
    out.to_csv("testpredictions.csv", index=False)

    print(f"✅ Predictions saved: testpredictions.csv")
    print(f"   Total predictions: {len(preds)}")


if __name__ == "__main__":
    main()
