import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --------------------------- Load & Prepare Data --------------------------- #
df = pd.read_csv("data/train.csv")  

# Check missing values
missing_summary = pd.DataFrame({
    "Total": df.count(),
    "Missing": df.isnull().sum()
})
missing_summary["Missing %"] = (missing_summary["Missing"] / len(df)) * 100
print(missing_summary)

# Clean dataset
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# --------------------------- Split Data --------------------------- #
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------- Define Models --------------------------- #
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Fit base models
rf.fit(X_train, y_train)
logreg.fit(X_train, y_train)
gb.fit(X_train, y_train)

# --------------------------- Find Best Thresholds Per Model --------------------------- #
def find_best_threshold(model, X_test, y_test, metric=f1_score):
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.3, 0.70, 0.01)
    best_thresh = 0.5
    best_score = 0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = metric(y_test, preds)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score

custom_thresholds = {}
for name, model in [("Random Forest", rf), ("Logistic Regression", logreg), ("Gradient Boosting", gb)]:
    best_t, best_s = find_best_threshold(model, X_test, y_test, metric=f1_score)
    custom_thresholds[name] = best_t
    print(f"{name} best threshold = {best_t:.2f} | Best F1 = {best_s:.3f}")

# --------------------------- Compare Individual Model Performance --------------------------- #
models = {
    "Random Forest": rf,
    "Logistic Regression": logreg,
    "Gradient Boosting": gb
}

print("\n--- Individual Model Performance (Using Custom Thresholds) ---")
for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    threshold = custom_thresholds[name]
    preds = (probs >= threshold).astype(int)
    print(f"\n{name} (Threshold = {threshold:.2f})")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"F1 Score: {f1_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall: {recall_score(y_test, preds):.3f}")

# --------------------------- Ensemble Model --------------------------- #
ensemble_model = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("logreg", logreg),
        ("gb", gb)
    ],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

# --------------------------- Predict Probabilities --------------------------- #
best_t_ensemble, best_f1_ensemble = find_best_threshold(ensemble_model, X_test, y_test)
print(f"\nEnsemble Model best threshold = {best_t_ensemble:.2f} | Best F1 = {best_f1_ensemble:.3f}")
threshold = best_t_ensemble # This is only used for the emsemble model

y_probs_ensemble = ensemble_model.predict_proba(X_test)[:, 1]
y_probs_rf = rf.predict_proba(X_test)[:, 1]

y_pred_ensemble = (y_probs_ensemble >= threshold).astype(int)
y_pred_rf = (y_probs_rf >= threshold).astype(int)

# --------------------------- Compare Performance --------------------------- #
print("\n--- Model Comparison ---")

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nEnsemble Model:")
print(classification_report(y_test, y_pred_ensemble))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))

# --------------------------- Save Model --------------------------- #
model_bundle = {
    "model": ensemble_model,
    "threshold": threshold,
    "features": X.columns.tolist()
}
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/final_model_bundle.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Model saved.")