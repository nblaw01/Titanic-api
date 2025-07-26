import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv("data/train.csv")
print(df.head())

# Missing summary
total_rows = len(df)
missing_summary = pd.DataFrame({
    "Total": df.count(),
    "Missing": df.isnull().sum()
})
missing_summary["Missing %"] = (missing_summary["Missing"] / total_rows) * 100
print(missing_summary)

# Drop missing & keep selected features
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print(df.head())

##########################################################
#                                                        Model Training                                                             #
##########################################################

X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest + RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best parameters found:")
print(random_search.best_params_)
print("\nBest recall score from cross-validation:")
print(random_search.best_score_)

results_df = pd.DataFrame(random_search.cv_results_)
top_results = results_df.sort_values(by="mean_test_score", ascending=False).head(10)
print("\nTop 10 parameter combinations:\n")
print(top_results[["mean_test_score", "std_test_score", "params"]])

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTuned Model Performance on Test Set:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

##########################################################
#                                               Custom Threshold Evaluation                                          #
##########################################################

# Probabilities instead of fixed threshold
y_probs = best_model.predict_proba(X_test)[:, 1]

# Set your own threshold
threshold = 0.4
y_pred_custom = (y_probs >= threshold).astype(int)

print(f"\n--- Custom Threshold = {threshold} ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))

##########################################################
#                                        Exploring High Confidence Survivors                                  #
##########################################################

probs_df = pd.DataFrame({
    "Pclass": X_test["Pclass"].values,
    "Sex": X_test["Sex"].values,
    "Age": X_test["Age"].values,
    "Fare": X_test["Fare"].values,
    "Actual": y_test.values,
    "Predicted_Prob_Survived": y_probs,
    "Predicted_Label": y_pred_custom
})

print(probs_df.head(10))

high_conf_survivors = probs_df[
    (probs_df["Predicted_Prob_Survived"] >= threshold)
    & (probs_df["Predicted_Label"] == 1)
]
high_conf_true_survivors = high_conf_survivors[high_conf_survivors["Actual"] == 1]

print(df.groupby("Pclass")["Fare"].median())

print(f"\nTotal survivors: {df['Survived'].sum()}")
print(f"\nThere are {len(high_conf_survivors)} high-confidence survivors (>= {threshold*100:.0f}%)")
print("\nAverage values of high-confidence survivors:")
print("Means:\n", high_conf_true_survivors[["Pclass", "Sex", "Age", "Fare"]].mean())
print("\nMedians:\n", high_conf_true_survivors[["Pclass", "Sex", "Age", "Fare"]].median())

sns.boxplot(x=high_conf_true_survivors["Fare"])
plt.title("Fare Distribution of High-Confidence Survivors")
plt.xlabel("Fare")
plt.show()

##########################################################
#                                                     Feature Importance                                                       #
##########################################################

importances = best_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importances from Random Forest")
plt.show()

##########################################################
#                                                              ROC Plot                                                                   #
##########################################################

fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

##########################################################
#                                                           Confusion Matrix                                                      #
##########################################################

import seaborn as sns

cm = confusion_matrix(y_test, y_pred_custom)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"Confusion Matrix (Threshold = {threshold})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

##########################################################
#                                                              Save Model                                                              #
##########################################################

model_bundle = {
    "model": best_model,
    "threshold": threshold,
    "features": X.columns.tolist()
}

os.makedirs("saved_models", exist_ok=True)
with open("saved_models/final_model_bundle.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Model and metadata saved.")
