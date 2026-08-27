

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

df=pd.read_csv(r"Cleaned_Data\titanic_cleaned.csv")

# ==========================================
# 1. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=["survived"]
)

y = df["survived"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# ==========================================
# 2. ENCODE CATEGORICAL FEATURES
# ==========================================

X = pd.get_dummies(
    X,
    columns=["sex", "embarked"],
    drop_first=True
)

print("Encoded Features:")
print(X.head())


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 4. FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("Feature Scaling Completed")


# ==========================================
# 5. LOGISTIC REGRESSION
# ==========================================

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)


# ==========================================
# 6. K NEAREST NEIGHBORS
# ==========================================

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_scaled,
    y_train
)

knn_pred = knn_model.predict(
    X_test_scaled
)


# ==========================================
# 7. DECISION TREE
# ==========================================

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_pred = dt_model.predict(
    X_test
)


# ==========================================
# 8. RANDOM FOREST
# ==========================================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)


# ==========================================
# 9. SUPPORT VECTOR MACHINE
# ==========================================

from sklearn.svm import SVC

svm_model = SVC()

svm_model.fit(
    X_train_scaled,
    y_train
)

svm_pred = svm_model.predict(
    X_test_scaled
)


# ==========================================
# 10. MODEL ACCURACY
# ==========================================

from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": logistic_pred,
    "KNN": knn_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "SVM": svm_pred
}

results = []

for name, prediction in models.items():

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

results_df = pd.DataFrame(results)

print("\nModel Accuracy:")
print(results_df)


# ==========================================
# 11. MODEL ACCURACY VISUALIZATION
# ==========================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model"
)

plt.title("Titanic Model Accuracy Comparison")

plt.xlabel("Accuracy")
plt.ylabel("Model")

plt.xlim(0, 1)
plt.savefig("Model_accuracy_visualization.png")
plt.tight_layout()
plt.show()


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

from sklearn.metrics import classification_report

print("Random Forest Classification Report:")

print(
    classification_report(
        y_test,
        rf_pred
    )
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"]
)
plt.savefig("Confusion_matrix.png")
plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ==========================================
# 14. FIND BEST MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = results_df["Accuracy"].max()

print("Best Model:", best_model_name)

print("Best Accuracy:", best_accuracy)


# ==========================================
# 15. SAVE BEST MODEL
# ==========================================

# ==========================================
# SAVE BEST MODEL
# ==========================================
import joblib

joblib.dump(rf_model, "titanic_best_model.pkl")

print("Model Saved Successfully!")

# ==========================================
# 16. SAVE SCALER
# ==========================================

joblib.dump(scaler, "titanic_scaler.pkl")

print("Scaler Saved Successfully!")

# ==========================================
# 17. PREDICT NEW PASSENGER
# ==========================================

new_passenger = pd.DataFrame({
    "pclass": [3],
    "sex": ["male"],
    "age": [25],
    "sibsp": [0],
    "parch": [0],
    "fare": [10],
    "embarked": ["S"]
})

# Encode new passenger

new_passenger = pd.get_dummies(
    new_passenger,
    columns=["sex", "embarked"],
    drop_first=True
)

# Match training columns

new_passenger = new_passenger.reindex(
    columns=X.columns,
    fill_value=0
)


# ==========================================
# 18. PREDICT SURVIVAL
# ==========================================

# ==========================================
# 18. PREDICT SURVIVAL
# ==========================================

prediction = rf_model.predict(
    new_passenger
)

if prediction[0] == 1:
    print("Predicted Result: Passenger Survived")
else:
    print("Predicted Result: Passenger Did Not Survive")
# ==========================================
# 19. SAVE CLEANED DATASET
# ==========================================

df.to_csv("../data/titanic_cleaned.csv", index=False)

print("Cleaned Dataset Saved Successfully!")

# ==========================================
# 20. FINAL PROJECT INFORMATION
# ==========================================

print("\n====================================")
print("TITANIC SURVIVAL PREDICTION PROJECT")
print("====================================")

print("Best Model:", best_model_name)

print("Best Accuracy:", best_accuracy)

print("Project Completed Successfully!")