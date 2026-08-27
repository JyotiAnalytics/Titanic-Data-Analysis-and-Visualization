# ==========================================
# TITANIC SURVIVAL PREDICTION PROJECT
# ==========================================


# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# ==========================================
# 2. LOAD TITANIC DATASET
# ==========================================

df = sns.load_dataset("titanic")

print(df.head())


# ==========================================
# 3. CHECK DATASET SHAPE
# ==========================================

print("Dataset Shape:", df.shape)


# ==========================================
# 4. CHECK DATA TYPES
# ==========================================

print("Data Types:")
print(df.dtypes)


# ==========================================
# 5. CHECK MISSING VALUES
# ==========================================

print("Missing Values:")
print(df.isnull().sum())


# ==========================================
# 6. CHECK DUPLICATE VALUES
# ==========================================

print("Duplicate Rows:", df.duplicated().sum())


# ==========================================
# 7. REMOVE DUPLICATE VALUES
# ==========================================

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# ==========================================
# 8. SELECT IMPORTANT COLUMNS
# ==========================================

df = df[
    [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]
]

print(df.head())


# ==========================================
# 9. HANDLE MISSING VALUES
# ==========================================

df["age"] = df["age"].fillna(
    df["age"].median()
)

df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

print("Missing Values After Cleaning:")
print(df.isnull().sum())


# ==========================================
# 10. STATISTICAL SUMMARY
# ==========================================

print(df.describe())


# ==========================================
# 11. SURVIVAL DISTRIBUTION
# ==========================================

print("Survival Distribution:")
print(df["survived"].value_counts())


# ==========================================
# 12. SAVE DATA MODEL
# ==========================================
df.to_csv("titanic_cleaned.csv")



# ==========================================
# 13. SURVIVAL VISUALIZATION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="survived"
)

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Passenger Count")
plt.savefig("Survival visualization.png")
plt.tight_layout()
plt.show()


# ==========================================
# 14. GENDER VS SURVIVAL
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="sex",
    hue="survived"
)

plt.title("Gender vs Survival")
plt.xlabel("Gender")
plt.ylabel("Passenger Count")
plt.savefig("gender vs survival.png")
plt.tight_layout()
plt.show()


# ==========================================
# 15. PASSENGER CLASS VS SURVIVAL
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="pclass",
    hue="survived"
)

plt.title("Passenger Class vs Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Passenger Count")
plt.savefig(" Passengerclass_vs_survival.png")

plt.tight_layout()
plt.show()


# ==========================================
# 16. AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("Age_distribution.png")
plt.tight_layout()
plt.show()


# ==========================================
# 17. FARE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="fare",
    bins=30,
    kde=True
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.savefig("Fare_distribution.png")
plt.tight_layout()
plt.show()


# ==========================================
# 18. AGE BOXPLOT
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="age"
)

plt.title("Age vs Survival")
plt.savefig("Age_boxplot.png")
plt.tight_layout()
plt.show()


# ==========================================
# 19. FARE BOXPLOT
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="fare"
)

plt.title("Fare vs Survival")
plt.savefig("Fare_boxplot.png")
plt.tight_layout()
plt.show()


# ==========================================
# 20. CORRELATION HEATMAP
# ==========================================

numeric_df = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(9, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Titanic Feature Correlation Heatmap")
plt.savefig("Correlation_Heatmap.png")
plt.tight_layout()
plt.show()
