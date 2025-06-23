import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load PIMA dataset
pima = pd.read_csv("Diabetes pima (clinical)/diabetes.csv")
pima_clean = pima[["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"]].copy()
pima_clean["FamilyHistory"] = pima["DiabetesPedigreeFunction"].apply(lambda x: 1 if x > 0.5 else 0)

# Load BRFSS dataset
brfss = pd.read_csv("BFSS datasets (alex)/BFSS datasets.csv", low_memory=False)
brfss_clean = brfss[["Smoker", "HvyAlcoholConsump", "PhysActivity"]].copy()
brfss_clean.columns = ["Smoking", "AlcoholDrinking", "PhysicalActivity"]

# Ensure binary values
brfss_clean["Smoking"] = brfss_clean["Smoking"].apply(lambda x: 1 if x == 1 else 0)
brfss_clean["AlcoholDrinking"] = brfss_clean["AlcoholDrinking"].apply(lambda x: 1 if x == 1 else 0)
brfss_clean["PhysicalActivity"] = brfss_clean["PhysicalActivity"].apply(lambda x: 1 if x == 1 else 0)

# Align rows
min_len = min(len(pima_clean), len(brfss_clean))
pima_clean = pima_clean.iloc[:min_len].reset_index(drop=True)
brfss_clean = brfss_clean.iloc[:min_len].reset_index(drop=True)
pima_clean["Outcome"] = pima["Outcome"].iloc[:min_len].reset_index(drop=True)

# Combine features
combined = pd.concat([pima_clean, brfss_clean], axis=1)

# Train model
X = combined.drop("Outcome", axis=1)
y = combined["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("✅ Hybrid model trained with 10 features and saved as model.pkl")
