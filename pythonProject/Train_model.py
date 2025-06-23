import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("diabetes.csv")

# Remove hard-to-measure features and engineer simpler ones
X = df.drop(["Outcome", "SkinThickness", "DiabetesPedigreeFunction"], axis=1)
X["FamilyHistory"] = df["DiabetesPedigreeFunction"].apply(lambda x: 1 if x > 0.5 else 0)

y = df["Outcome"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to file
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved!")
