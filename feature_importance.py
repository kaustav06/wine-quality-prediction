import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("WineQT.csv")

# Load trained model
model = joblib.load("model.pkl")

# Feature names
features = df.drop(columns=["quality", "Id"]).columns

# Feature importance
importances = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance - Wine Quality Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")

# Save plot
plt.savefig("static/feature_importance.png", bbox_inches="tight")
plt.show()
