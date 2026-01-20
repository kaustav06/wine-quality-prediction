import pandas as pd
import joblib
import plotly.express as px # pyright: ignore[reportMissingImports]

# Load dataset
df = pd.read_csv("WineQT.csv")

# Load model
model = joblib.load("model.pkl")

# Feature names
features = df.drop(columns=["quality", "Id"]).columns
importances = model.feature_importances_

# DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

# Plotly interactive bar chart
fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance – Wine Quality Prediction",
    color="Importance"
)

# Save interactive HTML
fig.write_html("static/feature_importance.html")

print("Interactive graph saved successfully!")
