from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load model and dataset
model = joblib.load("travel_risk_model.pkl")
df = pd.read_csv("travel_risk.csv")

app = Flask(__name__)

def get_travel_advice(state_name):
    state_data = df[df["State"].str.lower() == state_name.lower()]
    if state_data.empty:
        return f"Data for {state_name} not found in the dataset."

    features = ['Crime Rate', 'Health Facilities', 'Police Presence',
                'Hospital Access', 'Road Condition', 'Recent Incidents']
    X = state_data[features]
    prediction = model.predict(X)[0]
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    risk_level = risk_map.get(prediction, "Unknown")
    values = X.iloc[0].to_dict()
    reason = ", ".join([
        f"{key.lower()} ({value})"
        for key, value in values.items()
        if value < df[key].median()
    ])

    if risk_level == "High":
        return (
            f"For **{state_name}**, the predicted travel risk is **High**. "
            f"This is primarily influenced by: {reason}. "
            f"We strongly advise against non-essential travel."
        )
    elif risk_level == "Medium":
        return (
            f"For **{state_name}**, the predicted travel risk is **Medium**. "
            f"This may be due to: {reason}. Travel with caution."
        )
    else:
        return (
            f"For **{state_name}**, the predicted travel risk is **Low**. "
            f"Generally safe, but always observe standard safety precautions."
        )

# Single route to handle both GET and POST
@app.route("/", methods=["GET", "POST"])
def home():
    states = sorted(df["State"].unique())
    advice = None

    if request.method == "POST":
        selected_state = request.form.get("state")
        advice = get_travel_advice(selected_state)

    return render_template("index.html", states=states, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)
