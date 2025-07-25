from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model and dataset
model = joblib.load("travel_risk_model.pkl")
df = pd.read_csv("travel_risk.csv")

app = Flask(__name__)

# Function to generate travel advice
def get_travel_advice(state_name):
    state_data = df[df["State"].str.lower() == state_name.lower()]
    if state_data.empty:
        return f"Data for <strong>{state_name}</strong> not found in the dataset."

    features = ['Crime Rate', 'Health Facilities', 'Police Presence',
                'Hospital Access', 'Road Condition', 'Recent Incidents']
    X = state_data[features]
    prediction = model.predict(X)[0]
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    risk_level = risk_map.get(prediction, "Unknown")

    # Determine weak indicators
    values = X.iloc[0].to_dict()
    poor_indicators = [
        key for key, value in values.items() if value < df[key].median()
    ]
    reason = ", ".join(poor_indicators).lower()

    # Advice logic
    if risk_level == "High":
        return (
            f"For <strong>{state_name}</strong>, the predicted travel risk is <strong>High</strong>. "
            f"This is primarily influenced by: {reason}. "
            f"We strongly advise against non-essential travel."
        )
    elif risk_level == "Medium":
        return (
            f"For <strong>{state_name}</strong>, the predicted travel risk is <strong>Medium</strong>. "
            f"This may be due to: {reason}. Travel with caution."
        )
    else:
        return (
            f"For <strong>{state_name}</strong>, the predicted travel risk is <strong>Low</strong>. "
            f"Generally safe, but always observe standard safety precautions."
        )

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def home():
    states = sorted(df["State"].unique())
    advice = None

    if request.method == "POST":
        selected_state = request.form["state"]
        advice = get_travel_advice(selected_state)

    return render_template("index.html", states=states, advice=advice)

# Optional: API route (not used in form, but kept for flexibility)
@app.route("/ask", methods=["POST"])
def ask():
    state = request.form.get("state")
    advice = get_travel_advice(state)
    states = sorted(df["State"].unique())
    return render_template("index.html", states=states, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)
