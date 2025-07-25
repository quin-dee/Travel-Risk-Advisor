from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Loading model and dataset
model = joblib.load("model.pkl")
df = pd.read_csv("travel_risk.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    advice = ""
    selected_state = ""
    states = sorted(df["State"].unique())  # dropdown options

    if request.method == "POST":
        selected_state = request.form["state"]
        advice = get_travel_advice(selected_state)

    return render_template("index.html", advice=advice, states=states, selected_state=selected_state)

def get_travel_advice(state_name):
    state_data = df[df["State"].str.lower() == state_name.lower()]
    if state_data.empty:
        return f"<strong>Data for {state_name} not found in the dataset.</strong>"

    features = ['Crime Rate', 'Health Facilities', 'Police Presence',
                'Hospital Access', 'Road Condition', 'Recent Incidents']
    X = state_data[features]
    prediction = model.predict(X)[0]
    risk_map = {0: ("Low", "🟢"), 1: ("Medium", "⚠️"), 2: ("High", "🔴")}
    risk_level, emoji = risk_map.get(prediction, ("Unknown", "❓"))

    values = X.iloc[0].to_dict()
  
    reason = ", ".join([
        key.replace("_", " ").replace("Rate", "").replace("Presence", "").replace("Access", "").strip()
        for key, value in values.items()
        if value < df[key].median()
    ])

    if risk_level == "High":
        return (
            f"{emoji} <strong>{state_name}</strong> has a <strong>High</strong> travel risk.<br>"
            f"⚠️ Main risk factors include: {reason}.<br>"
            f"<strong>We strongly advise against non-essential travel.</strong>"
        )
    elif risk_level == "Medium":
        return (
            f"{emoji} <strong>{state_name}</strong> has a <strong>Moderate</strong> travel risk.<br>"
            f"Possible concerns: {reason}.<br>"
            f"📝 Travel with caution."
        )
    else:
        return (
            f"{emoji} <strong>{state_name}</strong> has a <strong>Low</strong> travel risk.<br>"
            f"✅ Generally safe, but follow standard precautions."
        )

if __name__ == "__main__":
    app.run(debug=True)
