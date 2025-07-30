from flask import Flask, request, render_template
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("risk_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load the reference dataset to fetch state info
df = pd.read_csv("travel_risk_clean.csv")  

# Weather API config
WEATHER_API_KEY = "e2edee7f1c25fac4a1d45e181d1e676d"

def get_weather(state_name):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={state_name},NG&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = {
            "description": data["weather"][0]["description"],
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather
    else:
        return None

def generate_story(state, risk_level, weather):
    return (
        f"As you journey into {state}, the sky is {weather['description']} and a steady breeze blows. "
        f"The temperature is {weather['temp']}°C, humidity at {weather['humidity']}%, and winds around {weather['wind_speed']} km/h. "
        f"Currently, {state} faces a **{risk_level}** travel risk. If you must go, travel cautiously, avoid risky routes, and stay alert."
    )

@app.route("/", methods=["GET", "POST"])
def home():
    story = ""
    weather = {}
    risk = ""
    selected_state = ""

    if request.method == "POST":
        selected_state = request.form["state"]
        state_data = df[df["State"] == selected_state].drop(columns=["State", "Risk Level"])
        
        pred = model.predict(state_data)[0]
        risk = label_encoder.inverse_transform([pred])[0]

        weather = get_weather(selected_state)
        story = generate_story(selected_state, risk, weather) if weather else "Weather data unavailable."

    states = df["State"].unique()
    return render_template("index.html", states=states, risk=risk, story=story, weather=weather, selected_state=selected_state)

if __name__ == "__main__":
    app.run(debug=True)
