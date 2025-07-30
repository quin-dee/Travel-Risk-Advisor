from flask import Flask, request, render_template
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load model (no label encoder needed)
model = joblib.load("risk_model.pkl")

# Load your cleaned dataset
df = pd.read_csv("travel_risk_clean.csv")

# OpenWeatherMap API key
WEATHER_API_KEY = "e2edee7f1c25fac4a1d45e181d1e676d"

# Map state names to capital cities
weather_name_fix = {
    "Abia": "Umuahia", "Adamawa": "Yola", "Akwa Ibom": "Uyo", "Anambra": "Awka",
    "Bauchi": "Bauchi", "Bayelsa": "Yenagoa", "Benue": "Makurdi", "Borno": "Maiduguri",
    "Cross River": "Calabar", "Delta": "Asaba", "Ebonyi": "Abakaliki", "Edo": "Benin City",
    "Ekiti": "Ado Ekiti", "Enugu": "Enugu", "Gombe": "Gombe", "Imo": "Owerri",
    "Jigawa": "Dutse", "Kaduna": "Kaduna", "Kano": "Kano", "Katsina": "Katsina",
    "Kebbi": "Birnin Kebbi", "Kogi": "Lokoja", "Kwara": "Ilorin", "Lagos": "Lagos",
    "Nasarawa": "Lafia", "Niger": "Minna", "Ogun": "Abeokuta", "Ondo": "Akure",
    "Osun": "Osogbo", "Oyo": "Ibadan", "Plateau": "Jos", "FCT (Abuja)": "Abuja"
}

def get_weather(state_name):
    city_name = weather_name_fix.get(state_name, state_name)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name},NG&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "description": data["weather"][0]["description"],
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
    except Exception as e:
        print(f"Weather API error for {state_name}: {e}")
    return None

def generate_story(state, risk_level, weather):
    if weather:
        return (
            f"As you journey into {state}, the sky is {weather['description']} and a steady breeze blows. "
            f"The temperature is {weather['temp']}°C, humidity at {weather['humidity']}%, and winds around {weather['wind_speed']} km/h. "
            f"Currently, {state} faces a {risk_level} travel risk. If you must go, travel cautiously, avoid risky routes, and stay alert."
        )
    else:
        return "Weather data unavailable."

@app.route("/", methods=["GET", "POST"])
def home():
    risk = ""
    story = ""
    weather = {}
    selected_state = ""

    if request.method == "POST":
        selected_state = request.form.get("state")
        if selected_state:
            try:
                # Prepare features for model (drop label column)
                row = df[df["State"] == selected_state].drop(columns=["State", "Risk Level"])
                raw_pred = model.predict(row)[0]
                risk = risk_map.get(raw_pred, "Unknown")
            except Exception as e:
                print(f"Prediction error: {e}")
                risk = "Unavailable"

            # Weather
            weather = get_weather(selected_state)

            # Story
            story = generate_story(selected_state, risk, weather)

    states = sorted(df["State"].unique())
    return render_template("index.html", states=states, risk=risk, story=story, weather=weather, selected_state=selected_state)

if __name__ == "__main__":
    app.run(debug=True)
