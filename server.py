from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib  # For saving/loading the model
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Load CSV data
df = pd.read_csv("weatherHistory.csv")

# Check if the model is already saved
MODEL_FILE = "weather_model.pkl"

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    # Train the model if not found
    X = df[['Apparent Temperature (C)', 'Humidity', 'Visibility (km)', 'Wind Speed (km/h)', 'Pressure (millibars)']]
    y = df['Temperature (C)']

    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, MODEL_FILE)


# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = [[
            float(data["app_temp"]),
            float(data["humidity"]),
            float(data["visibility"]),
            float(data["wind_speed"]),
            float(data["pressure"])
        ]]
        predicted_temp = model.predict(features)[0]
        return jsonify({"predicted_temp": round(predicted_temp, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


# Route to render HTML form
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

