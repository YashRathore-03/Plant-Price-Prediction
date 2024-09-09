from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TFSMLayer  # May not be needed
import sklearn.ensemble  # May not be needed

# Load model (adjust based on format)
if "finalmodel.pkl" in globals():
    model = globals()["model"]
else:
    try:
        # Try pickle loading
        with open("finalmodel.pkl", "rb") as f:
            model = pickle.load(f)
    except (pickle.UnpicklingError, FileNotFoundError):
        # If pickle fails, try TensorFlow SavedModel loading
        model = load_model("finalmodel.pkl")

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/submit', methods=["POST"])
def prediction():
    # Extract user input
    state_code = request.form["state_code"]
    district = request.form["district"]
    market = request.form["market"]
    commodity = request.form["commodity"]
    variety = request.form["variety"]
    min_price = int(request.form["min_price"])
    max_price = int(request.form["max_price"])


    x_test = [[state_code, district, market, commodity, variety, min_price, max_price]]
    print(x_test)

    prediction = model.predict(x_test)  # Assuming x_test is appropriate for your model

    return render_template("index.html", prediction_text=prediction)  # Update with actual prediction

if __name__ == "__main__":
    app.run(debug=True)