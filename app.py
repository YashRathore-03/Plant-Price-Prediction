from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('finalmodel.pkl', 'rb'))

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extracting input values from the request JSON
    state = data.get('state', 0)
    district = data.get('district', 0)
    market = data.get('market', 0)
    commodity = data.get('commodity', 0)
    variety = data.get('variety', 0)
    min_price = data.get('min_price', 0)
    max_price = data.get('max_price', 0)

    # You can preprocess the inputs here if needed
    input_features = np.array([[
        state, district, market, commodity, variety, min_price, max_price
    ]])

    # Make the prediction using the model
    prediction = model.predict(input_features)[0]  # Assuming it returns a single prediction value

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
