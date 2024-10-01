from flask import Flask, render_template, request
import pickle
import numpy as np

# For TensorFlow model loading (if it's a TensorFlow model)
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None  # TensorFlow might not be installed if you're using a different model

app = Flask(__name__)

# Try to load the model as a Pickle (scikit-learn or any pickle model)
model = None
try:
    with open("finalmodel.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded using pickle.")
except Exception as e:
    print(f"Error loading model with pickle: {e}")
    
    # If pickle fails, try loading as a TensorFlow model
    if load_model is not None:
        try:
            model = load_model("finalmodel.pkl")
            print("Model loaded using TensorFlow.")
        except Exception as tf_e:
            print(f"Error loading model with TensorFlow: {tf_e}")

# Ensure model is loaded successfully
if model is None:
    raise Exception("Failed to load the model. Please check the model format and file path.")

@app.route('/')
def load_page():
    return render_template("index.html")

@app.route('/submit', methods=["POST"])
def prediction():
    try:
        # Get form input
        state_code = request.form["state_code"]
        district = request.form["district"]
        market = request.form["market"]
        commodity = request.form["commodity"]
        variety = request.form["variety"]
        min_price = int(request.form["min_price"])
        max_price = int(request.form["max_price"])

        # Prepare data for prediction
        x_test = [[state_code, district, market, commodity, variety, min_price, max_price]]
        print(f"Input data: {x_test}")

        # Perform prediction
        prediction = model.predict(x_test)
        print(f"Prediction: {prediction}")

        # Display the result
        return render_template("index.html", prediction_text=f"Predicted Price: {prediction}")

    except Exception as e:
        # Display any error that occurs
        return render_template("index.html", prediction_text=f"Error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
