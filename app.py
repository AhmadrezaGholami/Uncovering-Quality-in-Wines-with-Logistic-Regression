from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from custom_logistic_regression import LogisticRegression_Scratch

app = Flask(__name__)

# Load models
with open('F:/Programming/AI/machine learning_Cafetadris/DS_1/Logestic_Regression_Scratch/WebApp/models/logistic_regression_MCE_model.pkl', 'rb') as f:
    mce_model = pickle.load(f)
with open('F:/Programming/AI/machine learning_Cafetadris/DS_1/Logestic_Regression_Scratch/WebApp/models/logistic_regression_ova_model.pkl', 'rb') as f:
    ova_model = pickle.load(f)
with open('F:/Programming/AI/machine learning_Cafetadris/DS_1/Logestic_Regression_Scratch/WebApp/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Instantiate custom model from scratch
custom_model = LogisticRegression_Scratch()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data['input_data']  # Use 'input_data' instead of 'features'
    model_choice = data['model']

    # Convert input_data to a numpy array and reshape it
    features = np.array(list(input_data.values())).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Select the model
    if model_choice == "logistic_regression_MCE_model":
        model = mce_model
    elif model_choice == "logistic_regression_ova_model":
        model = ova_model
    else:
        return jsonify({"error": "Invalid model selected."}), 400

    # Make prediction
    prediction = model.predict(features_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
