from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from file
with open('rf_regressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        # Prepare input features for prediction
        features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                              free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

        # Make prediction using the trained model
        prediction = model.predict(features)

        # Return the result
        return render_template('index.html', prediction_text='Predicted class of wine: {}'.format(prediction[0]))

    except Exception as e:
        return render_template('index.html', prediction_text="Error in prediction: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)