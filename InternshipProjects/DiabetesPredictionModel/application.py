from flask import Flask, render_template, request
import pickle
import numpy as np
print(pickle.format_version)

# Load the model and scaler
model = pickle.load(open('models/modelForPrediction.pkl', 'rb'))
scaler = pickle.load(open('models/standard_scaler.pkl', 'rb'))

# Initialize Flask app
application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]

        # Standardize the inputs
        standardized_features = scaler.transform([features])

        # Predict using the loaded model
        prediction = model.predict(standardized_features)

        # Interpret the prediction
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        return render_template('index.html', prediction_text=f'The person is {result}.')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
