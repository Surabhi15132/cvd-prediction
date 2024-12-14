from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Load the trained model
model = joblib.load('models/cvd_model.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the user inputs from the form
        age = float(request.form["age"])
        gender = int(request.form["gender"])
        restingBP = float(request.form["restingBP"])
        serumcholesterol = float(request.form["serumcholesterol"])
        fastingbloodsugar = int(request.form["fastingbloodsugar"])
        maxheartrate = float(request.form["maxheartrate"])
        oldpeak = float(request.form["oldpeak"])
        noofmajorvessels = float(request.form["noofmajorvessels"])
        chestpain = int(request.form["chestpain"])  # User input for chest pain (0-3)
        restingrelectro = int(request.form["restingelectro"]) # User input for restingrelectro (0-2)
        slope = int(request.form["slope"]) # User input for slope (0-3)

        # Convert chest pain value to binary format
        chestpain_0 = 1 if chestpain == 0 else 0
        chestpain_1 = 1 if chestpain == 1 else 0
        chestpain_2 = 1 if chestpain == 2 else 0
        chestpain_3 = 1 if chestpain == 3 else 0

        # Convert restingrelectro value to binary format
        restingrelectro_0 = 1 if restingrelectro == 0 else 0
        restingrelectro_1 = 1 if restingrelectro == 1 else 0
        restingrelectro_2 = 1 if restingrelectro == 2 else 0

        # Convert slope value to binary format
        slope_0 = 1 if slope == 0 else 0
        slope_1 = 1 if slope == 1 else 0
        slope_2 = 1 if slope == 2 else 0
        slope_3 = 1 if slope == 3 else 0

        # Prepare the input data for prediction
        input_data = np.array([[age, gender, restingBP, serumcholesterol, fastingbloodsugar,
                               maxheartrate, oldpeak, noofmajorvessels, chestpain_0, chestpain_1,
                               chestpain_2, chestpain_3, restingrelectro_0, restingrelectro_1,
                               restingrelectro_2, slope_0, slope_1, slope_2, slope_3]])

        # Make prediction using the trained model
        prediction = model.predict(input_data)

        # Calculate the probability of the presence of heart disease
        # probability = model.predict_proba(input_data)[:, 1][0]

        # Prepare the response message
        if prediction == 1:
            result = f"Heart disease detected! Please consult a healthcare professional for further guidance."
        else:
            result = "No heart disease detected. Keep up the healthy lifestyle!"

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
