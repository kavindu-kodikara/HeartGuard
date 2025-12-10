from flask import Flask,render_template,request
import joblib
import numpy as np
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('patient_details.html')
@app.route('/getresults',methods=['POST'])
def getresults():

    model = joblib.load('models/heart_model.sav')
    result = request.form

    print(result)

    name = result['name']
    gender = float(result['gender'])
    age = float(result['age'])
    tc = float(result['tc'])
    hdl = float(result['hdl'])
    smoke = float(result['smoke'])
    bpm = float(result['bpm'])
    diab = float(result['diab'])

    test_data = np.array([gender, age, tc, hdl, smoke, bpm, diab]).reshape(1, -1)

    prediction = model.predict(test_data)

    resultDict = {"name":name, "risk":round(prediction[0])}

    return (render_template('patient_results.html',results = resultDict))

app.run(debug=True)