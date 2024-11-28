import numpy as np
from flask import Flask, request, render_template
import pickle

#create flask app
app=Flask(__name__)

#load the pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    bhk = int(request.form['bhk'])
    size = int(request.form['size'])
    area_type = request.form['area_type']
    area_locality = request.form['area_locality']
    city = request.form['city']
    furnishing_status = request.form['furnishing_status']
    bathroom = int(request.form['bathroom'])
    floor_level = int(request.form['floor_level'])
    total_floor = int(request.form['total_floor'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    #float_features=[float(x) for x in request.form.values()]
    #features=[np.array(float_features)]
    float_features = [bhk, size, area_type, area_locality, city, furnishing_status, bathroom,
                      floor_level, total_floor, month, day]
    final_features = np.array([float_features])

    print("Input values:")
    print(f"BHK: {bhk}, Size: {size}, Area Type: {area_type}, Area Locality: {area_locality}, City: {city}, "
          f"Furnishing Status: {furnishing_status}, Bathroom: {bathroom}",
          f"Floor Level: {floor_level}, Total Floor: {total_floor}",
          f"Month: {month}, Day: {day}")

    prediction = model.predict(final_features)
    formatted_prediction = f"{prediction[0]:.2f}"

    #print(float_features)
            #prediction = pjct_model.predict([float_features])

            # features=[np.array(float_features)]
            # prediction=pjct_model.predict(features)
    return render_template('index.html', prediction_text=f'Estimated House Rent is {formatted_prediction} units')
            #return render_template('index.html', prediction_text='Houserent is {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)