from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
from joblib import load


name= "www"


app = Flask(__name__, template_folder="template")
model = load('my_rain_model.joblib')
targetEncoder= load('encoder.joblib')
    
print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		# # DATE
		form_data = {}
		data= []
		for key in request.form:
				form_data[key] = request.form[key];
		data.append(form_data)
		data= pd.DataFrame(data)
		data[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']] = targetEncoder.transform(data[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']])
		data= data.reindex(columns= ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir',
															'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
															'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
															'Pressure3pm', 'Temp9am', 'Temp3pm', 'Month', 'Day', 'RainToday_Yes'])
		print(data)
		

		pred = model.predict(data)
		print(pred)
		output = pred
		if output == 0:
			return render_template("after_sunny.html")
		else:
			return render_template("after_rainy.html")
	return render_template("predictor.html")

if __name__=='__main__':
	app.run(debug=True)