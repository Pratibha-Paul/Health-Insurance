


from django.shortcuts import render
import joblib
import numpy as np


model = joblib.load("predictor/insurance_model.pkl")
columns = joblib.load("predictor/model_columns.pkl")

def predict_cost(request):
    if request.method == "POST":
        
        age = int(request.POST["age"])
        gender = request.POST.get("gender")
        bmi = float(request.POST["bmi"])
        children = int(request.POST["children"])
        smoker = request.POST["smoker"]
        region = request.POST["region"]
        input_data = np.zeros(len(columns))  

        input_data[0] = age
        input_data[1] = bmi
        input_data[2] = children

        
        if gender == "male":
            input_data[3] = 1
        if smoker == "yes":
            input_data[4] = 1
        if "region_northwest" in columns and region == "northwest":
            input_data[5] = 1
        if "region_southeast" in columns and region == "southeast":
            input_data[6] = 1
        if "region_southwest" in columns and region == "southwest":
            input_data[7] = 1

        
        prediction = model.predict([input_data])[0]

        return render(request, "result.html", {"prediction": prediction})
    return render(request,"index.html")