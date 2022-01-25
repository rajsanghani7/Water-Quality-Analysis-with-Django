from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pycaret.classification import *

# Create your views here.

def index(request):
    data = pd.read_csv("waterapp/static/data/water_potability.csv")
    data = data.fillna(data.mean())

    print("alis")
    print(data.head())

    # partitioning data into training data and testing data
    X = data.drop('Potability', axis=1)
    Y = data['Potability']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=100)
    train_data = pd.concat([X_train, Y_train], axis=1)

    # Model Training
    classification = setup(data=train_data, target="Potability", silent=True)

    # Model Prediction
    model = create_model("qda")
    predicted = predict_model(model, data=X_test)

    # Checking Accuracy
    Y_test.compare(predicted.Label).size
    Y_test.size
    accuracy = (Y_test.compare(predicted.Label).size / Y_test.size * 100)
    print(accuracy)
    status = 0
    if request.method == "POST":
        ph = request.POST.get("ph")
        hardness = request.POST.get("hardness")
        solids = request.POST.get("solids")
        chloramines = request.POST.get("chloramines")
        sulfate = request.POST.get("sulfate")
        conductivity = request.POST.get("conductivity")
        carbon = request.POST.get("carbon")
        trihalomethanes = request.POST.get("trihalomethanes")
        turbidity = request.POST.get("turbidity")

        data = [{'ph': ph, 'Hardness':hardness, 'Solids':solids, 'Chloramines':chloramines, 'Sulfate':sulfate, 'Conductivity':conductivity,
            'Organic_carbon':carbon, 'Trihalomethanes':trihalomethanes, 'Turbidity':turbidity,}]
        df = pd.DataFrame(data)

        model = create_model("qda")
        predicted = predict_model(model, data=df)
        print(predicted['Label'])
        if predicted.Label[0] == 1:
            status=1
        else:
            print("Not Drinkable")
            status = -1

    return render(request,'index.html', {'status':status})

def form(request):
    return render(request,'form.html')