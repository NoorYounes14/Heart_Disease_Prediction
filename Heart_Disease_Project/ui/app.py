from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(_name_)

model = joblib.load("../models/random_forest_baseline.pkl")

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction</title>
</head>
<body>
    <h2>Heart Disease Prediction</h2>
    <form method="POST">
        {% for f in features %}
            <label>{{f}}:</label>
            <input type="text" name="{{f}}" required><br>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
        <h3>Prediction: {{prediction}}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    features = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    if request.method == "POST":
        values = [float(request.form[f]) for f in features]
        arr = np.array(values).reshape(1,-1)
        prediction = model.predict(arr)[0]
    return render_template_string(html, features=features, prediction=prediction)

if _name_ == "_main_":
    app.run(debug=True)