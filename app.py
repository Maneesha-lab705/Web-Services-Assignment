from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load models
svm_model = joblib.load("svm_model.pkl")
logreg_model = joblib.load("logreg_model.pkl")

def prepare_input(data):
    """Format input for prediction"""
    return pd.DataFrame([{
        "age": int(data["age"]),
        "job": data["job"],
        "marital": data["marital"],
        "education": data["education"],
        "default": data["default"],
        "housing": data["housing"],
        "loan": data["loan"],
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 0,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.1,
        "euribor3m": 4.857,
        "nr.employed": 5191.0
    }])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/svm')
def svm_page():
    return render_template('svm.html')

@app.route('/logreg')
def logreg_page():
    return render_template('logreg.html')

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON input"}), 400

        input_df = prepare_input(data)
        prediction = svm_model.predict(input_df)[0]
        proba = svm_model.predict_proba(input_df)[0][1]

        return jsonify({
            "model": "SVM",
            "prediction": "Yes" if prediction == 1 else "No",
            "confidence": round(proba, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/logreg', methods=['POST'])
def predict_logreg():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON input"}), 400

        input_df = prepare_input(data)
        prediction = logreg_model.predict(input_df)[0]
        proba = logreg_model.predict_proba(input_df)[0][1]

        return jsonify({
            "model": "Logistic Regression",
            "prediction": "Yes" if prediction == 1 else "No",
            "confidence": round(proba, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

