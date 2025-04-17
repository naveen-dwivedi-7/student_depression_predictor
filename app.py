from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('student_depression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample manual encodings – these must match your training data
gender_map = {'Male': 0, 'Female': 1}
city_map = {'Delhi': 0, 'Mumbai': 1, 'Other': 2}
profession_map = {'Student': 0, 'Intern': 1, 'Working': 2}
diet_map = {'Healthy': 0, 'Unhealthy': 1}
degree_map = {'Bachelors': 0, 'Masters': 1, 'Other': 2}
thoughts_map = {'No': 0, 'Yes': 1}
history_map = {'No': 0, 'Yes': 1}

def preprocess_input(data):
    try:
        features = [
            gender_map.get(data['Gender'], 0),
            int(data['Age']),
            city_map.get(data['City'], 2),
            profession_map.get(data['Profession'], 0),
            int(data['AcademicPressure']),
            int(data['WorkPressure']),
            float(data['CGPA']),
            int(data['StudySatisfaction']),
            int(data['JobSatisfaction']),
            int(data['SleepDuration']),
            diet_map.get(data['DietaryHabits'], 0),
            degree_map.get(data['Degree'], 0),
            thoughts_map.get(data['SuicidalThoughts'], 0),
            int(data['WorkStudyHours']),
            int(data['FinancialStress']),
            history_map.get(data['FamilyHistory'], 0)
        ]
        return np.array([features])
    except Exception as e:
        print("Preprocessing error:", e)
        raise ValueError("Invalid input format")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = preprocess_input(data)
        prediction = model.predict(features)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
