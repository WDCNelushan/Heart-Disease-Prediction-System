import pickle
import pandas as pd
import os

print('cwd', os.getcwd())
print('files', sorted(os.listdir('.')))

with open('heart_disease_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

print('loaded model type:', type(model))
print('model module:', getattr(model, '__module__', None))
print('scaler module:', getattr(scaler, '__module__', None))
print('feature_names_in_ exists:', hasattr(model, 'feature_names_in_'))

input_dict = {
    'Age': 30,
    'Gender': 1,
    'ChestPainType': 0,
    'RestingBp': 120,
    'Cholesterol': 200,
    'FastingBS': 0,
    'RestingECG': 0,
    'MaxHR': 150,
    'ExerciseEngage': 0,
    'ST_Depression': 1.0,
    'ST_Slope': 0,
    'MajorVessels': 0,
    'Thalassemia': 0
}

categorical_columns = ['Gender', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseEngage', 'ST_Slope', 'MajorVessels', 'Thalassemia']
numerical_columns = ['Age', 'RestingBp', 'Cholesterol', 'MaxHR', 'ST_Depression']

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
print('encoded cols:', list(input_encoded.columns))

expected_encoded = model.feature_names_in_
print('expected cols len:', len(expected_encoded))

input_encoded = input_encoded.reindex(columns=expected_encoded, fill_value=0)
print('reindexed cols len:', len(input_encoded.columns))
print('input_encoded shape:', input_encoded.shape)

input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])
print('scaled ok')

pred = model.predict(input_encoded)
print('prediction:', pred)
