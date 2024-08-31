
import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle


# Load model
with open('model_lr-workgroup.pkl', 'rb') as file:
    model, level_encoder = pickle.load(file)

# Streamlit app
st.title("Lung Cancer Prediction")

# Get user input for each variable
age_input = st.slider('Enter Age in years (0 to 125):', min_value=0, max_value=125)
gender_input = st.selectbox('Select sex:', ['MALE', 'FEMALE'])
air_pollution_input = st.slider('Enter Air Pollution in Categorical (0 to 10):', min_value=0, max_value=10)
alcohol_use_input = st.slider('Enter Alcohol Use in Categorical (0 to 10):', min_value=0, max_value=10) 
dust_allergy_input = st.slider('Enter Dust Allergy in Categorical (0 to 10):', min_value=0, max_value=10) 
occupational_hazards_input = st.slider('Enter OccuPational Hazards in Categorical (0 to 10):', min_value=0, max_value=10) 
genetic_risk_input = st.slider('Enter Genetic Risk in Categorical (0 to 10):', min_value=0, max_value=10) 
chronic_lung_disease_input = st.slider('Enter Chronic Lung Disease in Categorical (0 to 10):', min_value=0, max_value=10) 
balanced_diet_input = st.slider('Enter Balanced Diet in Categorical (0 to 10):', min_value=0, max_value=10) 
obesity_input = st.slider('Enter Obesity in Categorical (0 to 10):', min_value=0, max_value=10) 
smoking_input = st.slider('Enter Smoking in Categorical (0 to 10):', min_value=0, max_value=10) 
passive_smoker_input = st.slider('Enter Passive Smoker in Categorical (0 to 10):', min_value=0, max_value=10) 
chest_pain_input = st.slider('Enter Chest Pains in Categorical (0 to 10):', min_value=0, max_value=10) 
coughing_of_blood_input = st.slider('Enter Coughing of Blood in Categorical (0 to 10):', min_value=0, max_value=10) 
fatigue_input = st.slider('Enter Fatigue in Categorical (0 to 10):', min_value=0, max_value=10) 
weight_loss_input = st.slider('Enter Weight Loss in Categorical (0 to 10):', min_value=0, max_value=10) 
shortness_of_breaths_input = st.slider('Enter Shortness of Breaths in Categorical (0 to 10):', min_value=0, max_value=10) 
wheezing_input = st.slider('Enter Wheezing in Categorical (0 to 10):', min_value=0, max_value=10) 
swallowing_difficulty_input = st.slider('Enter Swallowing Difficulty in Categorical (0 to 10):', min_value=0, max_value=10) 
clubbing_of_finger_nails_input= st.slider('Enter Clubbing of Finger Nails in Categorical (0 to 10):', min_value=0, max_value=10) 
frequent_cold_input = st.slider('Enter Frequent Cold in Categorical (0 to 10):', min_value=0, max_value=10) 
dry_cough_input = st.slider('Enter Dry Cough in Categorical (0 to 10):', min_value=0, max_value=10) 
snoring_input = st.slider('Enter Snoring in Categorical (0 to 10):', min_value=0, max_value=10) 

#gender_input = 'MALE'

# Create a DataFrame with user input
if gender_input == 'MALE':
   gender_input = 1
else:
   gender_input = 2 

#print(gender_input)

x_new = pd.DataFrame({
    'Age': [age_input],
    'Gender': [gender_input],
    'Air Pollution': [air_pollution_input],
    'Alcohol use':[alcohol_use_input],
    'Dust Allergy':[dust_allergy_input],
    'OccuPational Hazards':[occupational_hazards_input],
    'Genetic Risk':[genetic_risk_input],
    'chronic Lung Disease':[chronic_lung_disease_input],
    'Balanced Diet':[balanced_diet_input],
    'Obesity':[obesity_input],
    'Smoking':[smoking_input],
    'Passive Smoker':[passive_smoker_input],
    'Chest Pain':[chest_pain_input],
    'Coughing of Blood':[coughing_of_blood_input],
    'Fatigue':[fatigue_input],
    'Weight Loss':[weight_loss_input],
    'Shortness of Breath':[shortness_of_breaths_input],
    'Wheezing':[wheezing_input],
    'Swallowing Difficulty':[swallowing_difficulty_input],
    'Clubbing of Finger Nails':[clubbing_of_finger_nails_input],
    'Frequent Cold':[frequent_cold_input],
    'Dry Cough':[dry_cough_input],
    'Snoring':[snoring_input]
})


# Prediction
y_pred_new = model.predict(x_new)
result = level_encoder.inverse_transform(y_pred_new)

# Display result
st.subheader('Prediction Lung Cancer Result:')
st.write(f'Predicted Level: {result[0]}')
