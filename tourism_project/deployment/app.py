import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


# Download and load the model
model_path = hf_hub_download(repo_id="rashmicv09/Visit-with-Us-Prediction-Model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them
""")

Age = st.number_input("Age", min_value=0, max_value=70, value=35)
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (min)", min_value=1, max_value=60, value = 30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=5)
NumberOfFollowups = st.number_input("Number of Followup",min_value=1, max_value=3, value=2 )
PreferredPropertyStar = st.number_input("Preferred Property Star",min_value=1, max_value=5, value=4)
NumberOfTrips = st.number_input("Number of Trips",min_value=1, max_value=10, value=5)
Passport = st.number_input("Passport",min_value=0, max_value=1, value=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score",min_value=1, max_value=5, value=4)
OwnCar = st.number_input("Own Car",min_value=0, max_value=1, value=1)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting",min_value=0, max_value=10, value=2)
MonthlyIncome = st.number_input("Monthly Income",min_value=0, max_value=100000, value=50000)


categorical_features = ['TypeofContact','Occupation', 'Gender', 'ProductPitched', 'MaritalStatus','Designation' ]
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("ProductPitched", ["Tour Package", "In-Person Training"])
MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Single", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])



# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,

    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,

}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Purchase" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
