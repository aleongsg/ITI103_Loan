import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# Load the model, scaler, and encoders
model = joblib.load('Random_Forest.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])
if app_mode == 'Home':
    st.title('LOAN PREDICTION :')
    st.image('hipster_loan-1.jpg')
    st.write('App realized by: Nadia Mhadhbi')

elif app_mode == 'Prediction':
    st.image('slider-short-3.jpg')
    st.subheader('Sir/Mme, YOU need to fill all necessary information to get a reply to your loan request!')
    st.sidebar.header("Information about the client:")
    
    Gender = st.sidebar.radio('Gender', label_encoders['Gender'].classes_)
    Married = st.sidebar.radio('Married', label_encoders['Married'].classes_)
    Self_Employed = st.sidebar.radio('Self Employed', label_encoders['Self_Employed'].classes_)
    Dependents = st.sidebar.radio('Dependents', options=['0', '1', '2', '3+'])
    Education = st.sidebar.radio('Education', label_encoders['Education'].classes_)
    ApplicantIncome = st.sidebar.slider('ApplicantIncome', 0, 10000, 0)
    CoapplicantIncome = st.sidebar.slider('CoapplicantIncome', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('LoanAmount in K$', 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox('Loan_Amount_Term', (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History = st.sidebar.radio('Credit_History', (0.0, 1.0))
    Property_Area = st.sidebar.radio('Property_Area', label_encoders['Property_Area'].classes_)

    data = {
        'Gender': label_encoders['Gender'].transform([Gender])[0],
        'Married': label_encoders['Married'].transform([Married])[0],
        'Dependents': int(Dependents.replace('3+', '3')),
        'Education': label_encoders['Education'].transform([Education])[0],
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'Self Employed': label_encoders['Self_Employed'].transform([Self_Employed])[0],
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': label_encoders['Property_Area'].transform([Property_Area])[0],
    }

    feature_list = np.array(list(data.values())).reshape(1, -1)
    scaled_features = scaler.transform(feature_list)

    if st.button("Predict"):
        file_ = open("6m-rain.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
   
        file = open("green-cola-no.gif", "rb")
        contents = file.read()
        data_url_no = base64.b64encode(contents).decode("utf-8")
        file.close()

        prediction = model.predict(scaled_features)
        if prediction[0] == 0:
            st.error('According to our calculations, you will not get the loan from the bank.')
            st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">', unsafe_allow_html=True)
        elif prediction[0] == 1:
            st.success('Congratulations! You will get the loan from the bank.')
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">', unsafe_allow_html=True)
