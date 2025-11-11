import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Multi-Disease Predictor", layout="wide")

# --- LOAD SAVED MODELS AND SCALERS ---
try:
    # Diabetes
    with open('diabetes_model.pkl', 'rb') as f:
        diabetes_model = pickle.load(f)
    with open('diabetes_scaler.pkl', 'rb') as f:
        diabetes_scaler = pickle.load(f)

    # Heart Disease
    with open('heart_model.pkl', 'rb') as f:
        heart_model = pickle.load(f)
    with open('heart_scaler.pkl', 'rb') as f:
        heart_scaler = pickle.load(f)

    # Parkinson's
    with open('parkinsons_model.pkl', 'rb') as f:
        parkinsons_model = pickle.load(f)
    with open('parkinsons_scaler.pkl', 'rb') as f:
        parkinsons_scaler = pickle.load(f)

except FileNotFoundError:
    st.error("Model or scaler files not found. Please run the training scripts first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models or scalers: {e}")
    st.stop()


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Disease Prediction Menu")
selection = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Diabetes Prediction",
        "Heart Disease Prediction",
        "Parkinson's Prediction"
    ]
)

# --- HOME PAGE ---
if selection == "Home":
    st.title("Welcome to the Multi-Disease Prediction System")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*bcfb-k-t2G212c5b_1K-hw.png", use_column_width=True)
    st.write("""
    This application uses machine learning to predict your risk for several common diseases.
    Please select a disease from the sidebar to get started.
    
    **Disclaimer:** This tool is for educational purposes only and is not a substitute for
    professional medical advice, diagnosis, or treatment. Always seek the advice of your
    physician or other qualified health provider with any questions you may have regarding
    a medical condition.
    """)

# --- DIABETES PREDICTION PAGE ---
elif selection == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    st.write("Enter the following details to predict diabetes risk:")

    # Input fields in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=99.0, step=0.1)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.001, format="%.3f")
    
    with col2:
        Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=200.0, step=0.1)
        Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=850.0, step=0.1)
        Age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)

    with col3:
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=122.0, step=0.1)
        BMI = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=67.0, step=0.1)

    # Prediction button
    if st.button("Predict Diabetes Risk", key="diabetes_predict"):
        # 1. Collect data
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        
        # 2. Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # 3. Standardize the input
        std_data = diabetes_scaler.transform(input_data_reshaped)
        
        # 4. Make prediction
        prediction = diabetes_model.predict(std_data)
        
        # 5. Display result
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.success("You are at low risk for Diabetes.")
        else:
            st.error("You are at high risk for Diabetes. Please consult a doctor.")


# --- HEART DISEASE PREDICTION PAGE ---
elif selection == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    st.write("Enter the following details to predict heart disease risk:")

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        sex = st.selectbox("Sex", ("Male", "Female"))
        cp = st.selectbox("Chest Pain Type (cp)", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=200, step=1)
    
    with col2:
        chol = st.number_input("Serum Cholestoral (chol)", min_value=0, max_value=600, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ("True", "False"))
        restecg = st.selectbox("Resting ECG Results (restecg)", ("Normal", "ST-T wave abnormality", "Probable or definite left ventricular hypertrophy"))
        thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=0, max_value=220, step=1)
    
    with col3:
        exang = st.selectbox("Exercise Induced Angina (exang)", ("Yes", "No"))
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", ("Upsloping", "Flat", "Downsloping"))
        ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, step=1)
        thal = st.selectbox("Thalassemia (thal)", ("Normal", "Fixed defect", "Reversable defect"))

    # Prediction button
    if st.button("Predict Heart Disease Risk", key="heart_predict"):
        
        # Map categorical inputs to numbers
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "True" else 0
        exang = 1 if exang == "Yes" else 0
        
        cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp = cp_mapping[cp]
        
        restecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Probable or definite left ventricular hypertrophy": 2}
        restecg = restecg_mapping[restecg]
        
        slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope = slope_mapping[slope]
        
        thal_mapping = {"Normal": 1, "Fixed defect": 2, "Reversable defect": 3} # Note: 0 is not used in dataset
        thal = thal_mapping[thal]

        # 1. Collect data
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        # 2. Convert and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # 3. Standardize
        std_data = heart_scaler.transform(input_data_reshaped)
        
        # 4. Predict
        prediction = heart_model.predict(std_data)
        
        # 5. Display
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.success("You are at low risk for Heart Disease.")
        else:
            st.error("You are at high risk for Heart Disease. Please consult a doctor.")


# --- PARKINSON'S PREDICTION PAGE ---
elif selection == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")
    st.write("Enter the following voice measurement details:")

    # This dataset has many features, so we use more columns
    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.3f")
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, format="%.3f")
        flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, format="%.3f")
        jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.5f")
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, format="%.7f")
        rap = st.number_input("MDVP:RAP", min_value=0.0, format="%.5f")
        ppq = st.number_input("MDVP:PPQ", min_value=0.0, format="%.5f")
        ddp = st.number_input("Jitter:DDP", min_value=0.0, format="%.5f")
    
    with col2:
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, format="%.5f")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, format="%.3f")
        apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, format="%.5f")
        apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, format="%.5f")
        apq = st.number_input("MDVP:APQ", min_value=0.0, format="%.5f")
        dda = st.number_input("Shimmer:DDA", min_value=0.0, format="%.5f")
        nhr = st.number_input("NHR", min_value=0.0, format="%.5f")
    
    with col3:
        hnr = st.number_input("HNR", min_value=0.0, format="%.3f")
        rpde = st.number_input("RPDE", min_value=0.0, format="%.6f")
        dfa = st.number_input("DFA", min_value=0.0, format="%.6f")
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=10.0, format="%.6f")
        spread2 = st.number_input("spread2", min_value=-10.0, max_value=10.0, format="%.6f")
        d2 = st.number_input("D2", min_value=0.0, format="%.6f")
        ppe = st.number_input("PPE", min_value=0.0, format="%.6f")

    # Prediction button
    if st.button("Predict Parkinson's Risk", key="parkinsons_predict"):
        
        # 1. Collect data (in the correct order)
        input_data = [
            fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
            shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]
        
        # 2. Convert and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # 3. Standardize
        std_data = parkinsons_scaler.transform(input_data_reshaped)
        
        # 4. Predict
        prediction = parkinsons_model.predict(std_data)
        
        # 5. Display
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.success("You are at low risk for Parkinson's Disease.")
        else:
            st.error("You are at high risk for Parkinson's Disease. Please consult a doctor.")