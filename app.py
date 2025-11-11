import streamlit as st
import pickle
import numpy as np
import time  # Added for the spinner effect

# --- PAGE CONFIG ---
# This is new! Set a page icon and title.
st.set_page_config(
    page_title="GuardianHealth: Multi-Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
def load_css():
    st.markdown("""
    <style>
        /* Main background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #212121; /* Sets all default text to dark grey */
        }

        /* --- NEW FIX HERE --- */
        /* Force all widget labels (like 'Number of Pregnancies') to be dark */
        [data-testid="stAppViewContainer"] label {
            color: #212121 !important; 
        }
        /* --- END OF NEW FIX --- */

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 2px solid #E0E0E0;
            color: #333333; /* FIX for sidebar text */
        }
        
        /* Ensure sidebar radio buttons are visible */
        [data-testid="stSidebar"] .stRadio > label {
            color: #111111;
        }

        /* Prediction Buttons */
        .stButton > button {
            width: 100%;
            border: 2px solid #4CAF50;
            border-radius: 25px;
            color: #FFFFFF;
            background-color: #4CAF50;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #FFFFFF;
            color: #4CAF50;
            border: 2px solid #4CAF50;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* Result & Alert Boxes - with explicit text colors */
        [data-testid="stSuccess"] {
            background-color: #E8F5E9;
            border-left: 10px solid #4CAF50;
            border-radius: 8px;
            color: #1B5E20; /* Dark green text */
        }
        [data-testid="stError"] {
            background-color: #FFEBEE;
            border-left: 10px solid #F44336;
            border-radius: 8px;
            color: #B71C1C; /* Dark red text */
        }
        [data-testid="stWarning"] {
            background-color: #FFF3E0; /* Light orange */
            border-left: 10px solid #FF9800; /* Orange bar */
            color: #E65100; /* Dark orange text */
            border-radius: 8px;
        }
        [data-testid="stInfo"] {
            background-color: #E3F2FD; /* Light blue */
            border-left: 10px solid #2196F3; /* Blue bar */
            color: #0D47A1; /* Dark blue text */
            border-radius: 8px;
        }
        
        /* Titles */
        h1, h2, h3 {
            color: #212121; /* Dark grey for titles */
        }

        /* Expander styling */
        .st-expander-header {
            font-size: 16px;
            font-weight: 500;
            color: #212121;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()


# --- LOAD MODELS AND SCALERS ---
# (Using your new file structure with models in the root directory)
try:
    with open('diabetes_model.pkl', 'rb') as f:
        diabetes_model = pickle.load(f)
    with open('diabetes_scaler.pkl', 'rb') as f:
        diabetes_scaler = pickle.load(f)
    with open('heart_model.pkl', 'rb') as f:
        heart_model = pickle.load(f)
    with open('heart_scaler.pkl', 'rb') as f:
        heart_scaler = pickle.load(f)
    with open('parkinsons_model.pkl', 'rb') as f:
        parkinsons_model = pickle.load(f)
    with open('parkinsons_scaler.pkl', 'rb') as f:
        parkinsons_scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure all .pkl files are in the root directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --- SIDEBAR NAVIGATION ---
# New: Added a title with an icon
st.sidebar.title("🩺 GuardianHealth Navigator")
st.sidebar.write("---")

selection = st.sidebar.radio(
    "Select a Prediction Model",
    [
        "Home",
        "Diabetes Prediction",
        "Heart Disease Prediction",
        "Parkinson's Prediction"
    ]
)
st.sidebar.write("---")
st.sidebar.info(
    "**Disclaimer:** This tool is for educational purposes only. "
    "Consult a medical professional for any health concerns."
)


# --- HOME PAGE ---
# New: Redesigned the home page to be more welcoming.
if selection == "Home":
    st.title("Welcome to the GuardianHealth Prediction System")
    st.write("""
    This application uses advanced machine learning models to help you get an *early indication*
    of your risk for several common diseases.
    
    Please select a disease from the sidebar to begin.
    """)
    
    st.image("https://media.istockphoto.com/vectors/medical-healthcare-service-vector-id1212850904?k=20&m=1212850904&s=612x612&w=0&h=cCrVqgC29-I3S-0K8vB-lq0n3Z-NlKk-Nq0u2T-c8Y8=", 
             caption="Empowering you with data-driven health insights.", use_column_width=True)

    st.subheader("How It Works")
    st.write("""
    1.  **Select a Disease:** Choose a model from the sidebar.
    2.  **Enter Your Data:** Fill in the required medical parameters. If you're unsure, **we've provided standard values and 'How to' guides** for you.
    3.  **Get Your Result:** Our model will analyze the data and provide a risk prediction.
    """)
    
    st.warning("**Important:** These predictions are not a medical diagnosis. They are based on statistical models and should be treated as informational only.")


# --- DIABETES PREDICTION PAGE ---
elif selection == "Diabetes Prediction":
    st.title("Type 2 Diabetes Prediction")
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=100)
    st.write("Fill in the details below. We've set standard values for you.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        # New: Added `value=` for standard/default values
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, value=1)
        # New: Added `st.expander` for help
        with st.expander("ℹ️ Help: Pregnancies"):
            st.write("Enter the total number of times you have been pregnant.")
            
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=99.0, step=0.1, value=20.0)
        with st.expander("ℹ️ Help: Skin Thickness"):
            st.write("This is the 'Triceps skin fold thickness'. A medical professional measures this. A typical value is ~20-30mm.")

    with col2:
        Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=200.0, step=0.1, value=100.0)
        with st.expander("ℹ️ Help: Glucose"):
            st.write("This is your 'Fasting Blood Sugar' level. A normal fasting glucose is typically **70-100 mg/dL**. This requires a blood test.")
            
        Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=850.0, step=0.1, value=79.0)
        with st.expander("ℹ️ Help: Insulin"):
            st.write("This is the '2-Hour serum insulin' level. A normal range is broad. **If you don't know this, leave the default.** This requires a blood test.")

    with col3:
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=122.0, step=0.1, value=69.0)
        with st.expander("ℹ️ Help: Blood Pressure"):
            st.write("This is the 'Diastolic' (lower number) blood pressure. A normal value is **below 80 mm Hg**.")
            
        Age = st.number_input("Age (years)", min_value=1, max_value=120, step=1, value=30)

    st.write("---")
    
    # New: BMI Calculator built-in
    st.subheader("BMI Calculator")
    st.write("BMI is a key factor. You can calculate it here:")
    
    bmi_col1, bmi_col2 = st.columns(2)
    with bmi_col1:
        weight = st.number_input("Your Weight (in kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
    with bmi_col2:
        height = st.number_input("Your Height (in meters)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
    
    try:
        calculated_bmi = weight / (height ** 2)
        st.info(f"Your Calculated BMI is: **{calculated_bmi:.2f}**")
    except ZeroDivisionError:
        calculated_bmi = 0.0

    st.write("---")

    col_final1, col_final2 = st.columns(2)
    with col_final1:
        # New: The BMI input now *suggests* the calculated value.
        BMI = st.number_input("Enter Your BMI (or use calculator above)", min_value=0.0, max_value=67.0, step=0.1, value=round(calculated_bmi, 1) or 25.0)

    with col_final2:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.001, format="%.3f", value=0.471)
        with st.expander("ℹ️ Help: Pedigree Function"):
            st.write("This complex value scores your family history of diabetes. A higher number means a stronger genetic link. The average is ~0.47.")

    st.write("") # Add some space

    # New: Added spinner and balloons
    if st.button("Predict Diabetes Risk", key="diabetes_predict"):
        with st.spinner("Analyzing your data..."):
            time.sleep(1) # Simulate processing
            
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            std_data = diabetes_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = diabetes_model.predict(std_data)
        
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.success("You are at low risk for Diabetes.")
            st.balloons() # New: Celebration!
        else:
            st.error("You are at high risk for Diabetes. Please consult a doctor.")


# --- HEART DISEASE PREDICTION PAGE ---
elif selection == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.write("Fill in the details below. We've set standard values for you.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=50)
        sex = st.selectbox("Sex", ("Male", "Female"), index=0)
        cp = st.selectbox("Chest Pain Type (cp)", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"), index=3)
        with st.expander("ℹ️ Help: Chest Pain Type"):
            st.write("- **Typical Angina:** Chest pain related to exertion. \n- **Asymptomatic:** No chest pain.")

    with col2:
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=200, step=1, value=120)
        with st.expander("ℹ️ Help: Resting BP"):
            st.write("This is the 'Systolic' (upper number) blood pressure. A normal value is **below 120 mm Hg**.")
        
        chol = st.number_input("Serum Cholestoral (chol)", min_value=0, max_value=600, step=1, value=200)
        with st.expander("ℹ️ Help: Cholesterol"):
            st.write("A desirable total cholesterol level is **below 200 mg/dL**. This requires a blood test.")
        
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ("True", "False"), index=1)

    with col3:
        restecg = st.selectbox("Resting ECG Results (restecg)", ("Normal", "ST-T wave abnormality", "Probable/definite left ventricular hypertrophy"), index=0)
        thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=0, max_value=220, step=1, value=150)
        with st.expander("ℹ️ Help: Max Heart Rate"):
            st.write("This is typically measured during a stress test. A rough estimate is **220 - Your Age**.")
        
        exang = st.selectbox("Exercise Induced Angina (exang)", ("Yes", "No"), index=1)

    st.write("---")
    st.subheader("Advanced ECG & Test Details")
    col_adv1, col_adv2, col_adv3 = st.columns(3)

    with col_adv1:
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    with col_adv2:
        slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", ("Upsloping", "Flat", "Downsloping"), index=1)
    with col_adv3:
        ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, step=1, value=0)
        thal = st.selectbox("Thalassemia (thal)", ("Normal", "Fixed defect", "Reversable defect"), index=1)
        with st.expander("ℹ️ Help: Thalassemia"):
            st.write("This is a blood disorder found via a specific test. 'Normal' is a common default.")

    st.write("")
    
    if st.button("Predict Heart Disease Risk", key="heart_predict"):
        with st.spinner("Analyzing your data..."):
            time.sleep(1)
            
            sex_num = 1 if sex == "Male" else 0
            fbs_num = 1 if fbs == "True" else 0
            exang_num = 1 if exang == "Yes" else 0
            
            cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp_num = cp_mapping[cp]
            
            restecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Probable or definite left ventricular hypertrophy": 2}
            restecg_num = restecg_mapping[restecg]
            
            slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_num = slope_mapping[slope]
            
            thal_mapping = {"Normal": 1, "Fixed defect": 2, "Reversable defect": 3}
            thal_num = thal_mapping[thal]
            
            input_data = [age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalach, exang_num, oldpeak, slope_num, ca, thal_num]
            std_data = heart_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = heart_model.predict(std_data)
        
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.success("You are at low risk for Heart Disease.")
            st.balloons()
        else:
            st.error("You are at high risk for Heart Disease. Please consult a doctor.")


# --- PARKINSON'S PREDICTION PAGE ---
elif selection == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")
    st.image("https://img.icons8.com/color/96/000000/voice-recognition.png", width=100)
    
    # New: A very important warning for this specific model
    st.warning("""
    **Important:** This model predicts Parkinson's based on **advanced voice metrics**. 
    These values (like 'MDVP:Fo(Hz)') are **not** possible to measure at home. 
    They must be captured by a specialist (e.g., a speech-language pathologist) using professional equipment.
    
    We have provided standard average values, but **this tool is intended for use with data from a medical report.**
    """)

    col1, col2, col3 = st.columns(3)

    # Adding default values (`value=...`) to all inputs
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", value=154.2, format="%.3f")
        fhi = st.number_input("MDVP:Fhi(Hz)", value=197.1, format="%.3f")
        flo = st.number_input("MDVP:Flo(Hz)", value=116.3, format="%.3f")
        jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.006, format="%.5f")
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00004, format="%.7f")
        rap = st.number_input("MDVP:RAP", value=0.003, format="%.5f")
        ppq = st.number_input("MDVP:PPQ", value=0.003, format="%.5f")
        ddp = st.number_input("Jitter:DDP", value=0.009, format="%.5f")
    
    with col2:
        shimmer = st.number_input("MDVP:Shimmer", value=0.029, format="%.5f")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.269, format="%.3f")
        apq3 = st.number_input("Shimmer:APQ3", value=0.015, format="%.5f")
        apq5 = st.number_input("Shimmer:APQ5", value=0.016, format="%.5f")
        apq = st.number_input("MDVP:APQ", value=0.024, format="%.5f")
        dda = st.number_input("Shimmer:DDA", value=0.046, format="%.5f")
        nhr = st.number_input("NHR", value=0.024, format="%.5f")
    
    with col3:
        hnr = st.number_input("HNR", value=21.8, format="%.3f")
        rpde = st.number_input("RPDE", value=0.498, format="%.6f")
        dfa = st.number_input("DFA", value=0.718, format="%.6f")
        spread1 = st.number_input("spread1", value=-5.684, format="%.6f")
        spread2 = st.number_input("spread2", value=0.226, format="%.6f")
        d2 = st.number_input("D2", value=2.381, format="%.6f")
        ppe = st.number_input("PPE", value=0.206, format="%.6f")

    st.write("")

    if st.button("Predict Parkinson's Risk", key="parkinsons_predict"):
        with st.spinner("Analyzing vocal metrics..."):
            time.sleep(1)
            
            input_data = [
                fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]
            
            std_data = parkinsons_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = parkinsons_model.predict(std_data)
        
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.success("The voice metrics suggest a low risk of Parkinson's.")
            st.balloons()
        else:
            st.error("The voice metrics suggest a high risk of Parkinson's. Please consult a specialist.")