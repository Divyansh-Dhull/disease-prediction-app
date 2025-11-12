import streamlit as st
import pickle
import numpy as np
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GuardianHealth: Multi-Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CUSTOM CSS ---
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global font fix - Apply Inter font everywhere */
        html, body, [class*="css"], [data-testid="stAppViewContainer"], 
        .stApp, p, div, span, label, input, select, textarea, button, h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        /* Main background with gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #1a1a1a;
        }
        
        /* Content wrapper for better readability */
        .main .block-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 1rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        }

        /* Fix ALL labels to be dark and bold */
        label, [data-testid="stWidgetLabel"], .stSelectbox label, 
        .stNumberInput label, .stRadio label, [data-testid="stAppViewContainer"] label,
        div[data-baseweb="select"] label, .stTextInput label {
            color: #2c3e50 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* Fix input text color */
        input, select, textarea {
            color: #2c3e50 !important;
            font-weight: 500 !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
            border-right: none;
        }
        
        [data-testid="stSidebar"] * {
            color: #ecf0f1 !important;
        }
        
        [data-testid="stSidebar"] .stRadio > label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label {
            color: #ecf0f1 !important;
            padding: 0.75rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        /* Titles with gradient effect */
        h1 {
            color: #2c3e50 !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 1.5rem !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        h2, h3 {
            color: #34495e !important;
            font-weight: 600 !important;
        }

        /* Enhanced prediction buttons */
        .stButton > button {
            width: 100%;
            border: none;
            border-radius: 12px;
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        /* Enhanced result boxes */
        [data-testid="stSuccess"] {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 6px solid #28a745;
            border-radius: 12px;
            padding: 1.5rem;
            color: #155724 !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
        }
        
        [data-testid="stError"] {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 6px solid #dc3545;
            border-radius: 12px;
            padding: 1.5rem;
            color: #721c24 !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
        }
        
        [data-testid="stWarning"] {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border-left: 6px solid #ffc107;
            color: #856404 !important;
            border-radius: 12px;
            padding: 1.5rem;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);
        }
        
        [data-testid="stInfo"] {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border-left: 6px solid #17a2b8;
            color: #0c5460 !important;
            border-radius: 12px;
            padding: 1.5rem;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(23, 162, 184, 0.2);
        }

        /* Info box text color fix */
        [data-testid="stSuccess"] *, [data-testid="stError"] *, 
        [data-testid="stWarning"] *, [data-testid="stInfo"] * {
            color: inherit !important;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            font-weight: 600 !important;
            color: #2c3e50 !important;
        }

        /* Card-like sections */
        .disease-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border-left: 4px solid #667eea;
        }

        /* Divider styling */
        hr {
            margin: 2rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
        }

        /* Number input styling */
        [data-testid="stNumberInput"] input {
            border-radius: 8px;
            border: 2px solid #e9ecef;
            padding: 0.5rem;
            transition: all 0.3s ease;
        }

        [data-testid="stNumberInput"] input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        /* Selectbox styling */
        [data-baseweb="select"] {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()


# --- LOAD MODELS AND SCALERS ---
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
st.sidebar.title("🩺 GuardianHealth Navigator")
st.sidebar.write("---")

selection = st.sidebar.radio(
    "Select a Prediction Model",
    [
        "🏠 Home",
        "🍬 Diabetes Prediction",
        "❤️ Heart Disease Prediction",
        "🧠 Parkinson's Prediction"
    ]
)
st.sidebar.write("---")
st.sidebar.info(
    "**Disclaimer:** This tool is for educational purposes only. "
    "Consult a medical professional for any health concerns."
)


# --- HOME PAGE ---
if selection == "🏠 Home":
    st.title("🩺 Welcome to GuardianHealth")
    st.write("""
    ### Your Personal Health Risk Assessment Tool
    
    This application uses advanced machine learning models to provide early indications
    of your risk for common diseases. Our predictions are based on validated medical parameters
    and statistical models.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="disease-card">
        <h3>🍬 Diabetes</h3>
        <p>Assess your risk for Type 2 Diabetes based on glucose levels, BMI, and family history.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="disease-card">
        <h3>❤️ Heart Disease</h3>
        <p>Evaluate cardiovascular risk using blood pressure, cholesterol, and ECG data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="disease-card">
        <h3>🧠 Parkinson's</h3>
        <p>Analyze voice metrics for early detection of Parkinson's disease indicators.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    st.subheader("📋 How It Works")
    st.write("""
    1. **Select a Disease Model** from the sidebar navigation
    2. **Review the Disease Information** including symptoms and risk factors
    3. **Enter Your Medical Parameters** with help guides provided for each field
    4. **Receive Your Risk Assessment** with personalized recommendations
    5. **Consult a Healthcare Professional** if indicated
    """)
    
    st.warning("⚠️ **Important:** These predictions are not medical diagnoses. They are statistical assessments based on machine learning models and should be used for informational purposes only.")


# --- DIABETES PREDICTION PAGE ---
elif selection == "🍬 Diabetes Prediction":
    st.title("🍬 Type 2 Diabetes Risk Assessment")
    
    # Disease Information Section
    with st.expander("📚 About Type 2 Diabetes", expanded=True):
        st.markdown("""
        **Type 2 Diabetes** is a chronic condition that affects how your body processes blood sugar (glucose).
        
        **Common Symptoms You Can Identify:**
        - ✓ Increased thirst and frequent urination
        - ✓ Increased hunger, especially after eating
        - ✓ Unexplained weight loss or gain
        - ✓ Fatigue and weakness
        - ✓ Blurred vision
        - ✓ Slow-healing sores or frequent infections
        - ✓ Tingling or numbness in hands or feet
        - ✓ Darkened skin areas (often in armpits or neck)
        
        **Risk Factors:** Obesity, family history, age over 45, physical inactivity, high blood pressure
        """)

    st.write("---")
    st.subheader("Enter Your Medical Parameters")
    st.write("We've provided standard values as defaults. Adjust them based on your recent medical tests.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, value=1)
        with st.expander("ℹ️ Help: Pregnancies"):
            st.write("Enter the total number of times you have been pregnant. Enter 0 if not applicable.")
            
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=99.0, step=0.1, value=20.0)
        with st.expander("ℹ️ Help: Skin Thickness"):
            st.write("Triceps skin fold thickness measured by a healthcare professional. Typical range: 20-30mm.")

    with col2:
        Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=200.0, step=0.1, value=100.0)
        with st.expander("ℹ️ Help: Glucose"):
            st.write("Fasting blood sugar level. Normal: 70-100 mg/dL. Pre-diabetes: 100-125 mg/dL. Diabetes: ≥126 mg/dL.")
            
        Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=850.0, step=0.1, value=79.0)
        with st.expander("ℹ️ Help: Insulin"):
            st.write("2-Hour serum insulin level. If unknown, use the default value. Requires blood test.")

    with col3:
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=122.0, step=0.1, value=69.0)
        with st.expander("ℹ️ Help: Blood Pressure"):
            st.write("Diastolic blood pressure (lower number). Normal: <80 mm Hg. Elevated: 80-89 mm Hg.")
            
        Age = st.number_input("Age (years)", min_value=1, max_value=120, step=1, value=30)

    st.write("---")
    
    st.subheader("📊 BMI Calculator")
    
    bmi_col1, bmi_col2 = st.columns(2)
    with bmi_col1:
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
    with bmi_col2:
        height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
    
    try:
        calculated_bmi = weight / (height ** 2)
        st.info(f"**Calculated BMI: {calculated_bmi:.2f}** | " + 
                (f"Underweight" if calculated_bmi < 18.5 else
                 f"Normal" if calculated_bmi < 25 else
                 f"Overweight" if calculated_bmi < 30 else
                 f"Obese"))
    except ZeroDivisionError:
        calculated_bmi = 0.0

    st.write("---")

    col_final1, col_final2 = st.columns(2)
    with col_final1:
        BMI = st.number_input("BMI", min_value=0.0, max_value=67.0, step=0.1, value=round(calculated_bmi, 1) if calculated_bmi else 25.0)

    with col_final2:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.001, format="%.3f", value=0.471)
        with st.expander("ℹ️ Help: Pedigree Function"):
            st.write("Family history score for diabetes. Higher = stronger genetic link. Average: ~0.47")

    st.write("")

    if st.button("🔍 Predict Diabetes Risk", key="diabetes_predict"):
        with st.spinner("🔄 Analyzing your data..."):
            time.sleep(1.5)
            
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            std_data = diabetes_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = diabetes_model.predict(std_data)
        
        st.write("---")
        st.subheader("📊 Assessment Result")
        
        if prediction[0] == 0:
            st.success("✅ **Low Risk for Type 2 Diabetes**")
            st.balloons()
            st.write("""
            **Good News!** Your parameters suggest a low risk for diabetes. However, continue maintaining a healthy lifestyle.
            
            **Recommended Actions:**
            - ✓ Maintain a balanced diet with limited refined sugars
            - ✓ Engage in regular physical activity (150 minutes/week)
            - ✓ Maintain healthy body weight (BMI 18.5-24.9)
            - ✓ Get annual health check-ups
            - ✓ Monitor blood sugar if you have family history
            - ✓ Stay hydrated and get adequate sleep
            """)
        else:
            st.error("⚠️ **High Risk for Type 2 Diabetes Detected**")
            st.write("""
            **Important:** Your parameters indicate an elevated risk for Type 2 Diabetes.
            
            **Immediate Actions Required:**
            
            🏥 **Medical Consultation (Priority)**
            - Schedule an appointment with your doctor or endocrinologist immediately
            - Request comprehensive diabetes screening tests (HbA1c, fasting glucose)
            - Discuss family history and personal risk factors
            
            🍎 **Dietary Modifications**
            - AVOID: Refined sugars, white bread, sugary beverages, processed foods
            - REDUCE: Carbohydrate intake, especially simple carbs
            - INCREASE: Fiber-rich foods, vegetables, lean proteins, whole grains
            - Practice portion control and eat smaller, frequent meals
            
            🏃 **Lifestyle Changes**
            - Start with 30 minutes of moderate exercise daily (walking, swimming)
            - Aim for 150+ minutes of physical activity per week
            - Work towards achieving healthy BMI (18.5-24.9)
            - Reduce sedentary time and increase daily movement
            
            💊 **Medical Management**
            - Your doctor may prescribe Metformin or other medications
            - Regular blood glucose monitoring may be recommended
            - Consider consulting a nutritionist for meal planning
            
            ⚠️ **Warning Signs to Watch:**
            - Extreme thirst or hunger
            - Frequent urination, especially at night
            - Unexplained weight changes
            - Persistent fatigue or blurred vision
            
            **Seek emergency care if you experience:** Confusion, extreme weakness, rapid breathing, or fruity-smelling breath.
            """)


# --- HEART DISEASE PREDICTION PAGE ---
elif selection == "❤️ Heart Disease Prediction":
    st.title("❤️ Heart Disease Risk Assessment")
    
    with st.expander("📚 About Heart Disease", expanded=True):
        st.markdown("""
        **Cardiovascular Disease** includes conditions affecting the heart and blood vessels.
        
        **Common Symptoms You Can Identify:**
        - ✓ Chest pain or discomfort (angina)
        - ✓ Shortness of breath during activity or rest
        - ✓ Irregular heartbeat or palpitations
        - ✓ Dizziness or lightheadedness
        - ✓ Fatigue and weakness
        - ✓ Swelling in legs, ankles, or feet
        - ✓ Pain in neck, jaw, throat, or upper abdomen
        - ✓ Rapid or irregular pulse
        
        **Risk Factors:** High blood pressure, high cholesterol, smoking, diabetes, obesity, family history, age
        """)

    st.write("---")
    st.subheader("Enter Your Medical Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=50)
        sex = st.selectbox("Sex", ("Male", "Female"), index=0)
        cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"), index=3)
        with st.expander("ℹ️ Help: Chest Pain Type"):
            st.write("""
            - **Typical Angina:** Exercise-related chest pain
            - **Atypical Angina:** Chest pain not related to exertion
            - **Non-anginal:** Chest discomfort not heart-related
            - **Asymptomatic:** No chest pain
            """)

    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1, value=120)
        with st.expander("ℹ️ Help: Resting BP"):
            st.write("Systolic BP (upper number). Normal: <120. Elevated: 120-129. High: ≥130")
        
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, step=1, value=200)
        with st.expander("ℹ️ Help: Cholesterol"):
            st.write("Total cholesterol. Desirable: <200 mg/dL. Borderline high: 200-239. High: ≥240")
        
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("True", "False"), index=1)

    with col3:
        restecg = st.selectbox("Resting ECG Results", ("Normal", "ST-T wave abnormality", "Probable/definite left ventricular hypertrophy"), index=0)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=220, step=1, value=150)
        with st.expander("ℹ️ Help: Max Heart Rate"):
            st.write("Typically measured during stress test. Estimate: 220 - Your Age")
        
        exang = st.selectbox("Exercise Induced Angina", ("Yes", "No"), index=1)

    st.write("---")
    st.subheader("Advanced ECG & Test Details")
    col_adv1, col_adv2, col_adv3 = st.columns(3)

    with col_adv1:
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    with col_adv2:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ("Upsloping", "Flat", "Downsloping"), index=1)
    with col_adv3:
        ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1, value=0)
        thal = st.selectbox("Thalassemia", ("Normal", "Fixed defect", "Reversable defect"), index=1)

    st.write("")
    
    if st.button("🔍 Predict Heart Disease Risk", key="heart_predict"):
        with st.spinner("🔄 Analyzing your cardiovascular data..."):
            time.sleep(1.5)
            
            sex_num = 1 if sex == "Male" else 0
            fbs_num = 1 if fbs == "True" else 0
            exang_num = 1 if exang == "Yes" else 0
            
            cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp_num = cp_mapping[cp]
            
            restecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Probable/definite left ventricular hypertrophy": 2}
            restecg_num = restecg_mapping.get(restecg, 0)
            
            slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_num = slope_mapping[slope]
            
            thal_mapping = {"Normal": 1, "Fixed defect": 2, "Reversable defect": 3}
            thal_num = thal_mapping[thal]
            
            input_data = [age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalach, exang_num, oldpeak, slope_num, ca, thal_num]
            std_data = heart_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = heart_model.predict(std_data)
        
        st.write("---")
        st.subheader("📊 Assessment Result")
        
        if prediction[0] == 0:
            st.success("✅ **Low Risk for Heart Disease**")
            st.balloons()
            st.write("""
            **Excellent!** Your cardiovascular parameters indicate low risk.
            
            **Recommended Actions:**
            - ✓ Maintain heart-healthy diet (Mediterranean or DASH diet)
            - ✓ Regular aerobic exercise (150 minutes/week)
            - ✓ Keep blood pressure below 120/80 mm Hg
            - ✓ Maintain healthy cholesterol levels
            - ✓ Don't smoke and limit alcohol consumption
            - ✓ Manage stress through relaxation techniques
            - ✓ Annual cardiovascular check-ups after age 40
            """)
        else:
            st.error("⚠️ **High Risk for Heart Disease Detected**")
            st.write("""
            **Critical:** Your parameters suggest elevated cardiovascular risk.
            
            **Immediate Actions Required:**
            
            🏥 **Urgent Medical Consultation**
            - **See a cardiologist within 1-2 weeks**
            - Request comprehensive cardiac workup (ECG, echocardiogram, stress test)
            - Discuss your results and medical history in detail
            - **If experiencing chest pain, shortness of breath, or severe symptoms: CALL EMERGENCY SERVICES (911)**
            
            💊 **Medical Management**
            - Your doctor may prescribe:
              - **Statins** (Atorvastatin, Rosuvastatin) for cholesterol
              - **ACE inhibitors** or **Beta-blockers** for blood pressure
              - **Aspirin** (75-100mg daily) for blood thinning
              - **Nitroglycerin** for angina if needed
            - Take medications exactly as prescribed
            - Regular monitoring of BP and cholesterol required
            
            🍎 **Critical Dietary Changes**
            - AVOID: Trans fats, saturated fats, excess salt (>2300mg/day), processed meats
            - REDUCE: Red meat, fried foods, sugary items, refined carbs
            - INCREASE: Omega-3 rich fish, nuts, olive oil, fruits, vegetables, whole grains
            - Follow DASH or Mediterranean diet pattern
            - Limit sodium to 1500-2000mg per day
            
            🏃 **Lifestyle Modifications**
            - **STOP smoking immediately** (seek cessation program if needed)
            - Start gentle exercise (walking) - consult doctor before vigorous activity
            - Aim for 30 minutes of moderate activity most days
            - Maintain healthy weight (BMI <25)
            - Limit alcohol to 1 drink/day (women) or 2/day (men)
            - Get 7-9 hours quality sleep nightly
            - Practice stress management (meditation, yoga)
            
            📊 **Regular Monitoring**
            - Check blood pressure daily (keep a log)
            - Monitor for new or worsening symptoms
            - Follow-up appointments every 3-6 months
            - Annual or bi-annual cardiac stress tests
            
            🚨 **Emergency Warning Signs - Call 911 if you experience:**
            - Severe chest pain or pressure lasting >5 minutes
            - Pain radiating to arm, jaw, or back
            - Severe shortness of breath
            - Cold sweats, nausea with chest discomfort
            - Sudden weakness or loss of consciousness
            - Irregular or very rapid heartbeat
            
            **Important:** Heart disease is manageable with proper treatment. Follow your doctor's advice strictly.
            """)


# --- PARKINSON'S PREDICTION PAGE ---
elif selection == "🧠 Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Risk Assessment")
    
    with st.expander("📚 About Parkinson's Disease", expanded=True):
        st.markdown("""
        **Parkinson's Disease** is a progressive neurological disorder affecting movement control.
        
        **Common Symptoms You Can Identify:**
        - ✓ Tremor (shaking), usually starting in hands or fingers
        - ✓ Slowed movement (bradykinesia)
        - ✓ Muscle stiffness and rigidity
        - ✓ Impaired posture and balance
        - ✓ Loss of automatic movements (blinking, swinging arms when walking)
        - ✓ Speech changes (softer, slurred, monotone)
        - ✓ Writing changes (smaller handwriting)
        - ✓ Difficulty with fine motor skills
        
        **Risk Factors:** Age (60+), heredity, gender (men more likely), exposure to toxins
        """)
    
    st.warning("""
    ⚠️ **Important Notice:** This model analyzes **advanced voice metrics** that require professional equipment.
    
    These parameters (MDVP, Jitter, Shimmer, etc.) **cannot be measured at home**. They must be captured by:
    - Speech-language pathologist
    - Neurologist with specialized equipment
    - Voice analysis laboratory
    
    Standard values are provided for demonstration. **Use actual medical report data for accurate predictions.**
    """)

    st.write("---")
    st.subheader("Enter Voice Analysis Parameters")

    col1, col2, col3 = st.columns(3)

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

    if st.button("🔍 Predict Parkinson's Risk", key="parkinsons_predict"):
        with st.spinner("🔄 Analyzing vocal metrics..."):
            time.sleep(1.5)
            
            input_data = [
                fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]
            
            std_data = parkinsons_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = parkinsons_model.predict(std_data)
        
        st.write("---")
        st.subheader("📊 Assessment Result")
        
        if prediction[0] == 0:
            st.success("✅ **Low Risk for Parkinson's Disease**")
            st.balloons()
            st.write("""
            **Good News!** Voice metrics suggest low risk for Parkinson's disease.
            
            **Recommended Actions:**
            - ✓ Continue regular health check-ups
            - ✓ Stay physically active with regular exercise
            - ✓ Maintain brain health (mental exercises, social engagement)
            - ✓ Follow balanced diet rich in antioxidants
            - ✓ Monitor for any new symptoms as you age
            - ✓ Consider neurological evaluation if family history exists
            """)
        else:
            st.error("⚠️ **High Risk for Parkinson's Disease Detected**")
            st.write("""
            **Important:** Voice analysis indicates elevated risk for Parkinson's disease.
            
            **Immediate Actions Required:**
            
            🏥 **Specialist Consultation (Urgent)**
            - **Schedule appointment with neurologist immediately**
            - Request comprehensive neurological examination
            - Discuss voice analysis results and any symptoms
            - May need: MRI/CT scan, DaTscan, or other specialized tests
            - Consider movement disorder specialist evaluation
            
            💊 **Medical Management**
            - Early diagnosis allows better treatment outcomes
            - Possible medications if diagnosed:
              - **Levodopa/Carbidopa (Sinemet)** - Gold standard treatment
              - **Dopamine agonists** (Pramipexole, Ropinirole)
              - **MAO-B inhibitors** (Selegiline, Rasagiline)
              - **Anticholinergics** for tremor control
            - Medication timing is critical - follow prescriptions exactly
            - Regular follow-ups to adjust dosages
            
            🏃 **Physical Therapy & Exercise**
            - **CRITICAL:** Regular exercise slows progression
            - Recommended activities:
              - Walking, swimming, cycling (30 min daily)
              - Tai Chi or yoga for balance
              - Strength training 2-3x per week
              - Dance or rhythmic movement therapy
            - Physical therapy for:
              - Gait and balance training
              - Flexibility and range of motion
              - Posture correction
            
            🗣️ **Speech & Occupational Therapy**
            - **Speech therapy (LSVT LOUD program)** to improve voice volume
            - Swallowing evaluation if needed
            - Occupational therapy for:
              - Fine motor skills maintenance
              - Adaptive strategies for daily tasks
              - Home safety modifications
            
            🍎 **Dietary Recommendations**
            - INCREASE: Antioxidant-rich foods (berries, leafy greens)
            - Mediterranean diet pattern beneficial
            - Adequate fiber to prevent constipation
            - Stay well-hydrated (8+ glasses water/day)
            - Consider: Omega-3 supplements, Vitamin D, Coenzyme Q10 (consult doctor)
            - Take Levodopa 30-60 minutes before meals for best absorption
            - Limit protein intake near medication times if on Levodopa
            
            🧠 **Cognitive & Mental Health**
            - Monitor for depression/anxiety (common in Parkinson's)
            - Engage in mentally stimulating activities
            - Maintain social connections
            - Consider support groups (local or online)
            - Cognitive behavioral therapy if needed
            
            🏠 **Home Safety & Lifestyle**
            - Remove fall hazards (rugs, clutter)
            - Install grab bars in bathroom
            - Improve lighting throughout home
            - Use assistive devices as needed (cane, walker)
            - Get adequate sleep (7-9 hours)
            - Manage stress through relaxation techniques
            
            📊 **Monitoring & Follow-up**
            - Track symptoms in a diary (tremor, stiffness, mobility)
            - Regular neurological evaluations (every 3-6 months)
            - Monitor medication effectiveness and side effects
            - Stay informed about new treatments and clinical trials
            
            🚨 **Warning Signs Requiring Immediate Attention:**
            - Sudden worsening of symptoms
            - Difficulty swallowing or breathing
            - Severe confusion or hallucinations
            - Falls or severe balance problems
            - Inability to move or severe rigidity
            
            **Important Resources:**
            - Parkinson's Foundation: www.parkinson.org
            - Michael J. Fox Foundation: www.michaeljfox.org
            - Local support groups and resources
            
            **Remember:** Parkinson's progression varies greatly. Early intervention and consistent management significantly improve quality of life.
            """)

# --- FOOTER ---
st.write("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p><strong>GuardianHealth</strong> - Empowering Health Through Data</p>
    <p style='font-size: 0.9rem;'>⚕️ Always consult qualified healthcare professionals for medical decisions</p>
    <p style='font-size: 0.8rem;'>© 2024 GuardianHealth Prediction System | For Educational Use Only</p>
</div>
""", unsafe_allow_html=True)