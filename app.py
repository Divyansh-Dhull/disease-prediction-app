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

# --- ENHANCED CUSTOM CSS WITH FIXED VISIBILITY ---
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* =========================
           GLOBAL FONT & TEXT FIXES
           ========================= */
        html, body, [class*="css"], [data-testid="stAppViewContainer"], 
        .stApp, p, div, span, label, input, select, textarea, button, h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stText {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            letter-spacing: -0.01em;
        }

        /* =========================
           BACKGROUND & CONTAINER
           ========================= */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e57c2 100%);
        }
        
        /* White content wrapper with proper spacing */
        .main .block-container {
            background: #ffffff;
            border-radius: 20px;
            padding: 3rem 2.5rem;
            margin: 1.5rem auto;
            max-width: 1400px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        /* =========================
           TITLE & HEADING FIXES
           ========================= */
        /* Main title - Dark and Bold */
        h1 {
            color: #1a1a2e !important;
            font-weight: 800 !important;
            font-size: 2.8rem !important;
            margin-bottom: 1.5rem !important;
            text-align: left !important;
            line-height: 1.2 !important;
            text-shadow: none !important;
        }

        /* Subtitle */
        h2 {
            color: #2d3748 !important;
            font-weight: 700 !important;
            font-size: 1.8rem !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }

        /* Section headers */
        h3 {
            color: #4a5568 !important;
            font-weight: 600 !important;
            font-size: 1.3rem !important;
            margin-top: 1.5rem !important;
        }

        /* =========================
           LABEL & INPUT FIXES
           ========================= */
        /* ALL labels dark and readable */
        label, [data-testid="stWidgetLabel"], 
        .stSelectbox label, .stNumberInput label, 
        .stRadio label, div[data-baseweb="select"] label, 
        .stTextInput label {
            color: #1a1a2e !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.5 !important;
        }

        /* Input text color */
        input, select, textarea, [data-baseweb="input"] input {
            color: #1a1a2e !important;
            font-weight: 500 !important;
            background-color: #f7fafc !important;
        }

        /* Paragraph text */
        p, .stMarkdown p, li {
            color: #2d3748 !important;
            font-size: 1rem !important;
            line-height: 1.7 !important;
            margin-bottom: 0.8rem !important;
        }

        /* =========================
           SIDEBAR STYLING
           ========================= */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            border-right: none;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] h1 {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Radio buttons in sidebar - styled as buttons */
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 0.5rem;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
            padding: 1rem !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
            border: 2px solid transparent !important;
            cursor: pointer !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
            transform: translateX(5px);
        }
        
        /* Selected radio button */
        [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: #ffffff !important;
        }

        /* Hide default radio circles */
        [data-testid="stSidebar"] input[type="radio"] {
            display: none !important;
        }

        /* =========================
           BUTTON STYLING
           ========================= */
        .stButton > button {
            width: 100%;
            border: none;
            border-radius: 12px;
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem 2rem;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            margin-top: 1rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }

        /* =========================
           RESULT BOXES - FIXED TEXT
           ========================= */
        [data-testid="stSuccess"] {
            background: #d4edda !important;
            border-left: 6px solid #28a745 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2) !important;
        }
        
        [data-testid="stSuccess"], [data-testid="stSuccess"] * {
            color: #155724 !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stError"] {
            background: #f8d7da !important;
            border-left: 6px solid #dc3545 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2) !important;
        }
        
        [data-testid="stError"], [data-testid="stError"] * {
            color: #721c24 !important;
            font-weight: 500 !important;
        }
        
        [data-testid="stWarning"] {
            background: #fff3cd !important;
            border-left: 6px solid #ffc107 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2) !important;
        }
        
        [data-testid="stWarning"], [data-testid="stWarning"] * {
            color: #856404 !important;
            font-weight: 500 !important;
        }
        
        [data-testid="stInfo"] {
            background: #d1ecf1 !important;
            border-left: 6px solid #17a2b8 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 12px rgba(23, 162, 184, 0.2) !important;
        }
        
        [data-testid="stInfo"], [data-testid="stInfo"] * {
            color: #0c5460 !important;
            font-weight: 500 !important;
        }

        /* Fix text overlapping in result boxes */
        [data-testid="stSuccess"] p, [data-testid="stError"] p,
        [data-testid="stWarning"] p, [data-testid="stInfo"] p {
            line-height: 1.8 !important;
            margin-bottom: 1rem !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        }

        /* =========================
           EXPANDER STYLING
           ========================= */
        .streamlit-expanderHeader {
            background: #f7fafc !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            color: #1a1a2e !important;
            border: 2px solid #e2e8f0 !important;
        }

        .streamlit-expanderHeader:hover {
            background: #edf2f7 !important;
            border-color: #cbd5e0 !important;
        }

        .streamlit-expanderContent {
            background: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
            padding: 1rem !important;
        }

        /* =========================
           CARD STYLING
           ========================= */
        .disease-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
        }

        .disease-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        .disease-card h3 {
            color: #1a1a2e !important;
            margin-top: 0 !important;
        }

        .disease-card p {
            color: #4a5568 !important;
        }

        /* =========================
           DIVIDER
           ========================= */
        hr {
            margin: 2.5rem 0 !important;
            border: none !important;
            height: 2px !important;
            background: linear-gradient(90deg, transparent, #cbd5e0, transparent) !important;
        }

        /* =========================
           INPUT STYLING
           ========================= */
        [data-testid="stNumberInput"] input {
            border-radius: 8px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
            font-size: 1rem !important;
        }

        [data-testid="stNumberInput"] input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            outline: none !important;
        }

        /* Selectbox styling */
        [data-baseweb="select"] {
            border-radius: 8px !important;
        }

        [data-baseweb="select"] > div {
            border: 2px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }

        [data-baseweb="select"] > div:hover {
            border-color: #cbd5e0 !important;
        }

        /* =========================
           SPINNER
           ========================= */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }

        /* =========================
           SCROLLBAR
           ========================= */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
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
st.sidebar.title("🩺 GuardianHealth")
st.sidebar.write("---")

selection = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home",
        "🍬 Diabetes",
        "❤️ Heart Disease",
        "🧠 Parkinson's"
    ],
    label_visibility="collapsed"
)
st.sidebar.write("---")
st.sidebar.info(
    "**Disclaimer:** This tool is for educational purposes only. "
    "Always consult a medical professional for health concerns."
)


# --- HOME PAGE ---
if selection == "🏠 Home":
    st.title("GuardianHealth: AI-Powered Disease Prediction")
    st.write("""
    ### Your Personal Health Risk Assessment Platform
    
    Welcome to GuardianHealth, an advanced machine learning system designed to provide early risk assessments
    for common chronic diseases. Our models analyze medical parameters to help you understand your health better.
    """)
    
    st.write("")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="disease-card">
        <h3>🍬 Type 2 Diabetes</h3>
        <p>Evaluate your diabetes risk based on glucose levels, BMI, blood pressure, and family history indicators.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="disease-card">
        <h3>❤️ Heart Disease</h3>
        <p>Assess cardiovascular risk using blood pressure, cholesterol levels, ECG results, and exercise data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="disease-card">
        <h3>🧠 Parkinson's Disease</h3>
        <p>Analyze advanced voice metrics for early detection of Parkinson's disease indicators.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    st.subheader("How It Works")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("""
        **Step 1: Select a Disease Model**
        - Choose from the sidebar navigation
        - Review disease information and symptoms
        
        **Step 2: Enter Medical Parameters**
        - Fill in the required fields
        - Use help guides for each parameter
        - Standard values provided as defaults
        """)
    
    with col_b:
        st.write("""
        **Step 3: Get Your Assessment**
        - AI analyzes your data
        - Receive risk prediction
        - View personalized recommendations
        
        **Step 4: Take Action**
        - Follow medical advice provided
        - Consult healthcare professionals
        - Implement lifestyle changes
        """)
    
    st.write("---")
    
    st.warning("⚠️ **Medical Disclaimer:** These predictions are statistical assessments based on machine learning models. They are NOT medical diagnoses and should not replace professional medical advice, diagnosis, or treatment.")


# --- DIABETES PREDICTION PAGE ---
elif selection == "🍬 Diabetes":
    st.title("Type 2 Diabetes Risk Assessment")
    
    # Disease Information Section
    with st.expander("📚 About Type 2 Diabetes & Common Symptoms", expanded=False):
        st.write("""
        **What is Type 2 Diabetes?**
        
        Type 2 Diabetes is a chronic metabolic condition where your body either doesn't produce enough insulin 
        or doesn't use insulin effectively, leading to elevated blood sugar levels.
        
        **Common Symptoms You Can Identify:**
        
        - Increased thirst and frequent urination (especially at night)
        - Increased hunger, even after eating
        - Unexplained weight loss or gradual weight gain
        - Persistent fatigue and weakness
        - Blurred vision or visual disturbances
        - Slow-healing cuts, wounds, or frequent infections
        - Tingling, numbness, or pain in hands or feet
        - Darkened skin patches (acanthosis nigricans) in armpits or neck
        - Dry, itchy skin
        
        **Major Risk Factors:**
        
        - Obesity or being overweight (BMI > 25)
        - Family history of diabetes
        - Age over 45 years
        - Physical inactivity and sedentary lifestyle
        - High blood pressure (>130/80 mm Hg)
        - History of gestational diabetes
        - Polycystic ovary syndrome (PCOS)
        """)

    st.write("---")
    st.subheader("Enter Your Medical Parameters")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, value=1)
        with st.expander("ℹ️ Help"):
            st.write("Total number of pregnancies. Enter 0 if not applicable or if male.")
            
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=99.0, step=0.1, value=20.0)
        with st.expander("ℹ️ Help"):
            st.write("Triceps skin fold thickness. Measured by healthcare professional. Typical: 20-30mm.")

    with col2:
        Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=200.0, step=0.1, value=100.0)
        with st.expander("ℹ️ Help"):
            st.write("Fasting blood sugar. Normal: 70-100. Pre-diabetes: 100-125. Diabetes: ≥126 mg/dL.")
            
        Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=850.0, step=0.1, value=79.0)
        with st.expander("ℹ️ Help"):
            st.write("2-Hour serum insulin. Use default if unknown. Requires blood test.")

    with col3:
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=122.0, step=0.1, value=69.0)
        with st.expander("ℹ️ Help"):
            st.write("Diastolic BP (lower number). Normal: <80. Elevated: 80-89 mm Hg.")
            
        Age = st.number_input("Age (years)", min_value=1, max_value=120, step=1, value=30)

    st.write("---")
    
    st.subheader("BMI Calculator")
    
    bmi_col1, bmi_col2 = st.columns(2)
    with bmi_col1:
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
    with bmi_col2:
        height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
    
    try:
        calculated_bmi = weight / (height ** 2)
        bmi_category = (
            "Underweight" if calculated_bmi < 18.5 else
            "Normal Weight" if calculated_bmi < 25 else
            "Overweight" if calculated_bmi < 30 else
            "Obese"
        )
        st.info(f"**Calculated BMI: {calculated_bmi:.2f}** ({bmi_category})")
    except ZeroDivisionError:
        calculated_bmi = 25.0

    st.write("---")

    col_final1, col_final2 = st.columns(2)
    with col_final1:
        BMI = st.number_input("BMI Value", min_value=0.0, max_value=67.0, step=0.1, value=round(calculated_bmi, 1))

    with col_final2:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.001, format="%.3f", value=0.471)
        with st.expander("ℹ️ Help"):
            st.write("Family history score. Higher value = stronger genetic link. Average: ~0.47")

    st.write("")

    if st.button("🔍 Analyze Diabetes Risk", key="diabetes_predict"):
        with st.spinner("Analyzing your health data..."):
            time.sleep(1.5)
            
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            std_data = diabetes_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = diabetes_model.predict(std_data)
        
        st.write("---")
        st.subheader("Assessment Result")
        
        if prediction[0] == 0:
            st.success("✅ **Low Risk for Type 2 Diabetes**")
            st.balloons()
            
            st.write("**Good News!** Your parameters indicate a low risk for developing Type 2 Diabetes.")
            st.write("")
            st.write("**Recommended Preventive Actions:**")
            st.write("""
            - Maintain a balanced diet with limited refined sugars and processed foods
            - Engage in at least 150 minutes of moderate physical activity per week
            - Keep your BMI in the healthy range (18.5-24.9)
            - Schedule annual health check-ups and blood sugar monitoring
            - Stay well-hydrated (8+ glasses of water daily)
            - Get 7-9 hours of quality sleep each night
            - Manage stress through meditation, yoga, or other relaxation techniques
            - If you have a family history, consider more frequent glucose screening
            """)
        else:
            st.error("⚠️ **High Risk for Type 2 Diabetes**")
            
            st.write("**Important:** Your assessment indicates an elevated risk for Type 2 Diabetes.")
            st.write("")
            
            st.write("**🏥 IMMEDIATE MEDICAL ACTION REQUIRED:**")
            st.write("""
            - **Schedule appointment with your doctor or endocrinologist within 1-2 weeks**
            - Request comprehensive diabetes screening (HbA1c test, fasting glucose, oral glucose tolerance test)
            - Discuss your family history and all risk factors
            - Ask about pre-diabetes diagnosis and management options
            """)
            
            st.write("")
            st.write("**💊 Possible Medical Interventions:**")
            st.write("""
            Your doctor may recommend:
            - **Metformin** (500-1000mg) - First-line medication for pre-diabetes/diabetes
            - Regular blood glucose monitoring (fasting and post-meal)
            - Consultation with a certified diabetes educator
            - Referral to a registered dietitian for meal planning
            """)
            
            st.write("")
            st.write("**🍎 CRITICAL Dietary Changes:**")
            st.write("""
            **AVOID:**
            - Sugary beverages (soda, sweet tea, fruit juices)
            - White bread, white rice, refined carbohydrates
            - Processed and packaged snacks
            - Fried foods and trans fats
            - Excessive alcohol consumption
            
            **REDUCE:**
            - Total carbohydrate intake (especially simple carbs)
            - Portion sizes - use smaller plates
            - Red and processed meats
            - Salt intake (limit to 2300mg/day)
            
            **INCREASE:**
            - Non-starchy vegetables (broccoli, spinach, peppers)
            - Lean proteins (chicken, fish, legumes, tofu)
            - High-fiber foods (oats, beans, berries)
            - Healthy fats (nuts, avocado, olive oil)
            - Whole grains in moderation
            """)
            
            st.write("")
            st.write("**🏃 Essential Lifestyle Modifications:**")
            st.write("""
            - **Start exercising immediately:** Begin with 30 minutes of walking daily
            - Gradually increase to 150+ minutes of moderate activity per week
            - Add strength training exercises 2-3 times weekly
            - Reduce sedentary time - stand and move every hour
            - **Weight loss goal:** If overweight, aim to lose 5-10% of body weight
            - This alone can significantly reduce diabetes risk
            """)
            
            st.write("")
            st.write("**⚠️ Warning Signs - Seek Immediate Medical Care If You Experience:**")
            st.write("""
            - Extreme thirst or dry mouth that won't go away
            - Frequent urination (especially at night)
            - Unexplained weight loss
            - Extreme fatigue or weakness
            - Blurred vision or visual changes
            - Slow-healing wounds or frequent infections
            - Tingling or numbness in hands or feet
            """)
            
            st.write("")
            st.info("**Remember:** Type 2 Diabetes is preventable and manageable with early intervention. The steps you take today can significantly improve your health outcomes.")


# --- HEART DISEASE PREDICTION PAGE ---
elif selection == "❤️ Heart Disease":
    st.title("Heart Disease Risk Assessment")
    
    with st.expander("📚 About Heart Disease & Common Symptoms", expanded=False):
        st.write("""
        **What is Cardiovascular Disease?**
        
        Heart disease encompasses various conditions affecting the heart and blood vessels, including 
        coronary artery disease, heart attacks, angina, and heart failure.
        
        **Common Symptoms You Can Identify:**
        
        - Chest pain, tightness, pressure, or discomfort (angina)
        - Shortness of breath during activities or at rest
        - Pain radiating to arms, shoulders, neck, jaw, or back
        - Irregular heartbeat or heart palpitations
        - Dizziness, lightheadedness, or fainting
        - Persistent fatigue and weakness
        - Swelling in legs, ankles, feet, or abdomen
        - Rapid or slow heart rate
        - Nausea or lack of appetite
        - Cold sweats
        
        **Major Risk Factors:**
        
        - High blood pressure (hypertension)
        - High cholesterol levels
        - Smoking and tobacco use
        - Diabetes or pre-diabetes
        - Obesity and physical inactivity
        - Family history of heart disease
        - Age (men 45+, women 55+)
        - Chronic stress and poor sleep
        """)

    st.write("---")
    st.subheader("Enter Your Medical Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=50)
        sex = st.selectbox("Sex", ("Male", "Female"), index=0)
        cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"), index=3)
        with st.expander("ℹ️ Help"):
            st.write("Typical Angina: Exercise-related chest pain | Atypical: Chest pain not related to exertion | Non-anginal: Chest discomfort unrelated to heart | Asymptomatic: No chest pain")

    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1, value=120)
        with st.expander("ℹ️ Help"):
            st.write("Systolic BP (upper number). Normal: <120 | Elevated: 120-129 | High: ≥130")
        
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, step=1, value=200)
        with st.expander("ℹ️ Help"):
            st.write("Total cholesterol. Desirable: <200 | Borderline: 200-239 | High: ≥240 mg/dL")
        
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("True", "False"), index=1)

    with col3:
        restecg = st.selectbox("Resting ECG Results", ("Normal", "ST-T wave abnormality", "Probable/definite left ventricular hypertrophy"), index=0)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=220, step=1, value=150)
        with st.expander("ℹ️ Help"):
            st.write("Measured during stress test. Rough estimate: 220 - Your Age")
        
        exang = st.selectbox("Exercise Induced Angina", ("Yes", "No"), index=1)

    st.write("---")
    st.subheader("Advanced ECG & Test Details")
    col_adv1, col_adv2, col_adv3 = st.columns(3)

    with col_adv1:
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    with col_adv2:
        slope = st.selectbox("Slope of Peak Exercise ST", ("Upsloping", "Flat", "Downsloping"), index=1)
    with col_adv3:
        ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1, value=0)
        thal = st.selectbox("Thalassemia", ("Normal", "Fixed defect", "Reversable defect"), index=1)

    st.write("")
    
    if st.button("🔍 Analyze Heart Disease Risk", key="heart_predict"):
        with st.spinner("Analyzing your cardiovascular data..."):
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
        st.subheader("Assessment Result")
        
        if prediction[0] == 0:
            st.success("✅ **Low Risk for Heart Disease**")
            st.balloons()
            
            st.write("**Excellent!** Your cardiovascular parameters indicate a low risk.")
            st.write("")
            st.write("**Recommended Preventive Actions:**")
            st.write("""
            - Maintain a heart-healthy diet (Mediterranean or DASH diet)
            - Engage in regular aerobic exercise (150 minutes per week)
            - Keep blood pressure below 120/80 mm Hg
            - Maintain total cholesterol below 200 mg/dL
            - Avoid smoking and limit alcohol (1-2 drinks/day max)
            - Manage stress through yoga, meditation, or hobbies
            - Get 7-9 hours of quality sleep nightly
            - Schedule annual cardiovascular check-ups after age 40
            - Maintain healthy body weight (BMI 18.5-24.9)
            """)
        else:
            st.error("⚠️ **High Risk for Heart Disease**")
            
            st.write("**Critical:** Your assessment indicates elevated cardiovascular risk.")
            st.write("")
            
            st.write("**🏥 URGENT MEDICAL ACTION REQUIRED:**")
            st.write("""
            - **See a cardiologist within 1-2 weeks (or sooner if symptoms present)**
            - Request comprehensive cardiac evaluation:
              - ECG (Electrocardiogram)
              - Echocardiogram (heart ultrasound)
              - Cardiac stress test
              - Lipid panel and metabolic tests
            - Discuss all your results and medical history
            """)
            
            st.write("")
            st.write("**💊 Likely Medical Prescriptions:**")
            st.write("""
            Your cardiologist may prescribe:
            
            **For High Cholesterol:**
            - Atorvastatin (Lipitor) 10-80mg daily
            - Rosuvastatin (Crestor) 5-40mg daily
            
            **For Blood Pressure:**
            - ACE Inhibitors (Lisinopril, Enalapril)
            - Beta-blockers (Metoprolol, Atenolol)
            - Calcium Channel Blockers (Amlodipine)
            
            **For Blood Clot Prevention:**
            - Aspirin 75-100mg daily (low-dose)
            - Clopidogrel if aspirin intolerant
            
            **For Angina:**
            - Nitroglycerin (sublingual) as needed
            
            ⚠️ Take all medications exactly as prescribed. Do not stop without consulting your doctor.
            """)
            
            st.write("")
            st.write("**🍎 CRITICAL Dietary Changes:**")
            st.write("""
            **STRICTLY AVOID:**
            - Trans fats and hydrogenated oils
            - Saturated fats (fatty meats, full-fat dairy)
            - Excessive salt (limit to 1500-2000mg/day)
            - Processed and deli meats
            - Fried and fast foods
            - Sugary drinks and excess sugar
            
            **REDUCE:**
            - Red meat (limit to once weekly)
            - Egg yolks (max 3-4 per week)
            - Butter and cheese
            - Baked goods and pastries
            - Alcohol consumption
            
            **INCREASE:**
            - Fatty fish (salmon, mackerel, sardines) - 2-3 times/week
            - Nuts and seeds (almonds, walnuts, flaxseeds)
            - Extra virgin olive oil
            - Colorful vegetables and leafy greens
            - Whole grains (oats, quinoa, brown rice)
            - Legumes (beans, lentils, chickpeas)
            - Berries and citrus fruits
            """)
            
            st.write("")
            st.write("**🏃 Essential Lifestyle Modifications:**")
            st.write("""
            - **STOP SMOKING IMMEDIATELY** - single most important step
              - Seek smoking cessation program or nicotine replacement therapy
            
            - **Exercise Plan** (consult doctor before starting):
              - Start with gentle walking (15-20 minutes daily)
              - Gradually increase to 30-45 minutes most days
              - Add light resistance training 2-3 times weekly
              - Avoid sudden intense exertion
            
            - **Weight Management:**
              - If overweight, aim to lose 5-10% of body weight
              - This can significantly reduce cardiovascular risk
            
            - **Stress Reduction:**
              - Practice daily relaxation (meditation, deep breathing)
              - Consider cognitive behavioral therapy
              - Ensure adequate sleep (7-9 hours)
              - Limit caffeine intake
            
            - **Alcohol:** Maximum 1 drink/day (women) or 2/day (men)
            """)
            
            st.write("")
            st.write("**📊 Regular Monitoring Required:**")
            st.write("""
            - Monitor blood pressure daily (keep a log)
            - Track any new or worsening symptoms
            - Follow-up with cardiologist every 3-6 months
            - Annual or bi-annual cardiac stress tests
            - Regular lipid panel blood tests
            - Consider home BP monitor and pulse oximeter
            """)
            
            st.write("")
            st.write("**🚨 EMERGENCY WARNING SIGNS - CALL 911 IMMEDIATELY IF:**")
            st.write("""
            - Severe chest pain or pressure lasting more than 5 minutes
            - Pain radiating to arm, jaw, neck, or back
            - Severe shortness of breath
            - Cold sweats with chest discomfort
            - Sudden extreme weakness or loss of consciousness
            - Nausea/vomiting with chest pain
            - Irregular or very rapid heartbeat with dizziness
            
            **DO NOT DRIVE YOURSELF - Call emergency services immediately**
            """)
            
            st.write("")
            st.info("**Important:** Heart disease is highly manageable with proper treatment. Strict adherence to medical advice can significantly improve outcomes and quality of life.")


# --- PARKINSON'S PREDICTION PAGE ---
elif selection == "🧠 Parkinson's":
    st.title("Parkinson's Disease Risk Assessment")
    
    with st.expander("📚 About Parkinson's Disease & Common Symptoms", expanded=False):
        st.write("""
        **What is Parkinson's Disease?**
        
        Parkinson's Disease is a progressive neurological disorder that affects movement control due to 
        loss of dopamine-producing brain cells. It typically develops gradually over years.
        
        **Early Symptoms You Can Identify:**
        
        - Tremor (shaking) - usually starts in hands or fingers, often when at rest
        - Slowed movement (bradykinesia) - difficulty initiating movement
        - Muscle stiffness and rigidity throughout the body
        - Impaired posture and balance - stooped posture
        - Loss of automatic movements (reduced blinking, arm swing when walking)
        - Speech changes (softer voice, slurred, monotone, hesitant)
        - Writing changes (micrographia - smaller handwriting)
        - Difficulty with fine motor tasks (buttoning shirts, tying shoes)
        - Reduced facial expression (mask-like face)
        - Shuffling walk with small steps
        
        **Non-Motor Symptoms:**
        
        - Sleep disturbances and daytime drowsiness
        - Depression and anxiety
        - Constipation
        - Loss of sense of smell
        - Cognitive changes
        
        **Risk Factors:**
        
        - Age 60+ (most common onset)
        - Male gender (1.5x more likely than women)
        - Family history and genetics
        - Exposure to pesticides or herbicides
        - Head trauma history
        """)
    
    st.warning("""
    ⚠️ **IMPORTANT TECHNICAL NOTICE**
    
    This model analyzes **advanced voice analysis metrics** that CANNOT be measured at home.
    
    **These parameters require:**
    - Professional speech-language pathologist evaluation
    - Specialized voice analysis equipment and software
    - Clinical neurological voice assessment
    
    The values provided below are **standard defaults for demonstration purposes**.
    
    **For accurate prediction, you MUST use actual medical report data from a voice analysis test.**
    """)

    st.write("---")
    st.subheader("Voice Analysis Parameters (From Medical Report)")

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

    if st.button("🔍 Analyze Parkinson's Risk", key="parkinsons_predict"):
        with st.spinner("Analyzing vocal metrics..."):
            time.sleep(1.5)
            
            input_data = [
                fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]
            
            std_data = parkinsons_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = parkinsons_model.predict(std_data)
        
        st.write("---")
        st.subheader("Assessment Result")
        
        if prediction[0] == 0:
            st.success("✅ **Low Risk for Parkinson's Disease**")
            st.balloons()
            
            st.write("**Good News!** Voice analysis metrics suggest low risk for Parkinson's disease.")
            st.write("")
            st.write("**Recommended Preventive Actions:**")
            st.write("""
            - Continue regular health check-ups and neurological screenings
            - Stay physically active with regular exercise (especially aerobic)
            - Engage in mentally stimulating activities (reading, puzzles, learning)
            - Maintain strong social connections and relationships
            - Follow a brain-healthy diet rich in antioxidants
            - Get adequate quality sleep (7-9 hours nightly)
            - Avoid exposure to pesticides and environmental toxins
            - If you have family history, discuss genetic counseling with doctor
            - Monitor for any new symptoms as you age, especially after 60
            """)
        else:
            st.error("⚠️ **High Risk for Parkinson's Disease**")
            
            st.write("**Important:** Voice analysis indicates elevated risk for Parkinson's disease.")
            st.write("")
            
            st.write("**🏥 IMMEDIATE MEDICAL ACTION REQUIRED:**")
            st.write("""
            - **Schedule appointment with neurologist immediately (within 1-2 weeks)**
            - Request comprehensive neurological examination including:
              - Detailed movement and motor function assessment
              - Cognitive and mental status evaluation
              - Possible brain imaging (MRI or CT scan)
              - DaTscan (dopamine transporter scan) if warranted
            - Consider consultation with movement disorder specialist
            - Discuss voice analysis results and any symptoms you've noticed
            - Bring list of all medications you're currently taking
            """)
            
            st.write("")
            st.write("**💊 Medical Treatment Options (If Diagnosed):**")
            st.write("""
            **Early diagnosis allows for better treatment outcomes. Your neurologist may prescribe:**
            
            **First-Line Medications:**
            - **Levodopa/Carbidopa (Sinemet)** - Gold standard treatment
              - Most effective for motor symptoms
              - Dosage typically starts low and increases gradually
              - Take 30-60 minutes before meals for best absorption
            
            **Dopamine Agonists:**
            - Pramipexole (Mirapex) 0.375-4.5mg daily
            - Ropinirole (Requip) 0.25-24mg daily
            - Often used in younger patients or early stages
            
            **MAO-B Inhibitors:**
            - Selegiline (Eldepryl) 5-10mg daily
            - Rasagiline (Azilect) 0.5-1mg daily
            - May slow disease progression
            
            **For Tremor Control:**
            - Anticholinergics (Trihexyphenidyl, Benztropine)
            - Primarily for tremor-dominant Parkinson's
            
            **Important:** Medication timing is critical. Follow prescriptions exactly and report all side effects.
            """)
            
            st.write("")
            st.write("**🏃 CRITICAL: Exercise & Physical Therapy:**")
            st.write("""
            **Research shows exercise significantly slows Parkinson's progression!**
            
            **Recommended Activities (30-60 minutes, 4-5 days/week):**
            - Brisk walking or treadmill
            - Swimming or aquatic therapy
            - Cycling (stationary or outdoor)
            - Tai Chi for balance and flexibility
            - Yoga for strength and posture
            - Dancing (very beneficial - Rock Steady Boxing program)
            - Strength training 2-3 times weekly
            
            **Physical Therapy Focus Areas:**
            - Gait training and balance exercises
            - Flexibility and range of motion
            - Posture correction and core strengthening
            - Fall prevention strategies
            - Transfer techniques (getting up from chair, bed)
            
            **Occupational Therapy:**
            - Fine motor skills maintenance
            - Adaptive strategies for daily activities
            - Home safety assessment and modifications
            - Assistive device recommendations
            """)
            
            st.write("")
            st.write("**🗣️ Speech & Swallowing Therapy:**")
            st.write("""
            **LSVT LOUD Program (Highly Recommended):**
            - Specialized speech therapy for Parkinson's
            - Improves voice volume and clarity
            - 4 sessions per week for 4 weeks
            - Proven effective for voice symptoms
            
            **Additional Speech Support:**
            - Regular speech therapy to maintain vocal function
            - Swallowing evaluation if experiencing difficulties
            - Voice exercises and vocal strengthening
            - Communication strategies and tools
            """)
            
            st.write("")
            st.write("**🍎 Dietary Recommendations:**")
            st.write("""
            **INCREASE:**
            - Antioxidant-rich foods:
              - Berries (blueberries, strawberries)
              - Leafy greens (kale, spinach)
              - Nuts (walnuts, almonds)
              - Green tea
            - Mediterranean diet pattern
            - Omega-3 fatty acids (salmon, flaxseeds)
            - High-fiber foods (to prevent constipation)
            - Adequate hydration (8+ glasses water daily)
            
            **IMPORTANT for Levodopa Users:**
            - Take medication 30-60 minutes before meals
            - Avoid high-protein meals close to medication time
            - Protein can interfere with levodopa absorption
            - Distribute protein intake throughout the day
            
            **Consider Supplements (Consult Doctor):**
            - Vitamin D (many Parkinson's patients are deficient)
            - Coenzyme Q10 (may have neuroprotective effects)
            - Omega-3 supplements
            - B-complex vitamins
            """)
            
            st.write("")
            st.write("**🧠 Mental Health & Cognitive Support:**")
            st.write("""
            **Depression and anxiety are common in Parkinson's:**
            - Screen regularly for mood changes
            - Consider counseling or cognitive behavioral therapy
            - Antidepressants may be prescribed if needed
            - Join support groups (local or online)
            
            **Cognitive Stimulation:**
            - Engage in mentally challenging activities
            - Learn new skills or hobbies
            - Play strategy games and puzzles
            - Maintain social interactions
            - Consider cognitive training programs
            """)
            
            st.write("")
            st.write("**🏠 Home Safety & Lifestyle Adaptations:**")
            st.write("""
            **Fall Prevention:**
            - Remove tripping hazards (rugs, cords, clutter)
            - Install grab bars in bathroom
            - Improve lighting throughout home
            - Use non-slip mats in shower/tub
            - Keep floors clear and dry
            - Consider medical alert system
            
            **Daily Living Aids:**
            - Use assistive devices as needed (cane, walker)
            - Install raised toilet seat
            - Use adaptive utensils for eating
            - Button hooks and zipper pulls
            - Velcro closures instead of buttons
            
            **Sleep Hygiene:**
            - Maintain regular sleep schedule
            - Create comfortable sleep environment
            - Address REM sleep behavior disorder if present
            - Discuss sleep medications with doctor if needed
            
            **Stress Management:**
            - Practice relaxation techniques (meditation, deep breathing)
            - Engage in enjoyable activities
            - Maintain hobbies and interests
            - Stay connected with friends and family
            """)
            
            st.write("")
            st.write("**📊 Monitoring & Follow-Up:**")
            st.write("""
            - Keep symptom diary (track tremor, stiffness, mobility, mood)
            - Regular neurological evaluations (every 3-6 months)
            - Monitor medication effectiveness and side effects
            - Report any sudden worsening of symptoms
            - Track "on" and "off" periods if on Levodopa
            - Stay informed about new treatments and clinical trials
            """)
            
            st.write("")
            st.write("**🚨 Warning Signs Requiring Immediate Medical Attention:**")
            st.write("""
            - Sudden severe worsening of symptoms
            - Difficulty swallowing or choking episodes
            - Severe confusion, hallucinations, or delusions
            - Inability to move (freezing) for extended periods
            - Frequent falls or severe balance problems
            - Severe rigidity preventing movement
            - Signs of medication overdose (dyskinesia - involuntary movements)
            - Neuroleptic malignant syndrome (fever, confusion, rigid muscles)
            """)
            
            st.write("")
            st.write("**📚 Important Resources & Support:**")
            st.write("""
            - **Parkinson's Foundation:** www.parkinson.org | Helpline: 1-800-4PD-INFO
            - **Michael J. Fox Foundation:** www.michaeljfox.org
            - **American Parkinson Disease Association:** www.apdaparkinson.org
            - **Davis Phinney Foundation:** www.davisphinneyfoundation.org
            - Look for local support groups and exercise programs
            - Consider joining clinical trials through Fox Trial Finder
            """)
            
            st.write("")
            st.info("**Critical Message:** Parkinson's disease progression varies greatly among individuals. With early diagnosis, proper treatment, and especially **consistent exercise**, many people maintain excellent quality of life for years. The actions you take now make a significant difference in your long-term outcomes.")


# --- FOOTER ---
st.write("---")
st.markdown("""
<div style='text-align: center; color: #4a5568; padding: 2rem; margin-top: 2rem;'>
    <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>GuardianHealth AI Prediction System</p>
    <p style='font-size: 0.95rem; margin-bottom: 0.5rem;'>⚕️ Always consult qualified healthcare professionals for medical decisions</p>
    <p style='font-size: 0.85rem; color: #718096;'>© 2024 GuardianHealth | For Educational and Informational Purposes Only</p>
    <p style='font-size: 0.8rem; color: #a0aec0; margin-top: 1rem;'>This tool provides risk assessments based on machine learning models and should not replace professional medical advice, diagnosis, or treatment.</p>
</div>
""", unsafe_allow_html=True)