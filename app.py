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

# --- ENHANCED CSS WITH ALL FIXES ---
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        /* ========================= GLOBAL RESET ========================= */
        * {
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        
        /* ========================= BACKGROUND ========================= */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* ========================= MAIN CONTAINER ========================= */
        .main .block-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 30px;
            padding: 3rem 2.5rem;
            margin: 2rem auto;
            max-width: 1400px;
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* ========================= TYPOGRAPHY - DARKER FOR BETTER READABILITY ========================= */
        h1 {
            color: #1a202c !important;
            font-weight: 800 !important;
            font-size: 3rem !important;
            margin-bottom: 1rem !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2 {
            color: #2d3748 !important;
            font-weight: 700 !important;
            font-size: 1.9rem !important;
            margin-top: 2rem !important;
            margin-bottom: 1.2rem !important;
        }
        
        h3 {
            color: #4a5568 !important;
            font-weight: 600 !important;
            font-size: 1.4rem !important;
            margin-top: 1.5rem !important;
        }
        
        p, li, label, div {
            color: #1a202c !important;
            font-size: 1rem !important;
            line-height: 1.8 !important;
        }
        
        /* ========================= SIDEBAR - WIDER & BETTER VISIBILITY ========================= */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
            border-right: none;
            width: 350px !important;
            min-width: 350px !important;
        }
        
        /* Fix sidebar content container width */
        [data-testid="stSidebar"] > div:first-child {
            width: 350px !important;
            min-width: 350px !important;
        }
        
        [data-testid="stSidebar"] h1 {
            color: #ffffff !important;
            text-align: center;
            padding: 1.5rem 1rem;
            -webkit-text-fill-color: white !important;
            font-size: 1.8rem !important;
            line-height: 1.3 !important;
        }
        
        /* HIDE "keyboard_double_arrow_right" text for sidebar toggle */
        [data-testid="stSidebar"] button[kind="secondary"] span {
            font-size: 0 !important;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"] span::before {
            content: "≡";
            font-size: 2rem;
            font-weight: bold;
            color: #ffffff;
            display: inline-block;
        }
        
        /* Style the collapse button */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            padding: 0.8rem !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
            transform: scale(1.1);
        }
        
        /* Main area sidebar toggle button */
        button[kind="secondary"]:not([data-testid="stSidebar"] button) {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.8rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        button[kind="secondary"]:not([data-testid="stSidebar"] button):hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Hide keyboard arrow text and replace with icon */
        button[kind="secondary"] span {
            font-size: 0 !important;
        }
        
        button[kind="secondary"] span::before {
            content: "☰";
            font-size: 1.5rem;
            font-weight: bold;
            color: #ffffff;
            display: inline-block;
        }
        
        /* ========================= RADIO BUTTONS - HIGH CONTRAST TEXT ========================= */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.8rem;
        }
        
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 16px !important;
            padding: 1.2rem 1rem !important;
            margin: 0.4rem 0 !important;
            cursor: pointer !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            text-align: center !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 1.15rem !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Make text in radio buttons highly visible */
        [data-testid="stSidebar"] div[role="radiogroup"] label p,
        [data-testid="stSidebar"] div[role="radiogroup"] label div,
        [data-testid="stSidebar"] div[role="radiogroup"] label span {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.4) 0%, rgba(118, 75, 162, 0.4) 100%) !important;
            border-color: rgba(255, 255, 255, 0.5) !important;
            transform: translateX(8px) scale(1.02) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
        }
        
        /* Hide radio circle */
        [data-testid="stSidebar"] input[type="radio"] {
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            position: absolute !important;
        }
        
        /* Active state - when radio is checked */
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: #ffffff !important;
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.7) !important;
            transform: translateX(10px) scale(1.05) !important;
        }
        
        /* Ensure active text is also white */
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div,
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
            color: #ffffff !important;
        }
        
        /* ========================= EXPANDER ARROWS - REPLACE TEXT WITH ICONS ========================= */
        /* Hide "keyboard_arrow_right" text */
        .streamlit-expanderHeader svg {
            display: none !important;
        }
        
        summary::-webkit-details-marker {
            display: none !important;
        }
        
        .streamlit-expanderHeader::after {
            content: "▶";
            position: absolute;
            right: 1rem;
            font-size: 1.2rem;
            color: #4a5568;
            transition: transform 0.3s ease;
        }
        
        details[open] .streamlit-expanderHeader::after {
            content: "▼";
        }
        
        /* ========================= BUTTONS ========================= */
        .stButton > button {
            width: 100%;
            border: none;
            border-radius: 16px;
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.3rem 2rem;
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            margin-top: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button:before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton > button:hover:before {
            width: 400px;
            height: 400px;
        }
        
        .stButton > button:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        .stButton > button:active {
            transform: translateY(-2px);
        }
        
        /* ========================= INPUT FIELDS ========================= */
        [data-testid="stNumberInput"] input,
        [data-baseweb="input"] input {
            border: 2px solid #cbd5e0 !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            font-size: 1rem !important;
            color: #1a202c !important;
            background: #ffffff !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stNumberInput"] input:focus,
        [data-baseweb="input"] input:focus {
            border-color: #667eea !important;
            background: #ffffff !important;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
            outline: none !important;
            transform: scale(1.01);
        }
        
        /* Labels - Darker for better readability */
        label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
            color: #1a202c !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            margin-bottom: 0.6rem !important;
        }
        
        /* Selectbox */
        [data-baseweb="select"] > div {
            border: 2px solid #cbd5e0 !important;
            border-radius: 12px !important;
            background: #ffffff !important;
            transition: all 0.3s ease !important;
        }
        
        [data-baseweb="select"] > div:hover {
            border-color: #a0aec0 !important;
        }
        
        [data-baseweb="select"] > div:focus-within {
            border-color: #667eea !important;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
        }
        
        /* Selectbox text */
        [data-baseweb="select"] div {
            color: #1a202c !important;
        }
        
        /* ========================= CARDS ========================= */
        .disease-card {
            background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #667eea;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .disease-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            border-left-width: 8px;
        }
        
        .disease-card h3 {
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
            color: #2d3748 !important;
        }
        
        .disease-card p {
            color: #4a5568 !important;
        }
        
        /* ========================= RESULT BOXES ========================= */
        [data-testid="stSuccess"] {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
            border-left: 6px solid #28a745 !important;
            border-radius: 16px !important;
            padding: 1.8rem !important;
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2) !important;
            animation: slideIn 0.5s ease;
        }
        
        [data-testid="stSuccess"] p, [data-testid="stSuccess"] div {
            color: #155724 !important;
        }
        
        [data-testid="stError"] {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%) !important;
            border-left: 6px solid #dc3545 !important;
            border-radius: 16px !important;
            padding: 1.8rem !important;
            box-shadow: 0 8px 25px rgba(220, 53, 69, 0.2) !important;
            animation: slideIn 0.5s ease;
        }
        
        [data-testid="stError"] p, [data-testid="stError"] div {
            color: #721c24 !important;
        }
        
        [data-testid="stWarning"] {
            background: linear-gradient(135deg, #fff3cd 0%, #ffecb5 100%) !important;
            border-left: 6px solid #ffc107 !important;
            border-radius: 16px !important;
            padding: 1.8rem !important;
            box-shadow: 0 8px 25px rgba(255, 193, 7, 0.2) !important;
            animation: slideIn 0.5s ease;
        }
        
        [data-testid="stWarning"] p, [data-testid="stWarning"] div {
            color: #856404 !important;
        }
        
        [data-testid="stInfo"] {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%) !important;
            border-left: 6px solid #17a2b8 !important;
            border-radius: 16px !important;
            padding: 1.8rem !important;
            box-shadow: 0 8px 25px rgba(23, 162, 184, 0.2) !important;
            animation: slideIn 0.5s ease;
        }
        
        [data-testid="stInfo"] p, [data-testid="stInfo"] div {
            color: #0c5460 !important;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* ========================= EXPANDERS ========================= */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            color: #1a202c !important;
            border: 2px solid #cbd5e0 !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            padding-right: 3rem !important;
        }
        
        .streamlit-expanderHeader p {
            color: #1a202c !important;
            font-weight: 600 !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%) !important;
            border-color: #a0aec0 !important;
            transform: scale(1.01);
        }
        
        .streamlit-expanderContent {
            background: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-top: none !important;
            border-radius: 0 0 12px 12px !important;
            padding: 1.5rem !important;
        }
        
        .streamlit-expanderContent p, .streamlit-expanderContent li {
            color: #2d3748 !important;
        }
        
        /* ========================= DIVIDER ========================= */
        hr {
            margin: 3rem 0 !important;
            border: none !important;
            height: 3px !important;
            background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent) !important;
            opacity: 0.3;
        }
        
        /* ========================= SPINNER ========================= */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* ========================= SCROLLBAR ========================= */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
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
        
        /* ========================= BMI DISPLAY ========================= */
        .bmi-display {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 700;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            margin: 1rem 0;
            animation: pulse 2s ease infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        /* ========================= SIDEBAR INFO BOX ========================= */
        [data-testid="stSidebar"] [data-testid="stInfo"] {
            background: rgba(255, 255, 255, 0.1) !important;
            border-left-color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stInfo"] p,
        [data-testid="stSidebar"] [data-testid="stInfo"] div {
            color: #e2e8f0 !important;
        }
        
        /* ========================= MOBILE RESPONSIVE ========================= */
        @media (max-width: 768px) {
            h1 { font-size: 2rem !important; }
            h2 { font-size: 1.5rem !important; }
            .main .block-container { padding: 2rem 1.5rem; }
            [data-testid="stSidebar"] {
                width: 300px !important;
                min-width: 300px !important;
            }
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
    st.error("⚠️ Model files not found. Please ensure all .pkl files are in the root directory.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")
    st.stop()

# --- NAVIGATION ---
st.sidebar.title("🩺 GuardianHealth")
st.sidebar.write("---")

# Direct radio button
selection = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🍬 Diabetes", "❤️ Heart Disease", "🧠 Parkinson's"],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.write("---")
st.sidebar.info(
    "**⚕️ Medical Disclaimer**\n\n"
    "This tool is for educational and informational purposes only. "
    "Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment."
)

# --- DISPLAY SELECTED PAGE ---
# Now 'selection' comes directly from radio button, not session state

# --- HOME PAGE ---
if selection == "🏠 Home":
    st.title("🩺 GuardianHealth: AI-Powered Disease Prediction")
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
    
    st.subheader("✨ How It Works")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("""
        **🎯 Step 1: Select a Disease Model**
        - Choose from the sidebar navigation
        - Review disease information and symptoms
        
        **📝 Step 2: Enter Medical Parameters**
        - Fill in the required fields
        - Use help guides for each parameter
        - Standard values provided as defaults
        """)
    
    with col_b:
        st.write("""
        **🤖 Step 3: Get Your Assessment**
        - AI analyzes your data instantly
        - Receive comprehensive risk prediction
        - View personalized recommendations
        
        **💡 Step 4: Take Action**
        - Follow medical advice provided
        - Consult healthcare professionals
        - Implement lifestyle changes
        """)
    
    st.write("---")
    
    st.warning("⚠️ **Medical Disclaimer:** These predictions are statistical assessments based on machine learning models. They are NOT medical diagnoses and should not replace professional medical advice, diagnosis, or treatment.")

# --- DIABETES PREDICTION PAGE ---
elif selection == "🍬 Diabetes":
    st.title("🍬 Type 2 Diabetes Risk Assessment")
    
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
        - Darkened skin patches in armpits or neck
        - Dry, itchy skin
        
        **Major Risk Factors:**
        
        - Obesity or being overweight (BMI > 25)
        - Family history of diabetes
        - Age over 45 years
        - Physical inactivity
        - High blood pressure (>130/80 mm Hg)
        - History of gestational diabetes
        """)

    st.write("---")
    st.subheader("📊 Enter Your Medical Parameters")

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
    
    st.subheader("⚖️ BMI Calculator")
    
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
        st.markdown(f"""
        <div class="bmi-display">
            BMI: {calculated_bmi:.2f} - {bmi_category}
        </div>
        """, unsafe_allow_html=True)
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
        with st.spinner("🧪 Analyzing your health data..."):
            time.sleep(1.5)
            
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            std_data = diabetes_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = diabetes_model.predict(std_data)
        
        st.write("---")
        st.subheader("📋 Assessment Result")
        
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
            - Manage stress through meditation, yoga, or relaxation techniques
            - If you have family history, consider more frequent glucose screening
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
            - **Weight loss goal:** If overweight, aim to lose 5-10% of body weight
            - This alone can significantly reduce diabetes risk
            """)
            
            st.write("")
            st.info("**Remember:** Type 2 Diabetes is preventable and manageable with early intervention. The steps you take today can significantly improve your health outcomes.")

# --- HEART DISEASE PREDICTION PAGE ---
elif selection == "❤️ Heart Disease":
    st.title("❤️ Heart Disease Risk Assessment")
    
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
        
        **Major Risk Factors:**
        
        - High blood pressure (hypertension)
        - High cholesterol levels
        - Smoking and tobacco use
        - Diabetes or pre-diabetes
        - Obesity and physical inactivity
        - Family history of heart disease
        - Age (men 45+, women 55+)
        """)

    st.write("---")
    st.subheader("📊 Enter Your Medical Parameters")

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
    st.subheader("🔬 Advanced ECG & Test Details")
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
        with st.spinner("🩺 Analyzing your cardiovascular data..."):
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
        st.subheader("📋 Assessment Result")
        
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
            
            **For Blood Clot Prevention:**
            - Aspirin 75-100mg daily (low-dose)
            
            ⚠️ Take all medications exactly as prescribed.
            """)
            
            st.write("")
            st.write("**🍎 CRITICAL Dietary Changes:**")
            st.write("""
            **STRICTLY AVOID:**
            - Trans fats and hydrogenated oils
            - Saturated fats (fatty meats, full-fat dairy)
            - Excessive salt (limit to 1500-2000mg/day)
            - Processed and deli meats
            
            **INCREASE:**
            - Fatty fish (salmon, mackerel) - 2-3 times/week
            - Nuts and seeds (almonds, walnuts)
            - Extra virgin olive oil
            - Colorful vegetables and leafy greens
            - Whole grains and legumes
            """)
            
            st.write("")
            st.write("**🚨 EMERGENCY WARNING SIGNS - CALL 911 IMMEDIATELY IF:**")
            st.write("""
            - Severe chest pain or pressure lasting more than 5 minutes
            - Pain radiating to arm, jaw, neck, or back
            - Severe shortness of breath
            - Cold sweats with chest discomfort
            
            **DO NOT DRIVE YOURSELF - Call emergency services immediately**
            """)

# --- PARKINSON'S PREDICTION PAGE ---
elif selection == "🧠 Parkinson's":
    st.title("🧠 Parkinson's Disease Risk Assessment")
    
    with st.expander("📚 About Parkinson's Disease & Common Symptoms", expanded=False):
        st.write("""
        **What is Parkinson's Disease?**
        
        Parkinson's Disease is a progressive neurological disorder that affects movement control due to 
        loss of dopamine-producing brain cells.
        
        **Early Symptoms You Can Identify:**
        
        - Tremor (shaking) - usually starts in hands or fingers at rest
        - Slowed movement (bradykinesia)
        - Muscle stiffness and rigidity
        - Impaired posture and balance
        - Speech changes (softer voice, slurred, monotone)
        - Writing changes (smaller handwriting)
        - Reduced facial expression
        
        **Risk Factors:**
        
        - Age 60+ (most common onset)
        - Male gender (1.5x more likely than women)
        - Family history and genetics
        - Exposure to pesticides
        """)
    
    st.warning("""
    ⚠️ **IMPORTANT TECHNICAL NOTICE**
    
    This model analyzes **advanced voice analysis metrics** that CANNOT be measured at home.
    
    **These parameters require:**
    - Professional speech-language pathologist evaluation
    - Specialized voice analysis equipment
    - Clinical neurological voice assessment
    
    The values below are **standard defaults for demonstration**. For accurate prediction, you MUST use actual medical report data from a voice analysis test.
    """)

    st.write("---")
    st.subheader("🎤 Voice Analysis Parameters (From Medical Report)")

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
        with st.spinner("🧠 Analyzing vocal metrics..."):
            time.sleep(1.5)
            
            input_data = [
                fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]
            
            std_data = parkinsons_scaler.transform(np.asarray(input_data).reshape(1, -1))
            prediction = parkinsons_model.predict(std_data)
        
        st.write("---")
        st.subheader("📋 Assessment Result")
        
        if prediction[0] == 0:
            st.success("✅ **Low Risk for Parkinson's Disease**")
            st.balloons()
            
            st.write("**Good News!** Voice analysis metrics suggest low risk for Parkinson's disease.")
            st.write("")
            st.write("**Recommended Preventive Actions:**")
            st.write("""
            - Continue regular health check-ups and neurological screenings
            - Stay physically active with regular exercise
            - Engage in mentally stimulating activities
            - Maintain strong social connections
            - Follow a brain-healthy diet rich in antioxidants
            - Get adequate quality sleep (7-9 hours nightly)
            - Avoid exposure to pesticides and environmental toxins
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
            """)
            
            st.write("")
            st.write("**💊 Medical Treatment Options (If Diagnosed):**")
            st.write("""
            **First-Line Medications:**
            - **Levodopa/Carbidopa (Sinemet)** - Gold standard treatment
              - Most effective for motor symptoms
              - Take 30-60 minutes before meals for best absorption
            
            **Dopamine Agonists:**
            - Pramipexole (Mirapex) 0.375-4.5mg daily
            - Ropinirole (Requip) 0.25-24mg daily
            
            **MAO-B Inhibitors:**
            - Selegiline (Eldepryl) 5-10mg daily
            - Rasagiline (Azilect) 0.5-1mg daily
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
            - Dancing (Rock Steady Boxing program)
            - Strength training 2-3 times weekly
            """)
            
            st.write("")
            st.write("**🍎 Dietary Recommendations:**")
            st.write("""
            **INCREASE:**
            - Antioxidant-rich foods (berries, leafy greens, nuts)
            - Mediterranean diet pattern
            - Omega-3 fatty acids (salmon, flaxseeds)
            - High-fiber foods (to prevent constipation)
            
            **IMPORTANT for Levodopa Users:**
            - Take medication 30-60 minutes before meals
            - Avoid high-protein meals close to medication time
            - Protein can interfere with absorption
            """)
            
            st.write("")
            st.info("**Critical Message:** Parkinson's disease progression varies greatly. With early diagnosis, proper treatment, and especially **consistent exercise**, many people maintain excellent quality of life for years.")

# --- FOOTER ---
st.write("---")
st.markdown("""
<div style='text-align: center; color: #4a5568; padding: 2rem; margin-top: 2rem;'>
    <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>GuardianHealth AI Prediction System</p>
    <p style='font-size: 0.95rem; margin-bottom: 0.5rem;'>⚕️ Always consult qualified healthcare professionals for medical decisions</p>
    <p style='font-size: 0.85rem; color: #718096;'>© 2024 GuardianHealth | For Educational and Informational Purposes Only</p>
</div>
""", unsafe_allow_html=True)