import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config FIRST
st.set_page_config(page_title="NEO Hazard Analysis", layout="wide", initial_sidebar_state="expanded")

# Injecting Premium CSS for a "Pro" UI
custom_css = """
<style>
    /* Main Background - Deep Space Gradient */
    .stApp {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
        color: #e0e0e0;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Hide top header line */
    header {visibility: hidden;}

    /* Glassmorphism for Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 20, 30, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Sidebar Text */
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Main Title Glow */
    h1 {
        text-shadow: 0 0 20px rgba(0, 201, 255, 0.4);
        margin-bottom: 0.5rem !important;
    }

    /* Primary Execute Button - Animated Gradient & Glow */
    .stButton > button {
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%) !important;
        color: #090a0f !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 0rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3) !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 201, 255, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Panel/Container Styling */
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }

    /* Customizing Alerts/Success/Error boxes */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(5px);
        color: white !important;
    }
    
    div.stAlert:has(div:contains("CLASSIFICATION: HAZARDOUS")) {
        background: linear-gradient(135deg, rgba(255, 75, 43, 0.2) 0%, rgba(255, 65, 108, 0.2) 100%) !important;
        border-left: 5px solid #FF416C !important;
        box-shadow: 0 0 20px rgba(255, 65, 108, 0.2) !important;
    }
    
    div.stAlert:has(div:contains("CLASSIFICATION: SAFE")) {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.2) 0%, rgba(56, 239, 125, 0.2) 100%) !important;
        border-left: 5px solid #38ef7d !important;
        box-shadow: 0 0 20px rgba(56, 239, 125, 0.2) !important;
    }
    
    /* Markdown text */
    p {
        color: #b3b9c5;
        font-size: 1.05rem;
    }
    
    /* Sliders Label Customization */
    .stSlider label {
        font-weight: 600 !important;
        color: #e0e0e0 !important;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
        color: #b3b9c5;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #00C9FF !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load the trained model and scaler
try:
    model = joblib.load('asteroid_model.pkl')
    scaler = joblib.load('asteroid_scaler.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'asteroid_model.pkl' is in the same directory.")
    st.stop()

# Recreate preprocessing to get the exact column names
df = pd.read_csv('nasa.csv')
cols_to_drop = ['Neo Reference ID', 'Name', 'Close Approach Date', 'Epoch Date Close Approach','Orbiting Body', 'Orbit Determination Date', 'Equinox', 'Orbit ID','Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Relative Velocity km per hour', 'Miles per hour','Miss Dist.(Astronomical)', 'Miss Dist.(Lunispatial)', 'Miss Dist.(Miles)', 'Hazardous']
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
feature_columns = X.columns.tolist()

st.title("NEO Hazard Analysis System")
st.markdown("Advanced probabilistic modeling for Near-Earth Object threat assessment based on orbital telemetry and physical characteristics.")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["TELEMETRY DASHBOARD", "SYSTEM ARCHITECTURE"])

with tab1:
    col1, col_space, col2 = st.columns([1.2, 0.1, 1.5])

    with col1:
        st.markdown("### Telemetry Inputs")
        st.markdown("<span style='color:#828a99; font-size: 0.9rem;'>Adjust the sliders to simulate NEO parameters</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        abs_magnitude = st.slider(
            "Absolute Magnitude (H)", 
            float(X['Absolute Magnitude'].min()), 
            float(X['Absolute Magnitude'].max()), 
            float(X['Absolute Magnitude'].median())
        )
        
        est_dia = st.slider(
            "Estimated Diameter Max (km)", 
            float(X['Est Dia in KM(max)'].min()), 
            float(X['Est Dia in KM(max)'].max()), 
            float(X['Est Dia in KM(max)'].median())
        )
        
        rel_velocity = st.slider(
            "Relative Velocity (km/s)", 
            float(X['Relative Velocity km per sec'].min()), 
            float(X['Relative Velocity km per sec'].max()), 
            float(X['Relative Velocity km per sec'].median())
        )
        
        miss_dist = st.slider(
            "Miss Distance (km)", 
            float(X['Miss Dist.(kilometers)'].min()), 
            float(X['Miss Dist.(kilometers)'].max()), 
            float(X['Miss Dist.(kilometers)'].median())
        )
        
        min_orbit_inter = st.slider(
            "Minimum Orbit Intersection (AU)", 
            float(X['Minimum Orbit Intersection'].min()), 
            float(X['Minimum Orbit Intersection'].max()), 
            float(X['Minimum Orbit Intersection'].median())
        )
        
        eccentricity = st.slider(
            "Orbital Eccentricity", 
            float(X['Eccentricity'].min()), 
            float(X['Eccentricity'].max()), 
            float(X['Eccentricity'].median())
        )

    # Build the input array with median values for the remaining unexposed features
    input_dict = {}
    for col in feature_columns:
        input_dict[col] = X[col].median()

    # Override with user inputs and synchronize mathematically dependent features
    if 'Absolute Magnitude' in input_dict:
        input_dict['Absolute Magnitude'] = abs_magnitude
    if 'Est Dia in KM(max)' in input_dict:
        input_dict['Est Dia in KM(max)'] = est_dia
        if 'Est Dia in KM(min)' in input_dict:
            input_dict['Est Dia in KM(min)'] = est_dia / 2.236
    if 'Relative Velocity km per sec' in input_dict:
        input_dict['Relative Velocity km per sec'] = rel_velocity
        if 'Relative Velocity km per hr' in input_dict:
            input_dict['Relative Velocity km per hr'] = rel_velocity * 3600
    if 'Miss Dist.(kilometers)' in input_dict:
        input_dict['Miss Dist.(kilometers)'] = miss_dist
        if 'Miss Dist.(miles)' in input_dict:
            input_dict['Miss Dist.(miles)'] = miss_dist * 0.621371
        if 'Miss Dist.(lunar)' in input_dict:
            input_dict['Miss Dist.(lunar)'] = miss_dist / 384400
    if 'Minimum Orbit Intersection' in input_dict:
        input_dict['Minimum Orbit Intersection'] = min_orbit_inter
    if 'Eccentricity' in input_dict:
        input_dict['Eccentricity'] = eccentricity

    input_df = pd.DataFrame([input_dict])

    with col2:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("### System Processing")
        st.markdown("<span style='color:#828a99; font-size: 0.9rem;'>Initialize Logistic Regression classification</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Initialize Threat Analysis"):
            with st.spinner("Processing telemetry data through classification model..."):
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                probability = model.predict_proba(input_scaled)[0][1]
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Analysis Report")
                
                if prediction[0] == 1 or probability > 0.15: 
                    st.error("""
                    **CLASSIFICATION: HAZARDOUS**  
                    The input parameters map to a potentially hazardous profile. Constant monitoring recommended.
                    """)
                    st.markdown(f"<h2 style='text-align: center; color: #FF416C;'>Threat Probability: {probability:.1%}</h2>", unsafe_allow_html=True)
                else:
                    st.success("""
                    **CLASSIFICATION: SAFE**  
                    The NEO trajectory and physical bounds fall within acceptable safety thresholds.
                    """)
                    st.markdown(f"<h2 style='text-align: center; color: #38ef7d;'>Threat Probability: {probability:.1%}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("### Project Architecture & Documentation")
    st.markdown("""
    This application serves as the deployment interface for a machine learning classification project trained on the **NASA Near-Earth Object (NEO) Dataset**. 
    
    #### Dataset & Preprocessing
    The original dataset contained over 4,600 records of NEO telemetry, including multiple redundant measurements (e.g., diameters in kilometers, meters, miles, and feet). 
    A robust preprocessing pipeline was implemented in Python using `pandas` and `scikit-learn` to isolate the core orbital mechanics and physical characteristics, dropping statistically redundant identification columns. The remaining feature space was standardized using a `StandardScaler`.
    
    #### The Multicollinearity Optimization
    The classification engine utilizes a **Logistic Regression** model. Because astronomy datasets often contain mathematically identical but inversely scaled features (e.g., maximum diameter and minimum diameter, or Miss Distance in kilometers vs miles), the linear model is highly sensitive to **Multicollinearity**. 
    
    If the UI allowed a user to increase the maximum diameter of an asteroid without scaling the minimum diameter proportionally, the linear equation would receive physically impossible inputs, causing the mathematical weights to cancel each other out and return a 0% probability.
    
    To ensure physical consistency and robust predictions, the UI architecture features an integrated **parameter synchronization layer**. When a user adjusts an intuitive feature (like Relative Velocity in km/s), the backend dynamically recalculates and synchronizes all mathematically dependent features before passing the data to the prediction engine.
    
    #### Key Variables Exposed
    For UX simplicity, the dashboard exposes the 6 most intuitive features, while dynamically synchronizing or utilizing median values for the remaining parameters.
    * **Absolute Magnitude (H):** Intrinsic brightness. Negatively correlated with physical size.
    * **Estimated Diameter:** The maximum bound of the NEO's physical size.
    * **Relative Velocity:** The kinetic speed of the NEO relative to Earth.
    * **Miss Distance:** The closest approach distance during the orbital epoch.
    * **Minimum Orbit Intersection (MOI):** The minimum distance between Earth's orbit and the NEO's orbit.
    * **Eccentricity:** The elongation of the NEO's orbital path around the sun.
    """)
