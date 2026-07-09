# AsteroidClassifier

**Live Web App:** https://asteroidclassifier-qjnhy37sgzcmrctrvlwsed.streamlit.app/

##  Overview
This project is a machine learning classification system designed to predict whether a Near Earth Object is hazardous or safe based on orbital statistics and physical characteristics. 

## Machine Learning Pipeline
All data exploration, visualization, and model training are documented in the **`HazardousAsteroid.ipynb`** notebook. 
Key steps include:
- **Exploratory Data Analysis :** Visualized feature distributions and generated correlation heatmaps using Seaborn.
- **Feature Engineering:** Identified and resolved high multicollinearity by isolating mathematically redundant telemetry metrics (e.g., duplicate diameter measurements).
- **Preprocessing:** Applied `StandardScaler` to normalize orbital mechanics data.
- **Model Training:** Trained a Logistic Regression classification model.
- **Threshold Optimization:** Implemented custom probability thresholds to maximize Recall  for a reliable hazard detection system.

## Files in this Repository
- **`HazardousAsteroid.ipynb`**: The main notebook containing all ML and EDA work.
- **`app.py`**: The frontend Streamlit application.
- **`asteroid_model.pkl`**: The trained Logistic Regression model.
- **`asteroid_scaler.pkl`**: The standard scaler used for data normalization.
- **`nasa.csv`**: The NASA NEO dataset.
- **`requirements.txt`**: Dependency file for Streamlit Cloud deployment.
