import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('lasso_model.joblib')
scaler = joblib.load('scaler.joblib')

# Page configuration
st.set_page_config(page_title="Body Fat Predictor", layout="centered")

# Title
st.title("🏋️‍♀️ Body Fat % Estimator")
st.write("Estimate your body fat percentage using body measurements. The prediction is based on Siri's 1956 equation and a machine learning model.")

# Gender for interpreting the result
gender = st.selectbox("Select your gender:", ["Male", "Female"])

# Input section
st.header("📏 Enter Your Measurements")

col1, col2 = st.columns(2)

with col1:
    density = st.number_input("Density", value=1.0853, format="%.4f")
    age = st.number_input("Age", value=22.0)
    weight = st.number_input("Weight (kg)", value=78.58)
    height = st.number_input("Height (inches)", value=72.25)
    neck = st.number_input("Neck circumference (cm)", value=38.5)
    chest = st.number_input("Chest circumference (cm)", value=93.6)
    abdomen = st.number_input("Abdomen circumference (cm)", value=83.0)

with col2:
    hip = st.number_input("Hip circumference (cm)", value=98.7)
    thigh = st.number_input("Thigh circumference (cm)", value=58.7)
    knee = st.number_input("Knee circumference (cm)", value=37.3)
    ankle = st.number_input("Ankle circumference (cm)", value=23.4)
    biceps = st.number_input("Biceps circumference (cm)", value=30.5)
    forearm = st.number_input("Forearm circumference (cm)", value=28.9)
    wrist = st.number_input("Wrist circumference (cm)", value=18.2)

# Create input DataFrame
input_df = pd.DataFrame([{
    'Density': density,
    'Age': age,
    'Weight': weight,
    'Height': height,
    'Neck': neck,
    'Chest': chest,
    'Abdomen': abdomen,
    'Hip': hip,
    'Thigh': thigh,
    'Knee': knee,
    'Ankle': ankle,
    'Biceps': biceps,
    'Forearm': forearm,
    'Wrist': wrist
}])

# Predict button
if st.button("🔍 Predict Body Fat %"):
    # Predict body fat
    scaled_input = scaler.transform(input_df)
    predicted_bf = model.predict(scaled_input)[0]

    # Determine healthy zone
    safe = False
    if gender == "Male":
        safe = 6 <= predicted_bf <= 24
        healthy_range = "6% – 24%"
    else:
        safe = 14 <= predicted_bf <= 31
        healthy_range = "14% – 31%"

    # Display results
    st.header("📊 Prediction Result")
    st.metric("Predicted Body Fat %", f"{predicted_bf:.2f} %")

    if safe:
        st.success(f"✅ You are in a **healthy range** for {gender}s ({healthy_range})")
    else:
        st.error(f"⚠️ Your predicted body fat is **outside the healthy range** for {gender}s ({healthy_range})")

# Add explanation
with st.expander("ℹ️ About This Prediction"):
    st.markdown(r"""
This prediction is made using a **machine learning model** trained on body measurement data.  
""")

# Optional healthy body fat chart
with st.expander("📚 Healthy Body Fat % Chart (ACE Guidelines)"):
    st.markdown("""
| Category       | Men (%) | Women (%) |
|----------------|---------|-----------|
| Essential Fat  | 2–5     | 10–13     |
| Athletes       | 6–13    | 14–20     |
| Fitness        | 14–17   | 21–24     |
| Acceptable     | 18–24   | 25–31     |
| Obese          | 25+     | 32+       |
    """)
