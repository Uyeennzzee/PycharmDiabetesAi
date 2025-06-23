import streamlit as st
import pickle
import numpy as np

st.title("🔍 AI Diabetes Prediction Tool")
st.markdown("This tool predicts your risk of diabetes based on your medical data. Please enter accurate values.")

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.header("🧾 Enter Your Medical Information")

preg = st.number_input("Pregnancies", min_value=0)
st.caption("Number of times you’ve been pregnant.")

glucose = st.number_input("Glucose Level (mg/dL)", min_value=0)
st.caption("Normal fasting level is 70–99 mg/dL. Above 126 mg/dL may indicate diabetes.")

bp = st.number_input("Blood Pressure (Diastolic, mm Hg)", min_value=0)
st.caption("Diastolic BP (bottom number). Normal is under 80 mm Hg.")

insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
st.caption("Normal is 2–25. Low (<2) suggests Type 1 risk. High (>25) suggests resistance.")

bmi = st.number_input("BMI", min_value=0.0)
st.caption("""
**BMI Categories:**
- Underweight: < 18.5  
- Normal weight: 18.5 – 24.9  
- Overweight: 25 – 29.9  
- Obese (Class I): 30 – 34.9  
- Obese (Class II): 35 – 39.9  
- Obese (Class III): ≥ 40
""")

family_history = st.radio("Do you have a family history of diabetes?", ["Yes", "No"])
family = 1 if family_history == "Yes" else 0

age = st.number_input("Age", min_value=0)
st.caption("Your current age.")

if st.button("Predict"):
    features = np.array([[preg, glucose, bp, insulin, bmi, family, age]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ You are likely to have diabetes.")

        # Rule-based suggestion
        if insulin < 2 and bmi < 25:
            if age < 40:
                st.info("🧠 This could suggest **Type 1 Diabetes**. Please consult a doctor for confirmation.")
            else:
                st.info("🧠 Possibly Type 1, but age is unusual. Please consult a doctor.")
        elif insulin > 25 and bmi >= 28:
            st.info("🧠 Likely **Type 2 Diabetes**. Please consult a doctor.")
        else:
            st.info("🧠 Type unclear. Please consult a doctor.")

    else:
        st.success("✅ You are not likely to have diabetes.")

        # Extra feedback for borderline/high values
        if glucose >= 126:
            st.info("ℹ️ Your glucose level is above normal. Consider monitoring it regularly.")
        if bmi >= 25:
            st.info("ℹ️ Your BMI suggests you're overweight. This may increase your future diabetes risk.")
        if bp >= 90:
            st.info("ℹ️ Your blood pressure is slightly high. Keep an eye on it.")
