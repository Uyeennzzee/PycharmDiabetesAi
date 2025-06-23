import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="AI Diabetes Predictor", layout="centered")
st.title("🧠 AI Diabetes Prediction Tool")
st.markdown("This tool predicts your diabetes risk using both **medical** and **lifestyle** data.")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# --- Clinical Inputs ---
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

# --- Lifestyle Inputs ---
st.header("🌿 Lifestyle Habits")

smoke = st.radio("Do you currently smoke?", ["No", "Yes"])
drink = st.radio("Do you consume alcohol heavily?", ["No", "Yes"])
exercise = st.radio("Have you done any physical activity in the past 30 days (not counting work)?", ["No", "Yes"])

smoke_val = 1 if smoke == "Yes" else 0
drink_val = 1 if drink == "Yes" else 0
exercise_val = 1 if exercise == "Yes" else 0

# --- Prediction ---
if st.button("Predict"):
    features = np.array([[preg, glucose, bp, insulin, bmi, age, family, smoke_val, drink_val, exercise_val]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ You are likely to have diabetes.")

        # --- Clinical Rule-based Feedback ---
        if insulin < 2 and bmi < 25:
            if age < 40:
                st.info("🧠 This could suggest **Type 1 Diabetes**. Please consult a doctor for confirmation.")
            else:
                st.info("🧠 Possibly Type 1. Unusual for age — please consult a doctor.")
        elif insulin > 25 and bmi >= 28:
            st.info("🧠 Likely **Type 2 Diabetes**. Please consult a doctor.")
        else:
            st.info("🧠 Diabetes type unclear — consult a doctor for lab testing.")

        if glucose >= 126:
            st.warning("⚠️ Your glucose level is very high. Please seek medical evaluation.")
        if bmi >= 25:
            st.warning("⚠️ Your BMI indicates overweight or obesity — a major diabetes risk.")
        if bp >= 90:
            st.warning("⚠️ Your blood pressure is above normal. This contributes to diabetes complications.")

        # --- Lifestyle Rule-based Feedback ---
        if smoke_val == 1:
            st.warning("🚬 You reported that you smoke. Smoking increases insulin resistance and diabetes risk.")
        else:
            st.success("✅ Not smoking is a strong protective factor against diabetes.")

        if drink_val == 1:
            st.warning("🍷 Heavy alcohol consumption can impair blood sugar control. Consider moderating alcohol.")
        else:
            st.success("✅ Not drinking heavily helps maintain stable glucose levels.")

        if exercise_val == 0:
            st.warning("🏃‍♂️ Lack of physical activity raises your diabetes risk. Try to stay active weekly.")
        else:
            st.success("✅ Great! Physical activity lowers blood sugar and improves insulin sensitivity.")

    else:
        st.success("✅ You are not likely to have diabetes.")

        # --- Encouragement for healthy ranges ---
        if glucose >= 126:
            st.info("ℹ️ Your glucose level is borderline high. Even if not diabetic, keep monitoring it.")
        if bmi >= 25:
            st.info("ℹ️ Your BMI is above the healthy range. Consider exercise and diet improvement.")
        if bp >= 90:
            st.info("ℹ️ Your blood pressure is slightly high. Try to reduce salt, stress, or consult a doctor.")

        # --- Lifestyle Feedback for Healthy Users ---
        if smoke_val == 1:
            st.warning("🚬 Smoking still harms your overall health and increases diabetes risk long-term.")
        else:
            st.success("✅ Not smoking is a major health benefit.")

        if drink_val == 1:
            st.warning("🍷 Consider reducing alcohol. Even without diabetes, it can cause inflammation and sugar spikes.")
        else:
            st.success("✅ Great job avoiding heavy alcohol use!")

        if exercise_val == 0:
            st.warning("🏃‍♂️ Try to get at least 150 minutes of moderate exercise weekly.")
        else:
            st.success("✅ Keep up the good work staying active!")
