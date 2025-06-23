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
st.caption("Normal is under 80 mm Hg.")

insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
st.caption("Normal is 2–25. Low (<2) suggests Type 1 risk. High (>25) suggests resistance.")

bmi = st.number_input("BMI", min_value=0.0)
st.caption("""
**BMI Categories:**
- Underweight: < 18.5  
- Normal: 18.5–24.9  
- Overweight: 25–29.9  
- Obese: ≥ 30
""")

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
    features = np.array([[preg, glucose, bp, insulin, bmi, age, smoke_val, drink_val, exercise_val]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ You are likely to have diabetes.")

        # --- Clinical Rule-based Feedback ---
        if insulin < 2 and bmi < 25:
            if age < 40:
                st.info("🧠 This could suggest **Type 1 Diabetes**. Please consult a doctor.")
            else:
                st.info("🧠 Possibly Type 1 — but age is unusual. Please consult a doctor.")
        elif insulin > 25 and bmi >= 28:
            st.info("🧠 Likely **Type 2 Diabetes**. Please consult a doctor.")
        else:
            st.info("🧠 Type unclear — consult a doctor for lab testing.")

        if glucose >= 126:
            st.warning("⚠️ Your glucose level is very high. Please seek medical evaluation.")
        if bmi >= 25:
            st.warning("⚠️ Your BMI is high. This increases diabetes and heart risk.")
        if bp >= 90:
            st.warning("⚠️ Your blood pressure is above normal.")

        # --- Lifestyle Rule-based Feedback ---
        if smoke_val == 1:
            st.warning("🚬 Smoking raises your insulin resistance. Try to quit.")
        else:
            st.success("✅ Great! Not smoking protects your health.")

        if drink_val == 1:
            st.warning("🍷 Alcohol can cause sugar spikes and worsen diabetes. Cut down if possible.")
        else:
            st.success("✅ Excellent! Avoiding heavy alcohol supports glucose control.")

        if exercise_val == 0:
            st.warning("🏃‍♂️ Inactivity increases diabetes risk. Try to stay active weekly.")
        else:
            st.success("✅ Exercise helps regulate insulin and improve glucose metabolism.")

    else:
        st.success("✅ You are not likely to have diabetes.")

        # Extra feedback for healthy users
        if glucose >= 126:
            st.info("ℹ️ Your glucose level is high. Monitor regularly.")
        if bmi >= 25:
            st.info("ℹ️ Consider weight loss strategies for long-term health.")
        if bp >= 90:
            st.info("ℹ️ High blood pressure can be a warning sign. Watch it.")

        # Lifestyle encouragement
        if smoke_val == 1:
            st.warning("🚬 Try to reduce or quit smoking.")
        else:
            st.success("✅ Not smoking is great for your health.")

        if drink_val == 1:
            st.warning("🍷 Consider moderating alcohol.")
        else:
            st.success("✅ Great! No alcohol is better for your body.")

        if exercise_val == 0:
            st.warning("🏃‍♂️ Aim for at least 150 minutes of weekly activity.")
        else:
            st.success("✅ Excellent! Keep staying active.")
