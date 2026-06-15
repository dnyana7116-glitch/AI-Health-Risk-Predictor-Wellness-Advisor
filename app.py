import streamlit as st
import pandas as pd
import pickle
from groq import Groq

st.set_page_config(
    page_title="AI Health Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Health Risk Predictor & Wellness Advisor")
st.write("Predict health risk using ML and get AI-powered wellness guidance.")

with open("health_risk_model.pkl", "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
encoders = saved_data["encoders"]
columns = saved_data["columns"]

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
st.subheader("Enter Your Health Details")

user_input = {}

for col in columns:
    if col == "bmi":
        continue

    if col in encoders:
        options = list(encoders[col].classes_)
        user_input[col] = st.selectbox(col, options)
    else:
        user_input[col] = st.number_input(col, value=0.0)

if "height" in columns and "weight" in columns and "bmi" in columns:
    height_m = float(user_input["height"]) / 100

    if height_m > 0:
        bmi = round(float(user_input["weight"]) / (height_m ** 2), 2)
    else:
        bmi = 0

    user_input["bmi"] = bmi
    st.info(f"Calculated BMI: {bmi}")

if st.button("Predict Health Risk"):

    display_df = pd.DataFrame([user_input])
    input_df = display_df.copy()
    input_df = input_df[columns]

    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)
    risk = encoders["health_risk"].inverse_transform(prediction)[0]

    st.subheader("Prediction Result")

    if risk.lower() == "low":
        st.success(f"Predicted Health Risk: {risk}")
    elif risk.lower() == "moderate":
        st.warning(f"Predicted Health Risk: {risk}")
    else:
        st.error(f"Predicted Health Risk: {risk}")

    score = 100

    if bmi < 18.5:
        score -= 15
    elif 18.5 <= bmi <= 24.9:
        score -= 0
    elif 25 <= bmi <= 29.9:
        score -= 10
    else:
        score -= 20

    if "sleep" in user_input:
        sleep = float(user_input["sleep"])
        if sleep < 5:
            score -= 20
        elif sleep < 7:
            score -= 10

    if "smoking" in user_input:
        smoking = str(user_input["smoking"]).strip().lower()
        if smoking in ["yes", "true", "1"]:
            score -= 20

    if "alcohol" in user_input:
        alcohol = str(user_input["alcohol"]).strip().lower()
        if alcohol in ["yes", "true", "1"]:
            score -= 10

    if "sugar_intake" in user_input:
        sugar = str(user_input["sugar_intake"]).strip().lower()
        if sugar == "high":
            score -= 20
        elif sugar == "medium":
            score -= 10

    if "exercise" in user_input:
        exercise = str(user_input["exercise"]).strip().lower()
        if exercise == "none":
            score -= 20
        elif exercise == "low":
            score -= 10
        elif exercise == "medium":
            score -= 5

    score = max(0, min(score, 100))

    st.write("## Wellness Score")
    st.progress(score / 100)
    st.metric("Health Score", f"{score}/100")

    if score >= 80:
        st.success("Excellent lifestyle habits!")
    elif score >= 60:
        st.warning("Average wellness level. Some improvements are recommended.")
    else:
        st.error("Poor wellness score. Lifestyle improvements are strongly recommended.")

    st.write("### Entered Details")
    st.dataframe(display_df)

    st.session_state["risk"] = risk
    st.session_state["score"] = score
    st.session_state["user_input"] = user_input

    st.write("## AI Wellness Advisor")

    prompt = f"""
    You are a helpful wellness assistant.
    Do not give medical diagnosis or prescribe medicine.

    User details:
    {user_input}

    Predicted health risk:
    {risk}

    Wellness score:
    {score}/100

    Give:
    1. Simple health summary
    2. Main lifestyle concerns
    3. Personalized improvement suggestions
    4. Diet suggestions
    5. Exercise suggestions
    6. One-week improvement plan

    Keep it practical and beginner-friendly.
    """

    try:
        response = client.chat.completions.create(
           model="llama-3.3-70b-versatile",
           messages=[
              {
                "role": "user",
                "content": prompt
              }
           ]
        )

        st.write(response.choices[0].message.content)

        st.write(response.text)

    except Exception:
        st.warning("AI advisor API limit reached. Showing offline wellness advice.")

        st.info(
            """
            Basic Wellness Guidance:
            - Maintain 7–8 hours of sleep.
            - Exercise at least 4–5 days per week.
            - Reduce sugar intake and processed food.
            - Drink enough water.
            - Avoid smoking and limit alcohol.
            - Track BMI and lifestyle habits regularly.
            """
        )

st.write("---")
st.write("## Ask Health Assistant")

if "risk" in st.session_state:

    question = st.text_input("Ask a health-related question")

    if st.button("Ask AI"):

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            chat_prompt = f"""
            You are a helpful wellness assistant.
            Do not give medical diagnosis or prescribe medicine.

            User health profile:
            {st.session_state["user_input"]}

            Predicted health risk:
            {st.session_state["risk"]}

            Wellness score:
            {st.session_state["score"]}/100

            User question:
            {question}

            Give practical, safe, simple wellness guidance.
            """

            try:
                chat_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                      {
                         "role": "user",
                         "content": chat_prompt
                     }
                   ]
                )

                st.write(chat_response.choices[0].message.content)
            except Exception:
                st.warning("AI chatbot API limit reached. Showing offline assistant response.")

                q = question.lower()

                if "yogurt" in q or "curd" in q:
                    st.info(
                        "Yes, yogurt/curd can be a healthy option for many people because it provides protein and probiotics. Prefer plain unsweetened yogurt. Avoid adding sugar. If you are lactose intolerant or have a specific medical condition, consult a doctor."
                    )

                elif "sugar" in q:
                    st.info(
                        "Try reducing sugary drinks, packaged snacks, desserts, and sweet tea/coffee. Choose fruits, nuts, or plain yogurt instead."
                    )

                elif "sleep" in q:
                    st.info(
                        "Aim for 7–8 hours of sleep. Keep a fixed sleep time, avoid screens before bed, and reduce caffeine late in the day."
                    )

                elif "exercise" in q:
                    st.info(
                        "Start with 20–30 minutes of walking daily. Slowly add yoga, cycling, or light strength exercises."
                    )

                elif "bmi" in q:
                    st.info(
                        "BMI is calculated from height and weight. A very high or very low BMI can indicate lifestyle or nutrition risk."
                    )

                else:
                    st.info(
                        "Basic wellness advice: maintain balanced diet, regular exercise, proper sleep, controlled sugar intake, and avoid smoking/alcohol."
                    )

else:
    st.info("First predict your health risk, then ask the assistant a question.")
