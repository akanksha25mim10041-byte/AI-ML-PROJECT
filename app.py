
import streamlit as st

st.set_page_config(
    page_title="Academic Score Forecaster",
    page_icon="🎓"
)

st.title("🎓 Academic Score Forecaster")

st.write("Welcome to the Student Exam Score Prediction Portal!")

st.header("Enter Student Information")

study_hours = st.slider("Study Hours Per Day", 0, 12, 6)

attendance = st.slider("Attendance Percentage", 0, 100, 80)

previous_score = st.slider("Previous Exam Score", 0, 100, 75)

sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

if st.button("Predict Score"):
    st.success("Your input has been received!")
    st.info("ML model integration will be added next.")