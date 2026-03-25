import streamlit as st
import numpy as np
import pickle

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")
st.markdown("### Predict your marks based on your daily habits")

st.write("---")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📚 Study Hours", 0, 12)
    attendance = st.slider("🏫 Attendance (%)", 0, 100)

with col2:
    sleep = st.slider("😴 Sleep Hours", 0, 10)
    previous = st.slider("📊 Previous Score", 0, 100)

st.write("---")

if st.button("🚀 Predict Performance"):
    input_data = np.array([[study_hours, attendance, sleep, previous]])
    result = model.predict(input_data)

    st.success(f"🎯 Predicted Marks: {result[0]:.2f}")

    if result[0] > 80:
        st.balloons()
        st.write("🔥 Excellent performance expected!")
    elif result[0] > 60:
        st.write("👍 Good job, keep improving!")
    else:
        st.warning("⚠️ You need to focus more!")

st.write("### 📊 Model Insight")
st.write("This prediction is based on patterns learned from student data.")


#adding graphs    
import matplotlib.pyplot as plt
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

data = load_data()

st.write("### 📈 Study Hours vs Marks")
fig, ax = plt.subplots()
ax.scatter(data['study_hours'], data['final_score'])
ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Score")

st.pyplot(fig)
