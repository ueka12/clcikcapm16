import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Streamlit with ML",
    page_icon="🕹️",
    layout=None)

st.header("ระบบทำนายขยะอัญชะลิยะ")

df = pd.read_csv("sustainable_waste_management_dataset_2024.csv")

col1, col2 = st.columns(2)
with col1:
    population = st.number_input("ประชากรเมืง", 100, 50000000, 100000, 1000)
    selection = st.pills("", ["ล้น", "วันหยุดสุดสัปดาห์", "วันหุดเทสกาน", "รนนรงลดขยะ"], selection_mode="multi")
    temp = st.slider("อุนหพูม", -100, 50, 1, 30)
    rain = st.slider("ฝนเท่าไร", 0, 50, 1, 0)



# เลือก feature แล้วทำการทำความสะอาดข้อมูล
selected_features = ["population", "overflow", "is_weekend", "is_holiday", "recycling_campaign", "temp_c", "rain_mm"]
X = df[selected_features]
y = df['waste_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['waste_kg']

# แบ่ง dataframe เป็น 2 ส่วนหลัก ๆ โดยมีส่วนที่ใช้สำหรับฝึกโมเดลและทดสอบโมเดล
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R squared: ", r2_score(Y_test, Y_pred))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Y_test, Y_pred, alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
ax.set_xlabel('Actual Waste (Y_test)')
ax.set_ylabel('Predicted Waste (Y_pred)')
ax.set_title('Predicted vs. Actual Waste Created')
ax.legend()
ax.grid(True)

st.pyplot(fig)

# 1. Initialize the toggle state in session_state
if 'show_raw' not in st.session_state:
    st.session_state.show_raw = False

# 2. Define the labels based on the state
label = "เลิกแสดงข้อมูลดิบ ❌" if st.session_state.show_raw else "แสดงข้อมูลดิบ 🥩"

# 3. Create the toggle button
if st.button(label):
    # This flips the state (True becomes False, False becomes True)
    st.session_state.show_raw = not st.session_state.show_raw
    st.rerun() # Refresh immediately to update the button label

# 4. Display the data if the state is True
if st.session_state.show_raw:
    # Assuming 'df' is your dataframe
    st.dataframe(df)