import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

data_path = 'heart.csv' 
df = pd.read_csv(data_path)

X = df.drop(columns=['target']) 
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.title("Heart Disease Risk Assessment")

def user_input_features():
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.selectbox("Chest Pain Type (4 values)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (values 0,1,2)", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox("Slope of the peak exercise ST segment", options=[0, 1, 2])
    ca = st.number_input("Number of major vessels (0-3) colored by flourosopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)", options=[1, 2, 3])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

df_input = pd.concat([input_df, df.drop(columns=['target'])], axis=0)


df_input = df_input[:1]

prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)

st.subheader("Prediction")
st.write("High Risk" if prediction[0] == 1 else "Low Risk")

st.subheader("Prediction Probability")
st.write(f"Low Risk: {prediction_proba[0][0]:.2f}")
st.write(f"High Risk: {prediction_proba[0][1]:.2f}")

st.subheader("Model Evaluation on Test Data (Developer Info)")
y_pred = model.predict(X_test)
st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File {file_name} not found. Please ensure the CSS file is in the correct path.")

load_css("styles.css")
