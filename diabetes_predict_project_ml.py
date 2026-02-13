import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

st.title("Diabetes Prediction ML Project")

# -------------------------------
# Load Dataset (IMPORTANT FIX)
# -------------------------------
df = pd.read_excel("Diabetes_Dataset_Project.xlsx")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Clean Column Names
# -------------------------------
df.columns = df.columns.str.strip()

short_names = {
    'age': 'Age',
    'Height_cm': 'Height',
    'Weight_kg': 'Weight',
    'bmi': 'BMI',
    'Gender': 'Gender',
    'How would you describe your daily diet?': 'Diet',
    'family_history': 'FamilyHist',
    'How often do you consume sugary foods or drinks?': 'SugarFreq',
    'How many days per week do you exercise or do physical activity?': 'ExerciseDays',
    'Do you experience frequent thirst or dry mouth?': 'Thirst',
    'Do you feel unusually tired most of the time?': 'Tired',
    'Do you urinate more frequently than usual?': 'Urination',
    'Do you have high blood pressure?': 'HighBP',
    'diabetes': 'Diabetes'
}

df.rename(columns=short_names, inplace=True)

# -------------------------------
# Encode Categorical
# -------------------------------
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# -------------------------------
# Split Data
# -------------------------------
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Train Models
# -------------------------------
lr = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=42)

models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "Decision Tree": dt
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

st.subheader("Model Results")

for name, metrics in results.items():
    st.write(f"{name} → Accuracy: {metrics['Accuracy']:.4f} | F1 Score: {metrics['F1 Score']:.4f}")

# -------------------------------
# Best Model
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]["F1 Score"])
st.success(f"Best Model: {best_model_name}")

best_model = models[best_model_name]

# -------------------------------
# Plot 1: Diabetes Distribution
# -------------------------------
st.subheader("Diabetes Distribution")

fig1, ax1 = plt.subplots()
df['Diabetes'].value_counts().plot(kind='bar', ax=ax1)
st.pyplot(fig1)

# -------------------------------
# Plot 2: Confusion Matrix
# -------------------------------
st.subheader("Confusion Matrix")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

fig2, ax2 = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax2)
st.pyplot(fig2)


st.subheader("🩺 Diabetes Risk Predictor")

st.write("Enter basic details to check your diabetes risk")

# 1️⃣ Age
age = st.number_input("Enter Age", min_value=10, max_value=100, value=25)

# 2️⃣ Height (in meters)
height = st.number_input("Enter Height (in meters)", min_value=1.0, max_value=2.5, value=1.65)

# 3️⃣ Weight
weight = st.number_input("Enter Weight (in kg)", min_value=30.0, max_value=200.0, value=60.0)

# 4️⃣ Gender Dropdown
gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])

gender_mapping = {
    "Male": 1,
    "Female": 0,
    "Other": 2
}
gender_value = gender_mapping[gender]

# 5️⃣ Exercise Days
exercise = st.number_input("Enter Exercise Days per Week", min_value=0, max_value=7, value=3)

# Automatically Calculate BMI
bmi = weight / (height ** 2)

# Default values for remaining features
diet = 1
familyhist = 0
sugarfreq = 1
thirst = 0
tired = 0
urination = 0
highbp = 0

if st.button("Predict Diabetes Risk"):

    input_data = np.array([[ 
        age,
        height,
        weight,
        bmi,
        gender_value,
        diet,
        familyhist,
        sugarfreq,
        exercise,
        thirst,
        tired,
        urination,
        highbp
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = best_model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.info(f"Calculated BMI: {round(bmi, 2)}")

