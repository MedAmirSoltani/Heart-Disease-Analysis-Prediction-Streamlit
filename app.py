# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Set page configuration with custom color and icon
st.set_page_config(page_title="Heart Disease Data Analysis Dashboard", layout="wide", page_icon="❤️")

# Custom CSS styling for enhanced appearance
st.markdown("""
    <style>
    .main {background-color: #f9f9f9; color: #333333; font-family: Arial;}
    .header-text {font-size: 2.5em; color: #c0392b;}
    .insight-text {color: #8e44ad; font-size: 1.1em;}
    .summary-text {font-size: 1.2em; color: #2980b9;}
    </style>
""", unsafe_allow_html=True)

# Display Header with Icon
st.markdown('<h1 class="header-text">Heart Disease Data Analysis Dashboard ❤️</h1>', unsafe_allow_html=True)
st.markdown("""
Welcome to the Heart Disease Data Analysis Dashboard. This tool allows you to explore how various health, lifestyle, and demographic factors relate to heart disease. 
Use the interactive features to dynamically change plot types, customize settings, and gain deeper insights!
""")

# Load and preprocess dataset with caching
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('heart.csv')

    # BMI Categorization
    bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
    bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']
    data['BMICategory'] = pd.cut(data['BMI'], bins=bmi_bins, labels=bmi_labels)
    data.drop(columns=['BMI'], inplace=True)  # Drop original BMI column
    # Age Group Categorization
    age_mapping = {
        '18-24': 'Young', '25-29': 'Young', '30-34': 'Adult', '35-39': 'Adult',
        '40-44': 'Adult', '45-49': 'Adult', '50-54': 'Adult', '55-59': 'Senior',
        '60-64': 'Senior', '65-69': 'Senior', '70-74': 'Senior', '75-79': 'Senior', '80 or older': 'Senior'
    }
    data['AgeGroup'] = data['AgeCategory'].map(age_mapping)
    data.drop(columns=['AgeCategory'], inplace=True)  # Drop original AgeCategory column
    mental_bins = [0.0, 1.0, 15.0, float('inf')]
    mental_labels = ['Healthy', 'Occasional', 'Frequent']
    data['MentalHealthCategory'] = pd.cut(data['MentalHealth'], bins=mental_bins, labels=mental_labels)
    data.dropna(subset=['MentalHealthCategory'], inplace=True)  # Drop rows with NaN in MentalHealthCategory

    data['MentalHealthCategory'] = data['MentalHealthCategory'].astype(str)  # Convert to string to avoid NaN issues
    data.drop(columns=['MentalHealth'], inplace=True)  # Drop original BMI column


    # Keep categorical values for plotting
    data['HeartDisease'] = data['HeartDisease'].map({'Yes': 'Yes', 'No': 'No'})
    data['Smoking'] = data['Smoking'].map({'Yes': 'Yes', 'No': 'No'})
    data['AlcoholDrinking'] = data['AlcoholDrinking'].map({'Yes': 'Yes', 'No': 'No'})
    data['Stroke'] = data['Stroke'].map({'Yes': 'Yes', 'No': 'No'})
    data['DiffWalking'] = data['DiffWalking'].map({'Yes': 'Yes', 'No': 'No'})
    data['PhysicalActivity'] = data['PhysicalActivity'].map({'Yes': 'Yes', 'No': 'No'})
    data['Asthma'] = data['Asthma'].map({'Yes': 'Yes', 'No': 'No'})
    data['KidneyDisease'] = data['KidneyDisease'].map({'Yes': 'Yes', 'No': 'No'})
    data['SkinCancer'] = data['SkinCancer'].map({'Yes': 'Yes', 'No': 'No'})
    data['GenHealth'] = data['GenHealth'].map({'Excellent': 'Excellent', 'Very good': 'Very good', 'Good': 'Good', 'Fair': 'Fair', 'Poor': 'Poor'})
    data['Sex'] = data['Sex'].map({'Male': 'Male', 'Female': 'Female'})

    return data

data = load_and_preprocess_data()




import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
rf_model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")  # Ensure you saved the scaler with joblib








# Sidebar for feature selection and plot customization
st.sidebar.header("Customize Analysis")
selected_feature = st.sidebar.selectbox(
    "Select Feature to Analyze with Heart Disease",
    options=[col for col in data.columns if col != 'HeartDisease']
)

plot_type = st.sidebar.radio(
    "Choose Plot Type",
    ["Histogram","Box Plot",  "Scatter Plot", "Violin Plot", "Density Heatmap"]
)

# Add an option to swap x and y axes
axis_swap = st.sidebar.checkbox("Swap X and Y axes", value=False)

st.sidebar.header("Plot Customization")
color_option = st.sidebar.selectbox("Color by", options=["None"] + list(data.columns))
barmode_option = st.sidebar.radio("Bar Mode (if applicable)", options=["overlay", "group"], index=0)

# Display selected plot dynamically based on the user’s input
st.header(f"Analysis of Heart Disease with respect to {selected_feature}")
if plot_type == "Box Plot":
    if axis_swap:
        fig = px.box(data, x="HeartDisease", y=selected_feature, color=color_option if color_option != "None" else None,
                     title=f"Heart Disease by {selected_feature}")
    else:
        fig = px.box(data, x=selected_feature, y="HeartDisease", color=color_option if color_option != "None" else None,
                     title=f"Heart Disease by {selected_feature}")
elif plot_type == "Histogram":
    fig = px.histogram(data, x="HeartDisease", color=selected_feature, barmode=barmode_option,
                       title=f"Heart Disease Distribution with {selected_feature}",histnorm='percent')
elif plot_type == "Scatter Plot":
    if axis_swap:
        fig = px.scatter(data, x="HeartDisease", y=selected_feature, color=color_option if color_option != "None" else None,
                         title=f"Scatter Plot of {selected_feature} and Heart Disease")
    else:
        fig = px.scatter(data, x=selected_feature, y="HeartDisease", color=color_option if color_option != "None" else None,
                         title=f"Scatter Plot of {selected_feature} and Heart Disease")
elif plot_type == "Violin Plot":
    if axis_swap:
        fig = px.violin(data, x="HeartDisease", y=selected_feature, color=color_option if color_option != "None" else None,
                        title=f"Violin Plot of {selected_feature} and Heart Disease")
    else:
        fig = px.violin(data, x=selected_feature, y="HeartDisease", color=color_option if color_option != "None" else None,
                        title=f"Violin Plot of {selected_feature} and Heart Disease")
elif plot_type == "Density Heatmap":
    fig = px.density_heatmap(data, x=selected_feature, y="HeartDisease", color_continuous_scale='Viridis',
                             title=f"Density Heatmap of {selected_feature} and Heart Disease")

st.plotly_chart(fig, use_container_width=True)
st.markdown(f'<p class="insight-text">Insight: Observe how {selected_feature} influences heart disease prevalence. Experiment with different plot types for a deeper view.</p>', unsafe_allow_html=True)

# Add correlation analysis section
st.header("Correlation Analysis")
@st.cache_data
def compute_correlation_matrix():
    data = pd.read_csv('heart.csv')

    # Create a copy for correlation analysis to avoid modifying the original data
    correlation_data = data.copy()

    # Apply transformations for correlation analysis only
    binary_columns = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
                      'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic']
    for col in binary_columns:
        correlation_data[col] = correlation_data[col].apply(lambda x: 1 if x == 'Yes' else 0)

    correlation_data['Sex'] = correlation_data['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
    correlation_data['GenHealth'] = correlation_data['GenHealth'].map({'Excellent': 5, 'Very good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1})
    
    corr_features = ['HeartDisease', 'PhysicalHealth', 'BMI', 'SleepTime', 'Smoking', 'AlcoholDrinking', 
                     'Stroke', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
    
    return correlation_data[corr_features].corr()

correlation_matrix = compute_correlation_matrix()
fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='Plasma'
))
fig_corr.update_layout(title="Correlation Heatmap of Health and Lifestyle Factors")
st.plotly_chart(fig_corr, use_container_width=True)
# Summary section with customized styling and icons
st.header("Summary of Insights")
st.markdown('<p class="summary-text">Key Insights and Observations:</p>', unsafe_allow_html=True)
st.markdown("""
- **Demographics**: Heart disease prevalence shows clear distinctions across age groups and BMI categories, highlighting that both age and weight can be influential risk factors.
- **Lifestyle Choices**: Behaviors such as physical inactivity, smoking, and alcohol consumption demonstrate strong associations with heart disease, emphasizing the role of lifestyle choices in cardiovascular health.
- **Chronic Health Conditions**: Individuals with chronic conditions, particularly stroke and kidney disease, tend to have a significantly higher risk of heart disease, suggesting an interplay between various health conditions.
- **Self-Reported Health**: General health perceptions, as reported by individuals, appear to be strongly linked with heart disease risk, indicating that subjective health assessments may reflect underlying risks.
""")








# Define feature names (must match those used when fitting the scaler and model)
feature_names = [
    'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'DiffWalking', 'Sex', 
    'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 
    'SkinCancer', 'BMI_Category', 'NewAge'
]

# Input form for prediction
st.sidebar.header("Heart Disease Prediction")
with st.sidebar.form("prediction_form"):
    Smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    AlcoholDrinking = st.selectbox("Alcohol Drinking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Stroke = st.selectbox("Stroke", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    PhysicalHealth = st.slider("Physical Health (Poor days)", 0, 30, 15)
    DiffWalking = st.selectbox("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    Diabetic = st.selectbox("Diabetic", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    PhysicalActivity = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    GenHealth = st.selectbox("General Health", [0, 1, 2, 3, 4], format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x])
    SleepTime = st.slider("Sleep Time (Hours)", 0, 24, 7)
    Asthma = st.selectbox("Asthma", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    KidneyDisease = st.selectbox("Kidney Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    SkinCancer = st.selectbox("Skin Cancer", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    BMI_Category = st.selectbox("BMI Category", range(6), format_func=lambda x: ["Underweight", "Normal weight", "Overweight", "Obesity I", "Obesity II", "Obesity III"][x])
    NewAge = st.selectbox("Age Group", range(3), format_func=lambda x: ["Young", "Adult", "Old"][x])
    
    # Submit button for validation
    submit_button = st.form_submit_button("Validate")

# Run the prediction if the form is submitted
if submit_button:
    # Collect input values into a DataFrame for scaling
    input_data = np.array([[Smoking, AlcoholDrinking, Stroke, PhysicalHealth, DiffWalking, Sex, Diabetic,
                            PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer, BMI_Category, NewAge]])
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = rf_model.predict(scaled_input)
    prediction_proba = rf_model.predict_proba(scaled_input)
    probability = f"{prediction_proba[0][prediction[0]]:.2f}"

# Display result with styling and icons
if prediction[0] == 1:
    result = "High Risk ⚠️"
    color = "#D9534F"  # Red color for high risk
else:
    result = "Low Risk ✅"
    color = "#5CB85C"  # Green color for low risk

# Format the probability and result text with HTML, adding center alignment
probability = f"{float(probability):.2%}"
probability_text = f"<p style='font-size: 1.2em; color: {color}; text-align: center;'><strong>Probability: {probability}</strong></p>"
result_text = f"<h3 style='color: {color}; text-align: center;'>{result}</h3>"

# Display the centered and formatted result in the sidebar
st.sidebar.markdown("<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True)
st.sidebar.markdown(result_text, unsafe_allow_html=True)
st.sidebar.markdown(probability_text, unsafe_allow_html=True)







# Footer with a call-to-action and additional icons
st.markdown("---")
st.markdown("**Thank you for exploring the Heart Disease Data Analysis Dashboard!** Adjust the settings to uncover more insights and deepen your understanding.")
