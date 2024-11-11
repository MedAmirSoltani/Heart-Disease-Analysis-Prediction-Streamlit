# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Set page configuration with custom color and icon
st.set_page_config(page_title="Mental Health Data Analysis Dashboard", layout="wide", page_icon="💆")

# Custom CSS styling for enhanced appearance
st.markdown("""
    <style>
    .main {background-color: #f0f2f6; color: #333333; font-family: Arial;}
    .header-text {font-size: 2.5em; color: #4b6584;}
    .insight-text {color: #2d98da; font-size: 1.1em;}
    .summary-text {font-size: 1.2em; color: #20bf6b;}
    </style>
""", unsafe_allow_html=True)

# Display Header with Icon
st.markdown('<h1 class="header-text">Mental Health Data Analysis Dashboard 💆</h1>', unsafe_allow_html=True)
st.markdown("""
Welcome to the Mental Health Data Analysis Dashboard. This tool allows you to explore how various health, lifestyle, and demographic factors relate to mental health. 
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

    # Age Group Categorization
    age_mapping = {
        '18-24': 'Young', '25-29': 'Young', '30-34': 'Adult', '35-39': 'Adult',
        '40-44': 'Adult', '45-49': 'Adult', '50-54': 'Adult', '55-59': 'Senior',
        '60-64': 'Senior', '65-69': 'Senior', '70-74': 'Senior', '75-79': 'Senior', '80 or older': 'Senior'
    }
    data['AgeGroup'] = data['AgeCategory'].map(age_mapping)

    # Encode categorical columns for correlation
    binary_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
    for col in binary_columns:
        data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)

    data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
    data['Diabetic'] = data['Diabetic'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['GenHealth'] = data['GenHealth'].map({'Excellent': 5, 'Very good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1})

    # Mental Health Categorization
    bins = [0, 2, 4, 6, 10]  # Healthy (0-2), Struggling (2-4), Coping (4-6), Very Good (6+)
    labels = ['Very Bad', 'Bad', 'Good', 'Very Good']
    data['MentalHealthCategory'] = pd.cut(data['MentalHealth'], bins=bins, labels=labels)
    
    # Convert 'MentalHealthCategory' to numerical values for correlation
    category_mapping = {'Very Bad': 1, 'Bad': 2, 'Good': 3, 'Very Good': 4}
    data['MentalHealthCategoryNum'] = data['MentalHealthCategory'].map(category_mapping)

    return data

data = load_and_preprocess_data()

# Sidebar for feature selection and plot customization
st.sidebar.header("Customize Analysis")
selected_feature = st.sidebar.selectbox(
    "Select Feature to Analyze with Mental Health",
    options=[col for col in data.columns if col != 'MentalHealth' and col != 'MentalHealthCategory' and col != 'MentalHealthCategoryNum']
)

plot_type = st.sidebar.radio(
    "Choose Plot Type",
    ["Box Plot", "Histogram", "Scatter Plot", "Violin Plot", "Density Heatmap"]
)

# Add an option to swap x and y axes
axis_swap = st.sidebar.checkbox("Swap X and Y axes", value=False)

st.sidebar.header("Plot Customization")
color_option = st.sidebar.selectbox("Color by", options=["None"] + list(data.columns))
barmode_option = st.sidebar.radio("Bar Mode (if applicable)", options=["overlay", "group"], index=0)

# Display selected plot dynamically based on the user’s input
st.header(f"Analysis of Mental Health with respect to {selected_feature}")
if plot_type == "Box Plot":
    if axis_swap:
        fig = px.box(data, x="MentalHealthCategory", y=selected_feature, color=color_option if color_option != "None" else None,
                     title=f"Mental Health by {selected_feature}")
    else:
        fig = px.box(data, x=selected_feature, y="MentalHealthCategory", color=color_option if color_option != "None" else None,
                     title=f"Mental Health by {selected_feature}")
elif plot_type == "Histogram":
    fig = px.histogram(data, x="MentalHealthCategory", color=selected_feature, barmode=barmode_option,
                       title=f"Mental Health Distribution with {selected_feature}")
elif plot_type == "Scatter Plot":
    if axis_swap:
        fig = px.scatter(data, x="MentalHealthCategory", y=selected_feature, color=color_option if color_option != "None" else None,
                         title=f"Scatter Plot of {selected_feature} and Mental Health")
    else:
        fig = px.scatter(data, x=selected_feature, y="MentalHealthCategory", color=color_option if color_option != "None" else None,
                         title=f"Scatter Plot of {selected_feature} and Mental Health")
elif plot_type == "Violin Plot":
    if axis_swap:
        fig = px.violin(data, x="MentalHealthCategory", y=selected_feature, color=color_option if color_option != "None" else None,
                        title=f"Violin Plot of {selected_feature} and Mental Health")
    else:
        fig = px.violin(data, x=selected_feature, y="MentalHealthCategory", color=color_option if color_option != "None" else None,
                        title=f"Violin Plot of {selected_feature} and Mental Health")
elif plot_type == "Density Heatmap":
    fig = px.density_heatmap(data, x=selected_feature, y="MentalHealthCategory", color_continuous_scale='Viridis',
                             title=f"Density Heatmap of {selected_feature} and Mental Health")

st.plotly_chart(fig, use_container_width=True)
st.markdown(f'<p class="insight-text">Insight: Observe how {selected_feature} influences mental health scores. Experiment with different plot types for a deeper view.</p>', unsafe_allow_html=True)

# Add correlation analysis section
st.header("Correlation Analysis")
@st.cache_data
def compute_correlation_matrix(data):
    corr_features = ['MentalHealthCategoryNum', 'PhysicalHealth', 'BMI', 'SleepTime', 'Smoking', 'AlcoholDrinking', 
                     'Stroke', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
    return data[corr_features].corr()

correlation_matrix = compute_correlation_matrix(data)
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
st.markdown('<p class="summary-text">Key Observations:</p>', unsafe_allow_html=True)
st.markdown("""
- **Demographics**: Certain age groups and BMI categories exhibit distinct patterns in mental health scores.
- **Lifestyle Factors**: Physical activity, smoking, and alcohol consumption have significant associations with mental health.
- **Chronic Conditions**: Conditions like stroke and kidney disease tend to correlate with poorer mental health.
- **General Health**: Self-reported general health shows strong links to mental health outcomes.
""")

# Footer with a call-to-action and additional icons
st.markdown("---")
st.markdown("**Thank you for using the Mental Health Data Analysis Dashboard!** Explore further by adjusting the settings.")

