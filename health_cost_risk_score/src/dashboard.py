import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from load_data import load_and_clean_data

# Set page config for a premium look
st.set_page_config(
    page_title="Health Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a better aesthetic
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #E22121;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #007bff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_and_clean_data()

@st.cache_resource
def get_model():
    model_path = 'outputs/model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def main():
    st.title("🏥 Health Cost Risk Copilot")
    st.markdown("---")

    # Load resources
    df = get_data()
    model = get_model()

    if model is None:
        st.error("Model not found! Please run 'python -m src.train_model' first.")
        return

    # Sidebar for Patient Input
    st.sidebar.header("📝 New Patient Simulation")
    st.sidebar.info("Enter details to predict risk score")
    
    with st.sidebar:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 15.0, 60.0, 25.0)
        children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
        sex = st.radio("Sex", ["male", "female"])
        smoker = st.radio("Smoker", ["yes", "no"])
        region = st.selectbox("Region", df['region'].unique())
        
        if st.button("Calculate Risk Score"):
            # Create a dataframe for prediction
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })
            
            # Predict
            prob = model.predict_proba(input_data)[0][1]
            risk_score = round(prob * 100, 2)
            
            st.markdown("---")
            st.metric("Risk Score", f"{risk_score}/100")
            
            if risk_score > 50:
                st.warning("High Risk of Top 10% Costs")
            else:
                st.success("Lower Risk Category")

    # Main Dashboard Content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Dataset Overview")
        # Distribution of Charges
        fig_charges = px.histogram(
            df, x="charges", color="high_cost", 
            nbins=50, title="Distribution of Medical Charges",
            labels={'high_cost': 'Is High Cost?'},
            color_discrete_map={0: '#6c757d', 1: '#dc3545'}
        )
        st.plotly_chart(fig_charges, use_container_width=True)

    with col2:
        st.subheader("💡 Key Insights")
        if os.path.exists('outputs/insights.txt'):
            with open('outputs/insights.txt', 'r') as f:
                insights_txt = f.read()
            st.text_area("Findings", insights_txt, height=250)
        else:
            st.info("Run explain script to see insights.")

    st.markdown("---")
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🧬 Factors and Risk")
        # Box plot BMI vs High Cost
        fig_bmi = px.box(
            df, x="high_cost", y="bmi", color="high_cost",
            title="BMI vs High Cost Status",
            labels={'high_cost': 'Is High Cost?'},
            color_discrete_map={0: '#6c757d', 1: '#dc3545'}
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    with col4:
        st.subheader("🚬 Smoking & Cost")
        df_smoke = df.groupby(['smoker', 'high_cost']).size().reset_index(name='count')
        fig_smoke = px.bar(
            df_smoke, x="smoker", y="count", color="high_cost",
            title="Impact of Smoking on High Cost Chances",
            barmode="group",
            color_discrete_map={0: '#6c757d', 1: '#dc3545'}
        )
        st.plotly_chart(fig_smoke, use_container_width=True)

if __name__ == "__main__":
    main()