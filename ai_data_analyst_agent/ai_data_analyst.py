import tempfile
import csv
import streamlit as st
import pandas as pd
import requests
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools

# Function to fetch local models from your Ollama instance
def get_local_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return [model['name'] for model in response.json().get('models', [])]
        return []
    except:
        return []

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent")
# Initialize Session State for History
if "query_history" not in st.session_state:
    st.session_state.query_history = []
# Sidebar - Simplified for Local Use Only
with st.sidebar:
    st.header("Local Model Settings")
    local_models = get_local_ollama_models()
    
    if local_models:
        selected_model_id = st.selectbox("Select Your Ollama Model:", local_models)
        st.success(f"Ollama is Active")
        st.info("💡 Reminder: Models like 'llama3.1' or 'qwen2.5-coder' work best for SQL tools.")
    else:
        selected_model_id = None
        st.error("Ollama not detected! Please ensure the Ollama app is running on your machine.")

    if st.button("Clear History"):
        st.session_state.query_history = []
        st.rerun()
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and selected_model_id:
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.subheader("Uploaded Data:")
        st.dataframe(df.head(5), use_container_width=True)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        
        # Initialize DuckDbTools
        duckdb_tools = DuckDbTools()
        
        # Load the CSV file into DuckDB as a table
        duckdb_tools.load_local_csv_to_table(
            path=temp_path,
            table="uploaded_data",
        )
        
        # Initialize the Agent with DuckDB and Pandas tools
        data_analyst_agent = Agent(
            model=Ollama(id=selected_model_id),
            tools=[duckdb_tools, PandasTools()],
            system_message=(
                "You are a strict Data Analyst. Follow this process:\n"
                "1. Always use the 'duckdb_tools' to run a SQL query on 'uploaded_data'.\n"
                "2. DO NOT create placeholder tables with blanks like '____'.\n"
                "3. If the tool returns data, display it. If the tool fails, explain why.\n"
                "4. Your final answer must include the actual numbers returned by the tool."
            ),
            markdown=True,
        )
        
        user_query = st.text_area("Ask a query about the data:")
        if st.button('Submit Query'):
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from the agent
                        response = data_analyst_agent.run(user_query)

                        st.write("### Analysis Summary")
                        st.markdown(response.content if hasattr(response, 'content') else str(response))

                        # Display Raw Tool Results
                        if hasattr(response, 'messages'):
                            for message in response.messages:
                                if message.role == "tool":
                                    with st.expander(f"View Data Result ({message.tool_name})"):
                                        try:
                                            st.write(message.content)
                                        except Exception:
                                            st.text(message.content)

                except Exception as e:
                    st.error(f"Error: {e}")
