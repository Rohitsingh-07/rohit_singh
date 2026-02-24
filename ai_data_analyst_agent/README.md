# 📊 AI Data Analysis Agent

<img width="1918" height="911" alt="image" src="https://github.com/user-attachments/assets/18017676-d8e8-4497-b00b-df4fc59e12dd" />

An AI-powered Data Analyst Agent built using the Agno Agent framework, Streamlit, and **locally hosted Ollama model**. This application allows users to upload CSV or Excel files and analyze their data using natural language queries - automatically converting questions into SQL, executing them, and returning accurate insights.

## 🚀 Overview

This project enables seamless data exploration through a conversational interface. Users can upload datasets and ask business questions in plain English without writing SQL. The system processes the data, runs structured queries, and returns computed results with transparency.

It demonstrates applied skills in:

- LLM-powered agents
- Natural language → SQL conversion
- Data engineering workflows
- Tool-integrated AI systems
- Interactive analytics applications

---

🧠 Tech Stack

- LLM Runtime: Local Ollama Model  
- Agent Framework: Agno Agent
- Database Engine: DuckDB
- Data Processing: Pandas
- Frontend/UI: Streamlit
- Language: Python

### ✨ Features

- 📤 **File Upload Support**: 
  - Upload CSV and Excel files
  - Automatic data type detection and schema inference
  - Support for multiple file formats

- 💬 **Natural Language Queries**: 
  - Convert natural language questions into SQL queries
  - Get instant answers about your data
  - No SQL knowledge required

- 🔍 **Advanced Analysis**:
  - Perform complex data aggregations
  - Filter and sort data
  - Generate statistical summaries
  - Create data visualizations

- 🎯 **Interactive UI**:
  - User-friendly Streamlit interface
  - Real-time query processing
  - Clear result presentation

## How to Run

1. **Install Dependencies**
   ```bash
   pip install streamlit pandas duckdb requests agno
   ```

2. **Run the Application**
   ```bash
   streamlit run ai_data_analyst.py
   ```

## Usage

1. Launch the application using the command above
2. Upload your CSV or Excel file through the Streamlit interface
3. Select your available local Ollama model
4. Ask questions about your data in natural language
5. View the results and generated visualizations

