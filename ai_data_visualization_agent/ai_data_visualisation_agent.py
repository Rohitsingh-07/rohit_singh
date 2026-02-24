import os
import re
import base64
import requests
import ollama
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from typing import Optional, List, Any, Tuple
from e2b_code_interpreter import Sandbox


# ------------------------------------------------
# Get Local Ollama Models
# ------------------------------------------------
def get_local_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return [model['name'] for model in response.json().get('models', [])]
        return []
    except:
        return []


# ------------------------------------------------
# Extract Python Code from LLM Response
# ------------------------------------------------
def match_code_blocks(llm_response: str) -> str:
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    match = pattern.search(llm_response)
    return match.group(1) if match else ""


# ------------------------------------------------
# Execute Code in E2B Sandbox
# ------------------------------------------------
def execute_code(sandbox: Sandbox, code: str):
    with st.spinner("Executing code in sandbox..."):
        result = sandbox.run_code(code)
        if result.error:
            st.error(f"Execution Error: {result.error}")
            return None
        return result.results


# ------------------------------------------------
# Chat with Local Ollama
# ------------------------------------------------
def chat_with_llm(user_message: str, dataset_path: str) -> Tuple[str, str]:

    # cols = pd.read_csv(dataset_path).columns.tolist()

    system_prompt = f"""
You are a Python data scientist. The dataset is located at: {dataset_path}
Instructions:
- Use pandas and matplotlib or seaborn for visualization.
- IMPORTANT: Always add data labels (values) on top of bars or points in the chart.
- IMPORTANT: ROUND all data labels to 0 or 1 decimal place (e.g., use f-strings like f'{{val:.1f}}' or ax.bar_label with labels).
- IMPORTANT: You MUST always end your code with 'plt.show()' to display the plot.
- Always return explanation text and wrap Python code inside ```python ``` blocks.
"""

    response = ollama.chat(
        model=st.session_state.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    response_content = response["message"]["content"]
    python_code = match_code_blocks(response_content)

    return python_code, response_content


# ------------------------------------------------
# Streamlit App
# ------------------------------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("📊 AI Data Visualization Agent")

    if "model_name" not in st.session_state:
        st.session_state.model_name = ""

    if "e2b_api_key" not in st.session_state:
        st.session_state.e2b_api_key = ""

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Local Ollama Setup")

        models = get_local_ollama_models()

        if models:
            st.session_state.model_name = st.selectbox(
                "Select Local Model",
                options=models
            )
            st.success("Ollama Connected ✅")
        else:
            st.error("Ollama not detected. Run: ollama serve")

        st.divider()

        st.header("E2B Sandbox")
        st.session_state.e2b_api_key = st.text_input(
            "Enter E2B API Key",
            type="password"
        )

    # ---------------- File Upload ----------------
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file and st.session_state.model_name:

        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        query = st.text_area(
            "What visualization would you like?",
            "Create a histogram of numeric columns."
        )

        if st.button("Generate Visualization"):

            if not st.session_state.e2b_api_key:
                st.error("Please enter your E2B API Key.")
                return

            with Sandbox(api_key=st.session_state.e2b_api_key) as sandbox:

                dataset_path = f"./{uploaded_file.name}"
                sandbox.files.write(dataset_path, uploaded_file.getvalue())

                python_code, llm_response = chat_with_llm(query, dataset_path)

                # st.subheader("🧠 AI Explanation")
                # st.write(llm_response)
                with st.expander("🧠 View AI Explanation", expanded=True): st.write(llm_response)

                if python_code:
                    with st.expander("🔍 View Generated Python Code"):
                        st.code(python_code, language="python")

                    results = execute_code(sandbox, python_code)

                    if results:
                        for res in results:

                            # If image result
                            if hasattr(res, "png") and res.png:
                                img_data = base64.b64decode(res.png)
                                st.image(
                                    Image.open(BytesIO(img_data)),
                                    caption="Generated Visualization"
                                )

                            # If dataframe
                            elif isinstance(res, (pd.DataFrame, pd.Series)):
                                st.dataframe(res)

                            else:
                                st.write(res)

                    else:
                        st.warning("No results returned.")

                else:
                    st.warning("No Python code detected in LLM response.")


if __name__ == "__main__":
    main()