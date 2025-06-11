# app.py
import streamlit as st
import os
from utils.summarizer import TextSummarizer

# --- Configuration ---
# You can load your model once when the app starts
# Use st.cache_resource to avoid reloading the model every time the app re-runs
@st.cache_resource
def load_summarizer():
    # You can change the model name here based on your needs and resources
    # "facebook/bart-large-cnn" is a good general-purpose choice for news summarization
    # "t5-base" is another option, often smaller and faster
    # "google/pegasus-xsum" is specialized for highly abstractive summaries
    model_name = "facebook/bart-large-cnn"
    st.info(f"Loading summarization model: **{model_name}**. This may take a moment...")
    try:
        summarizer = TextSummarizer(model_name=model_name)
        st.success("Model loaded successfully!")
        return summarizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop() # Stop the app if the model can't be loaded

# Load the summarizer when the app starts
summarizer = load_summarizer()

# --- Streamlit UI ---
st.set_page_config(page_title="LLM Text Summarizer", layout="centered")

st.title("📄 LLM-Powered Text Summarizer")
st.markdown("""
Welcome to the Text Summarizer! Enter your text below, and an advanced Large Language Model
will generate a concise summary for you.
""")

# Input method selection
input_method = st.radio(
    "How would you like to provide text?",
    ("Type/Paste Text", "Upload Text File"),
    index=0 # Default to typing/pasting
)

input_text = ""

if input_method == "Type/Paste Text":
    input_text = st.text_area(
        "Enter your text here:",
        height=300,
        placeholder="Paste your article, document, or any text you want to summarize..."
    )
elif input_method == "Upload Text File":
    uploaded_file = st.file_uploader("Choose a text file (.txt)", type="txt")
    if uploaded_file is not None:
        # Read the uploaded file
        try:
            input_text = uploaded_file.read().decode("utf-8")
            st.success("File uploaded successfully!")
            st.text_area("Content of the uploaded file:", value=input_text, height=200, disabled=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Summary options
st.sidebar.header("Summary Options")
max_length = st.sidebar.slider("Maximum summary length", 50, 500, 150)
min_length = st.sidebar.slider("Minimum summary length", 10, 200, 30)
do_sample = st.sidebar.checkbox("Generate diverse summary (less deterministic)", False)

if st.button("Summarize Text"):
    if not input_text.strip():
        st.warning("Please enter some text or upload a file to summarize.")
    else:
        with st.spinner("Generating summary... This might take a few moments."):
            summary = summarizer.summarize(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )
            
            st.subheader("🎉 Generated Summary")
            if "An error occurred" in summary: # Check for error message from summarizer
                st.error(summary)
            else:
                st.write(summary)

st.markdown("---")
st.markdown("Built with ❤️ by Your Name/Team and powered by Hugging Face Transformers.")